"""
Execution trace and divergence detection for amphibious agent execution.

TraceStep records the complete state of a single observe-think-act cycle.
DivergenceDetector compares recorded vs actual execution for mode switching.
"""
import hashlib
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bridgic.core.cognitive._workflow import WorkflowToolCall


class TraceStep(BaseModel):
    """Complete record of a single execution step."""

    index: int
    name: str
    observation: Optional[str] = None
    observation_hash: str = ""
    step_content: str = ""
    tool_calls: List[WorkflowToolCall] = Field(default_factory=list)
    tool_results: List[Any] = Field(default_factory=list)
    result_hashes: List[str] = Field(default_factory=list)
    source: Literal["agent", "workflow"] = "agent"

    @staticmethod
    def compute_hash(value: Any) -> str:
        """Compute a stable hash for divergence detection."""
        text = str(value) if value is not None else ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def compute_observation_hash(self) -> str:
        """Compute and set observation_hash from observation."""
        self.observation_hash = self.compute_hash(self.observation)
        return self.observation_hash

    def compute_result_hashes(self) -> List[str]:
        """Compute and set result_hashes from tool_results."""
        self.result_hashes = [self.compute_hash(r) for r in self.tool_results]
        return self.result_hashes


class DivergenceLevel(Enum):
    """How much actual execution diverged from recorded."""

    MATCH = "match"
    MINOR = "minor"  # Continue but log warning (e.g. text differs, structure same)
    MAJOR = "major"  # Switch to agent mode


class DivergenceDetector:
    """Compare recorded TraceStep against actual execution results.

    Phase 1 uses hash-based exact comparison.
    Future phases may add LLM-based semantic comparison.
    """

    def check(
        self,
        recorded: TraceStep,
        actual_observation: Optional[str],
        actual_tool_names: List[str],
        actual_results: List[Any],
    ) -> DivergenceLevel:
        """Compare actual execution against a recorded step.

        Parameters
        ----------
        recorded : TraceStep
            The previously recorded step.
        actual_observation : Optional[str]
            The observation from the current execution.
        actual_tool_names : List[str]
            Tool names actually called.
        actual_results : List[Any]
            Results from actual tool execution.

        Returns
        -------
        DivergenceLevel
            MATCH, MINOR, or MAJOR divergence.
        """
        # Check tool name pattern match first (most critical)
        recorded_names = [tc.tool_name for tc in recorded.tool_calls]
        if recorded_names != actual_tool_names:
            return DivergenceLevel.MAJOR

        # Check observation hash
        obs_hash = TraceStep.compute_hash(actual_observation)
        obs_match = (obs_hash == recorded.observation_hash)

        # Check result hashes
        actual_hashes = [TraceStep.compute_hash(r) for r in actual_results]
        results_match = (actual_hashes == recorded.result_hashes)

        if obs_match and results_match:
            return DivergenceLevel.MATCH

        # Tool names match but data differs — minor divergence
        return DivergenceLevel.MINOR

    def check_tool_pattern(
        self,
        recorded_names: List[str],
        actual_names: List[str],
    ) -> bool:
        """Check if tool call patterns match (for LoopBlock iteration matching).

        For loops, we only require the same set of tool names in the same order,
        not identical arguments.
        """
        return recorded_names == actual_names
