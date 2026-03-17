"""
Workflow Module — data models and builder for agent execution traces.

This module contains:
- Data models: TraceStep, RecordedToolCall, RunConfig for trace recording.
- WorkflowBuilder: stack-based builder that captures phase-annotated trace steps.
- Utilities: observation fingerprinting.

The replay engine (Workflow, SequentialBlock, LoopBlock) has been replaced by the
generator-based workflow mode (cognition_workflow / _run_workflow) in AgentAutoma.

Key Types
---------
- TraceStep: One observe-think-act cycle record
- RunConfig: Captured run() parameters (for trace metadata)
- RecordedToolCall: Complete record of one tool invocation
- WorkflowBuilder: Captures raw trace data from agent execution
"""

from __future__ import annotations
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from bridgic.core.cognitive._agent_automa import AgentAutoma
    


################################################################################################################
# Trace-level data structures
################################################################################################################

class StepOutputType(str, Enum):
    """Discriminator for the kind of output a trace step produced."""
    TOOL_CALLS = "tool_calls"
    STRUCTURED = "structured"
    CONTENT_ONLY = "content_only"


class RecordedToolCall(BaseModel):
    """A complete record of one tool invocation."""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any


class TraceStep(BaseModel):
    """Record of one observe-think-act cycle."""
    model_config = ConfigDict(extra="forbid")

    name: str                                       # worker class name
    step_content: str                               # LLM reasoning content
    tool_calls: List[RecordedToolCall] = Field(default_factory=list)
    finished: bool = False                          # worker signalled finish?
    observation_hash: Optional[str] = None          # structural fingerprint
    output_type: StepOutputType = StepOutputType.TOOL_CALLS
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_class: Optional[str] = None


################################################################################################################
# Run configuration (captured from self.run())
################################################################################################################

class RunConfig(BaseModel):
    """Parameters captured from self.run(), used for trace metadata."""
    model_config = ConfigDict(extra="forbid")

    worker_class: str
    worker_thinking_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    max_attempts: int = 1
    on_error: str = "raise"


################################################################################################################
# Utilities
################################################################################################################

def observation_fingerprint(obs: Any) -> Optional[str]:
    """Compute a stable hash fingerprint of an observation value.

    Used for divergence detection during replay.  Returns None for
    None observations.
    """
    if obs is None:
        return None
    try:
        serialized = json.dumps(obs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(obs)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _detect_iteration_boundary(
    steps: List[dict],
) -> Tuple[List[dict], int]:
    """Detect repeating pattern in a flat step list and extract the first iteration.

    Uses tool-name signature sequences with KMP failure function to find the
    minimal repeating period.  Returns ``(first_iteration_steps, iteration_count)``.
    If no clean repetition is found, returns ``(steps, 1)``.
    """
    if not steps:
        return steps, 0

    # Build signature sequence from tool names
    signatures: List[Tuple[str, ...]] = []
    for s in steps:
        tool_calls = s.get("tool_calls", [])
        if tool_calls:
            sig = tuple(tc["tool_name"] for tc in tool_calls)
        else:
            sig = ("__content__",)
        signatures.append(sig)

    # KMP failure function
    n = len(signatures)
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and signatures[i] != signatures[j]:
            j = fail[j - 1]
        if signatures[i] == signatures[j]:
            j += 1
        fail[i] = j

    period = n - fail[-1]
    if period < n and n % period == 0:
        return steps[:period], n // period

    # No clean repetition detected
    return steps, 1


################################################################################################################
# Workflow Builder — trace collection only
################################################################################################################

class _PhaseAccumulator:
    """Accumulates trace steps for a single structural phase (sequential or loop)."""

    __slots__ = ("phase_type", "goal", "steps", "run_config")

    def __init__(self, phase_type: str, goal: Optional[str] = None):
        self.phase_type: str = phase_type  # "sequential" | "loop"
        self.goal: Optional[str] = goal
        self.steps: List[dict] = []
        self.run_config: Optional[dict] = None


class WorkflowBuilder:
    """Stack-based builder that captures phase-annotated trace steps.

    ``begin_phase()`` / ``end_phase()`` bracket a structural phase.
    ``record_step()`` routes step data to the active phase (or orphan list).
    ``set_run_config()`` attaches run() parameters to the current phase.
    ``suppress()`` temporarily stops recording.
    ``build()`` returns the collected phases as structured dicts for inspection.
    """

    def __init__(self):
        self._stack: List[_PhaseAccumulator] = []
        self._completed: List[_PhaseAccumulator] = []
        self._orphan_steps: List[dict] = []

    def begin_phase(self, phase_type: str, goal: Optional[str] = None) -> None:
        self._stack.append(_PhaseAccumulator(phase_type, goal=goal))

    def end_phase(self) -> None:
        if self._stack:
            self._completed.append(self._stack.pop())

    def record_step(self, step_data: dict) -> None:
        """Route a trace step to the active phase or the orphan list."""
        if self._stack:
            self._stack[-1].steps.append(step_data)
        else:
            self._orphan_steps.append(step_data)

    def set_run_config(self, config: dict) -> None:
        """Attach run() parameters to the current (topmost) phase."""
        if self._stack:
            self._stack[-1].run_config = config

    @contextmanager
    def phase(self, phase_type: str, goal: Optional[str] = None):
        """Context manager that brackets a structural phase."""
        self.begin_phase(phase_type, goal=goal)
        try:
            yield
        finally:
            self.end_phase()

    def build(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return collected trace data as a structured dict.

        Returns a dict with 'phases' (list of phase dicts) and 'metadata'.
        Each phase dict has: phase_type, goal, run_config, steps.
        """
        phases = []
        for phase in self._completed:
            trace_steps = [
                TraceStep(
                    name=s["name"],
                    step_content=s.get("step_content", ""),
                    tool_calls=[
                        RecordedToolCall(**tc) for tc in s.get("tool_calls", [])
                    ],
                    finished=s.get("finished", False),
                    observation_hash=s.get("observation_hash"),
                    output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                    structured_output=s.get("structured_output"),
                    structured_output_class=s.get("structured_output_class"),
                )
                for s in phase.steps
            ]
            run_config = RunConfig(**(phase.run_config or {"worker_class": "unknown"}))
            phases.append({
                "phase_type": phase.phase_type,
                "goal": phase.goal,
                "run_config": run_config,
                "steps": trace_steps,
            })

        orphan_steps = [
            TraceStep(
                name=s["name"],
                step_content=s.get("step_content", ""),
                tool_calls=[RecordedToolCall(**tc) for tc in s.get("tool_calls", [])],
                finished=s.get("finished", False),
                observation_hash=s.get("observation_hash"),
                output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                structured_output=s.get("structured_output"),
                structured_output_class=s.get("structured_output_class"),
            )
            for s in self._orphan_steps
        ]

        return {
            "phases": phases,
            "orphan_steps": orphan_steps,
            "metadata": metadata or {},
        }
