"""
Cognitive Architecture — Consolidated Data Models.

All Pydantic models, dataclasses, enums, and type aliases used across the
cognitive module are gathered here for a single source of truth.

Sections are annotated with the module(s) that consume each model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator
from pydantic.functional_validators import BeforeValidator


#################################################################################################################
# Context layer models  (used by: _context.py)
#################################################################################################################

class Step(BaseModel):
    """A single execution step with content, result, and metadata.

    Used by: _context.py (CognitiveHistory, CognitiveContext.add_info),
             _agent_automa.py (_action, _record_trace_step)
    """
    model_config = ConfigDict(extra="forbid")

    content: str = ""
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[bool] = None  # 兼容旧测试，可选


class Skill(BaseModel):
    """A skill definition following SKILL.md format.

    Used by: _context.py (CognitiveSkills)
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


#################################################################################################################
# Worker layer models  (used by: _cognitive_worker.py, _agent_automa.py)
#################################################################################################################

class DetailRequest(BaseModel):
    """Request for detailed information about a specific item in a LayeredExposure field.

    Used by: _cognitive_worker.py (acquiring policy)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["field", "index"],
            "additionalProperties": False,
        }
    )
    field: str = Field(description="Name of the field to get details from (e.g., 'cognitive_history', 'skills')")
    index: int = Field(description="0-based index of the item to get details for")


class ToolArgument(BaseModel):
    """A single tool argument as name-value pair.

    Used by: _cognitive_worker.py (StepToolCall, step()), _agent_automa.py (action phase)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["name", "value"],
            "additionalProperties": False,
        }
    )
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value as string")

    @field_validator('value', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        return str(v) if not isinstance(v, str) else v


class StepToolCall(BaseModel):
    """A single tool call specification.

    Used by: _cognitive_worker.py (ThinkModel output, step()), _agent_automa.py (action phase)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool", "tool_arguments"],
            "additionalProperties": False,
        }
    )
    tool: str = Field(description="Name of the tool to call")
    tool_arguments: List[ToolArgument] = Field(
        description="Arguments as list of name-value pairs, e.g., [{name: 'city', value: 'Beijing'}]"
    )


class _ThinkBase(BaseModel):
    """Unified base for all dynamically-generated ThinkModel variants.

    Factory (_create_think_model) adds: output, details, rehearsal,
    reflection — all optional and conditional on configuration.

    Used by: _cognitive_worker.py (_create_think_model)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["step_content"],
            "additionalProperties": False,
        }
    )

    step_content: str = Field(
        default="",
        description="Description of what to do in this step, or your analysis/reasoning"
    )
    finish: bool = Field(
        default=False,
        description="Set True when your current sub-task is FULLY complete and no more steps are needed."
    )

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)


def _coerce_none_to_list(v: Any) -> list:
    """Coerce None to empty list for field validation."""
    return [] if v is None else v


################################################################################################################
# Workflow mode models  (used by: _cognitive_worker.py, _agent_automa.py)
################################################################################################################

class WorkflowDecision(BaseModel):
    """Single-step deterministic decision for workflow mode.

    Used by: _cognitive_worker.py (step() helper), _agent_automa.py (_run_workflow)
    """
    model_config = ConfigDict(extra="forbid")

    step_content: str = ""
    output: List[StepToolCall] = Field(default_factory=list)


@dataclass
class WorkflowStep:
    """Yielded by cognition_workflow() for deterministic execution.

    Used by: _agent_automa.py (_run_workflow)
    """
    worker: Any  # CognitiveWorker
    decision: WorkflowDecision


@dataclass
class AgentFallback:
    """Yielded by cognition_workflow() to fall back to agent mode.

    Used by: _agent_automa.py (_run_workflow)
    """
    worker: Any  # CognitiveWorker
    goal: str
    tools: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    max_attempts: int = 1


################################################################################################################
# Action result models  (used by: _agent_automa.py)
################################################################################################################

class ErrorStrategy(Enum):
    """Error handling strategy for worker execution via ``self.run()``.

    Used by: _agent_automa.py (run method)
    """
    RAISE = "raise"    # Re-raise exceptions (default)
    IGNORE = "ignore"  # Silently ignore exceptions
    RETRY = "retry"    # Retry up to max_retries times


class ActionStepResult(BaseModel):
    """Result of executing one tool in the action phase.

    Used by: _agent_automa.py (action_tool_call)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool_id", "tool_name", "tool_arguments", "tool_result", "success"],
            "additionalProperties": False,
        }
    )
    tool_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    success: bool = True
    error: Optional[str] = None


class ActionResult(BaseModel):
    """Overall result of the action phase (one or more tool executions).

    Used by: _agent_automa.py (action_tool_call, _record_trace_step)
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["results"],
            "additionalProperties": False,
        }
    )
    results: List[ActionStepResult]


################################################################################################################
# Trace data models  (used by: _agent_automa.py — AgentTrace)
################################################################################################################

class StepOutputType(str, Enum):
    """Discriminator for the kind of output a trace step produced.

    Used by: _agent_automa.py (AgentTrace, _record_trace_step)
    """
    TOOL_CALLS = "tool_calls"
    STRUCTURED = "structured"
    CONTENT_ONLY = "content_only"


class RecordedToolCall(BaseModel):
    """A complete record of one tool invocation.

    Used by: _agent_automa.py (AgentTrace.build)
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any
    success: bool = True
    error: Optional[str] = None


class TraceStep(BaseModel):
    """Record of one observe-think-act cycle.

    Used by: _agent_automa.py (AgentTrace.build)
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    step_content: str
    tool_calls: List[RecordedToolCall] = Field(default_factory=list)
    observation: Optional[str] = None
    observation_hash: Optional[str] = None
    output_type: StepOutputType = StepOutputType.TOOL_CALLS
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_class: Optional[str] = None


class RunConfig(BaseModel):
    """Parameters captured from self.run(), used for trace metadata.

    Used by: _agent_automa.py (AgentTrace)
    """
    model_config = ConfigDict(extra="forbid")

    worker_class: str
    worker_thinking_prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    max_attempts: int = 1
    on_error: str = "raise"


################################################################################################################
# Utility functions  (used by: _agent_automa.py)
################################################################################################################

def observation_fingerprint(obs: Any) -> Optional[str]:
    """Compute a stable hash fingerprint of an observation value.

    Used for divergence detection during replay. Returns None for
    None observations.
    """
    if obs is None:
        return None
    try:
        serialized = json.dumps(obs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(obs)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
