"""
Cognitive Architecture Module — Amphibious Agent Framework.

A framework for building dual-mode agents that can operate in both
LLM-driven (agent) and deterministic (workflow) modes, with automatic
fallback between them.

Architecture Layers
-------------------

**Abstraction Layer:**
- Exposure: Base abstraction for layered data exposure
- LayeredExposure: Supports progressive disclosure (summary + per-item details)
- EntireExposure: Summary only, no per-item detail queries
- Context: Base class for agent context with automatic Exposure field detection

**Implementation Layer - Context:**
- Step: A single execution step with content, status, and result
- Skill: A skill definition following SKILL.md format
- CognitiveTools: Tool management (EntireExposure implementation)
- CognitiveSkills: Skill management (LayeredExposure implementation)
- CognitiveHistory: Execution history with layered memory (LayeredExposure implementation)
- CognitiveContext: The default cognitive context combining all above

**Implementation Layer - Worker:**
- CognitiveWorker: The "think" unit of one observe-think-act cycle.
  Observation is injected before calling arun(); action is executed by the
  agent after arun() returns. Multiple LLM rounds may occur within a single
  arun() call when cognitive policies (acquiring, rehearsal, reflection) fire.

**Orchestration Layer:**
- AmphibiousAutoma: Dual-mode agent engine (agent mode + workflow mode)
  - think_unit: Descriptor for declaring think steps (used in on_agent)
  - step / steps: Helpers for declaring workflow steps (used in on_workflow)
- ErrorStrategy: Error handling strategies (RAISE, IGNORE, RETRY)

Example
-------
>>> class MyAgent(AmphibiousAutoma[CognitiveContext]):
...     main_think = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=20)
...     async def on_agent(self, ctx):
...         await self.main_think
...
>>> ctx = await MyAgent(llm=llm).arun(goal="Complete the task")
"""
from ._context import (
    # Abstraction layer
    Exposure,
    LayeredExposure,
    EntireExposure,
    Context,
    # Implementation layer - Context components
    Step,
    Skill,
    CognitiveTools,
    CognitiveSkills,
    CognitiveHistory,
    CognitiveContext,
)
from ._cognitive_worker import (
    # Worker
    CognitiveWorker,
    # Sentinel
    _DELEGATE,
    # Workflow helpers
    step
)
from ._amphibious_automa import (
    # Orchestration
    AmphibiousAutoma,
    AgentTrace,
    # Think unit descriptor
    think_unit,
    ThinkUnitDescriptor,
)
from ._type import (
    # Worker data structures
    DetailRequest,
    ToolArgument,
    StepToolCall,
    # Workflow mode
    WorkflowDecision,
    WorkflowStep,
    AgentFallback,
    # Action result data structures
    ErrorStrategy,
    ActionResult,
    ActionStepResult,
    ToolResult,
    # Trace data models
    TraceStep,
    RunConfig,
    RecordedToolCall,
    StepOutputType,
)

# Deprecated alias
AgentAutoma = AmphibiousAutoma

__all__ = [
    # Abstraction layer
    "Exposure",
    "LayeredExposure",
    "EntireExposure",
    "Context",

    # Implementation layer - Context components
    "Step",
    "Skill",
    "CognitiveTools",
    "CognitiveSkills",
    "CognitiveHistory",
    "CognitiveContext",

    # Implementation layer - Worker
    "CognitiveWorker",
    "_DELEGATE",
    "step",

    # Orchestration layer
    "AmphibiousAutoma",
    "AgentAutoma",  # deprecated alias
    "AgentTrace",
    "think_unit",
    "ThinkUnitDescriptor",

    # Worker data structures
    "DetailRequest",
    "ToolArgument",
    "StepToolCall",
    # Workflow mode
    "WorkflowDecision",
    "WorkflowStep",
    "AgentFallback",
    # Action result data structures
    "ErrorStrategy",
    "ActionResult",
    "ActionStepResult",
    "ToolResult",
    # Trace data models
    "TraceStep",
    "RunConfig",
    "RecordedToolCall",
    "StepOutputType",
]
