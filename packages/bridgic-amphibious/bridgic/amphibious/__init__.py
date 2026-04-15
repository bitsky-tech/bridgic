"""
Amphibious Agent Framework — Dual-Mode Agent Orchestration.

A framework for building agents that can operate in both LLM-driven (agent)
and deterministic (workflow) modes, with automatic fallback between them.

Architecture Layers
-------------------

**Abstraction Layer (Data Exposure):**
- Exposure: Base abstraction for field-level data management
- LayeredExposure: Supports progressive disclosure (summary + per-item details)
- EntireExposure: Summary only, no per-item detail queries
- Context: Base class for agent context with automatic Exposure field detection

**Implementation Layer — Context:**
- Step: A single execution step with content, result, and metadata
- Skill: A skill definition following SKILL.md format
- CognitiveTools: Tool management (EntireExposure)
- CognitiveSkills: Skill management with progressive disclosure (LayeredExposure)
- CognitiveHistory: Execution history with layered memory (LayeredExposure)
- CognitiveContext: The default cognitive context combining all above

**Implementation Layer — Worker (Think Unit):**
- CognitiveWorker: Pure thinking unit of one observe-think-act cycle.
  Cognitive policies (acquiring, rehearsal, reflection) enable multi-round
  thinking within a single call.

**Orchestration Layer:**
- AmphibiousAutoma: Dual-mode agent engine (agent mode + workflow mode)
  - think_unit: Descriptor for declaring think units (used in on_agent)
  - ActionCall, HumanCall, AgentCall: Workflow yield types (used in on_workflow)
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
)
from ._amphibious_automa import (
    # Orchestration
    AmphibiousAutoma,
    AgentTrace,
    # Think unit descriptor
    think_unit,
    ThinkUnitDescriptor,
)
from .scaffold import create_project
from .builtin_tools import request_human_tool

from ._type import (
    # Worker data structures
    RunMode,
    DetailRequest,
    ToolArgument,
    StepToolCall,
    # Workflow mode yield types
    HUMAN_INPUT_EVENT_TYPE,
    WorkflowDecision,
    ActionCall,
    HumanCall,
    AgentCall,
    # Action result data structures
    ErrorStrategy,
    ActionResult,
    ActionStepResult,
    ToolResult,
    # Trace data models
    TraceStep,
    RecordedToolCall,
    StepOutputType,
)

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

    # Orchestration layer
    "AmphibiousAutoma",
    "AgentTrace",
    "think_unit",
    "ThinkUnitDescriptor",

    # Worker data structures
    "RunMode",
    "DetailRequest",
    "ToolArgument",
    "StepToolCall",
    # Workflow mode yield types
    "HUMAN_INPUT_EVENT_TYPE",
    "WorkflowDecision",
    "ActionCall",
    "HumanCall",
    "AgentCall",
    # Action result data structures
    "ErrorStrategy",
    "ActionResult",
    "ActionStepResult",
    "ToolResult",
    # Trace data models
    "TraceStep",
    "RecordedToolCall",
    "StepOutputType",
    # Built-in tools
    "request_human_tool",
    # Scaffolding
    "create_project",
]
