"""
Cognitive Architecture Module.

A framework for building LLM-powered agents with structured thinking,
progressive information disclosure, and layered memory management.

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
- AgentAutoma: Agent automaton that orchestrates CognitiveWorkers
  - self.run(worker, ...) — primary execution method (observe-think-act)
- ErrorStrategy: Error handling strategies for self.run() (RAISE, IGNORE, RETRY)

Example
-------
>>> class MyAgent(AgentAutoma[CognitiveContext]):
...     async def cognition(self, ctx):
...         planner = CognitiveWorker.inline("Plan approach", llm=self.llm)
...         executor = CognitiveWorker.inline("Execute step", llm=self.llm)
...         await self.run(planner)
...         await self.run(executor,
...                        until=lambda ctx: ctx.done, max_attempts=20)
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
    ThinkDecision,
    # Workflow helper
    step,
)
from ._agent_automa import (
    # Orchestration
    AgentAutoma,
    AgentTrace,
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
    # Trace data models
    TraceStep,
    RunConfig,
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
    "ThinkDecision",
    "step",

    # Orchestration layer
    "AgentAutoma",
    "AgentTrace",

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
    # Trace data models
    "TraceStep",
    "RunConfig",
    "RecordedToolCall",
    "StepOutputType",
]
