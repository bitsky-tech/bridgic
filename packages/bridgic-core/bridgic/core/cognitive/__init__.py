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
  - self.execute_plan(operator) — structured flow operators
  - exception_scope() — mark steps as exception handlers (excluded from workflow)
- ErrorStrategy: Error handling strategies for self.run() (RAISE, IGNORE, RETRY)

**Workflow & Trace:**
- Workflow: Serialisable structured workflow (StepBlock/LoopBlock/LinearTraceBlock)
- WorkflowStepWorker: Flat-replay worker node (re-executes recorded tool calls)
- TraceStep, ExecutionTrace: Per-step and per-run execution recording
- DivergenceDetector, DivergenceLevel: Replay divergence detection
- FlowStep/Loop/Sequence/Branch: Structured flow operators for execute_plan()

Example
-------
>>> class MyAgent(AgentAutoma[CognitiveContext]):
...     async def cognition(self, ctx):
...         planner = CognitiveWorker.inline("Plan approach", llm=self.llm)
...         executor = CognitiveWorker.inline("Execute step", llm=self.llm)
...         await self.run(planner, name="plan")
...         await self.run(executor, name="execute",
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
    # Data structures
    DetailRequest,
    ToolArgument,
    StepToolCall,
    ThinkDecision,
)
from ._agent_automa import (
    # Orchestration
    AgentAutoma,
    ErrorStrategy,
    # Action result data structures
    ActionResult,
    ActionStepResult,
)
from ._workflow import (
    WorkflowToolCall,
    WorkflowStepWorker,
    # Amphibious workflow data models
    Workflow,
    WorkflowBlock,
    StepBlock,
    LoopBlock,
    LinearTraceBlock,
    WorkerConfig,
    WorkflowPatch,
)
from ._trace import (
    TraceStep,
    ExecutionTrace,
    DivergenceDetector,
    DivergenceLevel,
)
from ._operators import (
    Step as FlowStep,
    Loop,
    Sequence,
    Branch,
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

    # Data structures
    "DetailRequest",
    "ToolArgument",
    "StepToolCall",
    "ThinkDecision",
    "ActionResult",
    "ActionStepResult",

    # Orchestration layer
    "AgentAutoma",
    "ErrorStrategy",

    # Workflow capture & replay
    "WorkflowToolCall",
    "WorkflowStepWorker",

    # Amphibious workflow
    "Workflow",
    "WorkflowBlock",
    "StepBlock",
    "LoopBlock",
    "LinearTraceBlock",
    "WorkerConfig",
    "WorkflowPatch",

    # Trace
    "TraceStep",
    "ExecutionTrace",
    "DivergenceDetector",
    "DivergenceLevel",

    # Flow operators
    "FlowStep",
    "Loop",
    "Sequence",
    "Branch",
]
