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
- CognitiveWorker: One "observe-think-act" cycle (GraphAutoma implementation)
- ThinkingMode: FAST (merged thinking+tools) or DEFAULT (separated)

**Orchestration Layer:**
- AgentAutoma: Agent automaton that orchestrates multiple CognitiveWorkers
- think_step: Descriptor factory for declaring thinking steps
- ThinkStepDescriptor: Descriptor that wraps a CognitiveWorker as an awaitable step
- ErrorStrategy: Error handling strategies (RAISE, IGNORE, RETRY)

Example
-------
>>> from cognitive import (
...     AgentAutoma, CognitiveContext, CognitiveWorker,
...     ThinkingMode, think_step, ErrorStrategy
... )
>>>
>>> class ReactWorker(CognitiveWorker):
...     def __init__(self, llm):
...         super().__init__(llm, mode=ThinkingMode.FAST)
...
...     async def thinking(self):
...         return "Plan ONE immediate next step."
>>>
>>> class MyAgent(AgentAutoma[CognitiveContext]):
...     analyze = think_step(ReactWorker(llm))
...     execute = think_step(ReactWorker(llm), on_error=ErrorStrategy.RETRY)
...
...     async def cognition(self, ctx):
...         await self.analyze
...         while not ctx.finish:
...             await self.execute
>>>
>>> result = await MyAgent(llm=llm).arun(goal="Complete the task")
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
    ThinkingMode,
    # Data structures
    DetailRequest,
    ToolArgument,
    StepToolCall,
    FastThinkResult,
    FastThinkDecision,
    DefaultThinkResult,
    DefaultThinkDecision,
    ActionResult,
    ActionStepResult,
)
from ._agent_automa import (
    # Orchestration
    AgentAutoma,
    think_step,
    ThinkStepDescriptor,
    ErrorStrategy,
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
    "ThinkingMode",

    # Data structures
    "DetailRequest",
    "ToolArgument",
    "StepToolCall",
    "FastThinkResult",
    "FastThinkDecision",
    "DefaultThinkResult",
    "DefaultThinkDecision",
    "ActionResult",
    "ActionStepResult",
    
    # Orchestration layer
    "AgentAutoma",
    "think_step",
    "ThinkStepDescriptor",
    "ErrorStrategy",
]
