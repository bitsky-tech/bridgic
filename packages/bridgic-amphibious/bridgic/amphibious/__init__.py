"""Amphibious Agent Framework — dual-mode (LLM-driven + deterministic)
agent orchestration with automatic fallback between the two.

Layers:

* **Abstraction** — ``Exposure`` / ``LayeredExposure`` / ``EntireExposure``
  / ``Context``: field-level data management with progressive disclosure.
* **Context impl** — ``Step``, ``Skill``, ``CognitiveTools``,
  ``CognitiveSkills``, ``CognitiveHistory``, ``CognitiveContext``.
* **Worker** — ``CognitiveWorker``: one in-process observe-think-act cycle,
  anchored on a ``BaseLlm``. Cognitive policies (acquiring / rehearsal /
  reflection) enable multi-round thinking. Symmetric peer: ``AgentWorker``,
  one external-agent delegation, anchored on a ``BaseAgent`` (the external
  coding-agent abstraction; ``ClaudeCodeAgent`` and ``CodexAgent`` are the
  shipped drivers).
* **Orchestration** — ``AmphibiousAutoma`` + yield primitives (``ThinkUnit``,
  ``ThinkAgent``, ``EnterAgent``, ``ActionCall``, ``HumanCall``, ``LLMCall``,
  ``RETURN``) + ``think_unit`` / ``think_agent`` descriptors.

>>> class MyAgent(AmphibiousAutoma[CognitiveContext]):
...     main_think = think_unit(CognitiveWorker.inline("Execute step"), max_attempts=20)
...     async def on_agent(self, ctx):
...         yield ThinkUnit("main_think")
...
>>> answer = await MyAgent().arun(llm=llm, goal="Complete the task")
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
    # In-process worker (LLM-driven OTC cycle)
    CognitiveWorker,
    # Sentinel
    _DELEGATE,
)
from ._agent_worker import (
    # External-agent worker (delegates one cycle to an external agent)
    AgentWorker,
)
from .temp._base_agent import (
    # External coding-agent abstraction (the BASE of AgentWorker)
    BaseAgent,
    # Concrete CLI drivers shipped with the framework
    ClaudeCodeAgent,
    CodexAgent,
    # Request / result value objects for BaseAgent.run
    AgentRequest,
    AgentResult,
)
from ._amphibious_automa import (
    # Orchestration
    AmphibiousAutoma,
    AgentTrace,
    # Human channel decorator
    human_channel,
)
from ._think_unit import (
    # Think unit (in-process CognitiveWorker)
    think_unit,
    ThinkUnitDescriptor,
)
from ._think_agent import (
    # Think agent (external-agent delegation)
    think_agent,
    ThinkAgentDescriptor,
)
from .scaffold import create_project
from .builtin_tools import (
    request_human_tool,
    bash_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    glob_tool,
    grep_tool,
)

from ._type import (
    # Worker data structures
    RunMode,
    DetailRequest,
    ToolArgument,
    StepToolCall,
    # Yield primitives (scope rules — see AmphibiousAutoma docstring)
    WorkflowDecision,
    ActionCall,
    HumanCall,
    LLMCall,
    EnterAgent,
    ThinkUnit,
    ThinkAgent,
    RETURN,
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
    "AgentWorker",
    "BaseAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "AgentRequest",
    "AgentResult",

    # Orchestration layer
    "AmphibiousAutoma",
    "AgentTrace",
    "think_unit",
    "ThinkUnitDescriptor",
    "think_agent",
    "ThinkAgentDescriptor",

    # Worker data structures
    "RunMode",
    "DetailRequest",
    "ToolArgument",
    "StepToolCall",
    # Yield primitives
    "WorkflowDecision",
    "ActionCall",
    "HumanCall",
    "LLMCall",
    "EnterAgent",
    "ThinkUnit",
    "ThinkAgent",
    "RETURN",
    # Human channel decorator
    "human_channel",
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
    "bash_tool",
    "read_file_tool",
    "write_file_tool",
    "edit_file_tool",
    "glob_tool",
    "grep_tool",
    # Scaffolding
    "create_project",
]
