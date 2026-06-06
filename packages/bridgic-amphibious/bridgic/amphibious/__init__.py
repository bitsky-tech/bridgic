"""Amphibious Agent Framework — dual-mode (LLM-driven + deterministic)
agent orchestration with automatic fallback between the two.

Layers:

* **Context** — ``Context`` (free-form, big-loop) + ``OTAContext`` (small-loop,
  framework-owned: the run's ``user_input`` + OTA round trace + tools). The
  OTA context declares the tools it carries via ``OTAContext.tool`` (decorator
  or call); nothing is auto-injected.
* **Worker** — ``CognitiveWorker``: one in-process observe-think-act cycle,
  anchored on a ``BaseLlm``; subclass and implement its ``thinking`` template
  method. Symmetric peer: ``AgentWorker``, one external-agent delegation,
  anchored on a ``BaseAgent`` (the external coding-agent abstraction;
  ``ClaudeCodeAgent`` and ``CodexAgent`` are the shipped drivers).
* **Orchestration** — ``AmphibiousAutoma`` + yield primitives (``ThinkUnit``,
  ``ThinkAgent``, ``EnterAgent``, ``ActionCall``, ``HumanCall``, ``LLMCall``,
  ``RETURN``) + ``think_unit`` / ``think_agent`` descriptors.

>>> class MyThink(CognitiveWorker):
...     async def thinking(self, ota_context, context=None):
...         return await self._llm.aselect_tool(messages=[...], tools=[...])
>>> class MyAgent(AmphibiousAutoma[OTAContext, Context]):
...     main_think = think_unit(MyThink(), max_attempts=20)
...     async def on_agent(self, ota_ctx):
...         yield ThinkUnit("main_think")
...
>>> answer = await MyAgent().arun(llm=llm, user_input="Complete the task")
"""
from ._context import (
    # Base (free-form big-loop) + small-loop OTA context
    Context,
    OTAContext,
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
    # Context-layer models
    Step,
    # Worker data structures
    RunMode,
    ToolArgument,
    StepToolCall,
    # Small-loop round record (one OTA round)
    OTARecord,
    # Yield primitives (scope rules — see AmphibiousAutoma docstring)
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
    # Context layer
    "Context",
    "OTAContext",
    "Step",
    "OTARecord",

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
    "ToolArgument",
    "StepToolCall",
    # Yield primitives
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
