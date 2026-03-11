"""
Workflow capture and replay for AgentAutoma.

Two parallel replay paths live in this module:

1. **Flat replay** (``WorkflowStepWorker``):
   A lightweight ``Worker`` that re-executes a pre-recorded list of tool calls
   without any LLM involvement.  Used by ``AgentAutoma._build_workflow()`` to
   produce a ``GraphAutoma``-based linear replay of a completed run.

2. **Amphibious workflow** (``Workflow`` and its block types):
   A serialisable Pydantic model that captures the structured shape of an agent
   run (steps, loops, free-form linear traces).  Used by ``AmphibiousRunner``
   to replay deterministically and fall back to agent mode on divergence.

Data models for the amphibious workflow system:
- ``WorkflowToolCall``: single recorded tool call (name + arguments + result)
- ``WorkflowStepWorker``: flat-replay worker node for GraphAutoma
- ``Workflow``: top-level container (blocks + patches + version)
- ``StepBlock`` / ``LoopBlock`` / ``LinearTraceBlock``: block types
- ``WorkerConfig``: worker reconstruction metadata
- ``WorkflowPatch``: learning patch record generated after divergence
"""
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import override

from bridgic.core.automa.worker import Worker
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.model.types import ToolCall

from bridgic.core.cognitive._context import Step


class WorkflowToolCall(BaseModel):
    """A single tool call recorded during a workflow step."""

    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any = None  # auditing only — ignored on replay


class WorkflowStepWorker(Worker):
    """
    Executes a pre-recorded set of tool calls.

    Tool implementations are resolved at replay time from ``context.tools``.
    This worker is meant to be added as a node in a ``GraphAutoma`` that
    represents the flat replay of a prior agent run.

    Parameters
    ----------
    tool_calls : List[WorkflowToolCall]
        The pre-recorded tool calls for this step.
    step_content : str
        The thinking content of the original step (for auditing).
    """

    def __init__(
        self,
        tool_calls: Optional[List[WorkflowToolCall]] = None,
        step_content: str = "",
        observation_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.tool_calls: List[WorkflowToolCall] = tool_calls or []
        self.step_content: str = step_content
        self.observation_fn: Optional[Callable] = observation_fn

    async def arun(self, context: Any) -> Any:  # context: CognitiveContextT
        # Call observation before executing tools to refresh internal state
        # (e.g. browser accessibility tree ref table).
        if self.observation_fn is not None:
            await self.observation_fn(context)

        if not self.tool_calls:
            context.add_info(Step(
                content=self.step_content,
                status=True,
                metadata={"tool_calls": [], "replayed": True},
            ))
            return context

        # Build ToolCall objects from pre-recorded data
        raw_calls = [
            ToolCall(id=f"replay_{i}", name=wf.tool_name, arguments=wf.tool_arguments)
            for i, wf in enumerate(self.tool_calls)
        ]

        # Resolve ToolSpec instances from context.tools by name
        _, tool_specs = context.get_field("tools")
        matched: List[tuple] = []
        for tc in raw_calls:
            spec = next((s for s in tool_specs if s.tool_name == tc.name), None)
            if spec is None:
                raise ValueError(
                    f"Replay: tool '{tc.name}' not found in context.tools"
                )
            matched.append((tc, spec))

        # Execute tools via ConcurrentAutoma (mirrors AgentAutoma._run_tools)
        sandbox = ConcurrentAutoma()
        for tc, spec in matched:
            sandbox.add_worker(
                key=f"tool_{tc.name}_{tc.id}",
                worker=spec.create_worker(),
                args_mapping_rule=ArgsMappingRule.UNPACK,
            )

        tool_args = [tc.arguments for tc, _ in matched]
        try:
            results_raw = await sandbox.arun(InOrder(tool_args))
            action_results = [
                {
                    "tool_name": tc.name,
                    "tool_arguments": tc.arguments,
                    "tool_result": r,
                }
                for (tc, _), r in zip(matched, results_raw)
            ]
            status = True
        except Exception:
            action_results = [
                {
                    "tool_name": tc.name,
                    "tool_arguments": tc.arguments,
                    "tool_result": traceback.format_exc(),
                }
                for tc, _ in matched
            ]
            status = False

        context.add_info(Step(
            content=self.step_content,
            status=status,
            metadata={
                "tool_calls": [tc.name for tc, _ in matched],
                "action_results": action_results,
                "replayed": True,
            },
        ))
        return context

    # ── Serialization ──────────────────────────────────────────────────────

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["step_content"] = self.step_content
        state_dict["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        # observation_fn captures live agent/worker objects (LLM clients, event loops, etc.)
        # that cannot be serialized across processes.  It is intentionally omitted here;
        # in-process replay (the primary use case) keeps it in memory just fine.
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.step_content = state_dict.get("step_content", "")
        self.tool_calls = [
            WorkflowToolCall(**tc) for tc in state_dict.get("tool_calls", [])
        ]
        # observation_fn is not persisted; after deserialization it stays None.


################################################################################################################
# Amphibious Workflow Data Models
################################################################################################################

class WorkerConfig(BaseModel):
    """Configuration for reconstructing a CognitiveWorker.

    Stores enough information to recreate a worker for workflow replay,
    or to describe the worker that originally executed a step.
    """

    prompt: str
    class_name: Optional[str] = None
    enable_rehearsal: bool = False
    enable_reflection: bool = False
    output_schema: Optional[str] = None  # fully qualified class name


class WorkflowBlock(BaseModel):
    """Base class for workflow execution blocks."""

    type: Literal["step", "loop", "sequence", "branch", "linear_trace"]
    name: str = ""


class StepBlock(WorkflowBlock):
    """A single step execution block."""

    type: Literal["step"] = "step"
    worker_config: Optional[WorkerConfig] = None
    tool_calls: List[WorkflowToolCall] = Field(default_factory=list)
    tools_filter: Optional[List[str]] = None
    skills_filter: Optional[List[str]] = None


class LoopBlock(WorkflowBlock):
    """A loop execution block, recording each iteration's trace."""

    type: Literal["loop"] = "loop"
    worker_config: Optional[WorkerConfig] = None
    iterations: List[List[Dict[str, Any]]] = Field(default_factory=list)
    max_attempts: int = 10
    tools_filter: Optional[List[str]] = None
    skills_filter: Optional[List[str]] = None


class LinearTraceBlock(WorkflowBlock):
    """A block of free-form steps recorded linearly."""

    type: Literal["linear_trace"] = "linear_trace"
    steps: List[Dict[str, Any]] = Field(default_factory=list)


class WorkflowPatch(BaseModel):
    """A learning patch applied to a workflow after execution divergence."""

    type: Literal["guard", "replace", "extend"]
    block_index: int
    trigger_pattern: str
    resolution_steps: List[Dict[str, Any]] = Field(default_factory=list)
    verified: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class Workflow(BaseModel):
    """Serializable, executable workflow combining structured and linear blocks.

    A workflow is generated from a successful agent run and can be replayed
    deterministically. When replay encounters divergence, the amphibious
    engine switches to agent mode.
    """

    blocks: List[Union[StepBlock, LoopBlock, LinearTraceBlock]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    patches: List[WorkflowPatch] = Field(default_factory=list)

    def add_step_block(
        self,
        name: str,
        tool_calls: Optional[List[WorkflowToolCall]] = None,
        worker_config: Optional[WorkerConfig] = None,
        tools_filter: Optional[List[str]] = None,
        skills_filter: Optional[List[str]] = None,
    ) -> StepBlock:
        """Add a StepBlock to the workflow."""
        block = StepBlock(
            name=name,
            tool_calls=tool_calls or [],
            worker_config=worker_config,
            tools_filter=tools_filter,
            skills_filter=skills_filter,
        )
        self.blocks.append(block)
        return block

    def add_loop_block(
        self,
        name: str,
        max_attempts: int = 10,
        worker_config: Optional[WorkerConfig] = None,
        tools_filter: Optional[List[str]] = None,
        skills_filter: Optional[List[str]] = None,
    ) -> LoopBlock:
        """Add a LoopBlock to the workflow."""
        block = LoopBlock(
            name=name,
            max_attempts=max_attempts,
            worker_config=worker_config,
            tools_filter=tools_filter,
            skills_filter=skills_filter,
        )
        self.blocks.append(block)
        return block

    def add_linear_trace_block(
        self,
        name: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> LinearTraceBlock:
        """Add a LinearTraceBlock to the workflow."""
        block = LinearTraceBlock(name=name, steps=steps or [])
        self.blocks.append(block)
        return block
