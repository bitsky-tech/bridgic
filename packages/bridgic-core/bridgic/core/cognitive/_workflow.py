"""
Workflow capture and replay for AgentAutoma.

A workflow is a flat, ordered sequence of tool calls recorded during a successful
agent run, stored as a native GraphAutoma so it can be re-executed without LLM.
"""
import traceback
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel
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
            context.last_step_has_tools = False
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
        context.last_step_has_tools = True
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
