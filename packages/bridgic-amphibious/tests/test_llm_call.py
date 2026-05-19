"""Tests for the LLMCall workflow yield primitive.

Covers:
* Dataclass-shape validation (factories + __post_init__ guards).
* Dispatch happy paths for the three protocols (chat / structure_output /
  tool_selector), including history ordering.
* Error paths: no LLM configured, LLM does not implement the requested
  protocol, LLM raises during the call → AMPHIFLOW fallback to on_agent.
* asend round-trip (the result reaches user code at the yield site).
* Generator exhaustion after a final LLMCall.
* Trace recording when ``trace_running=True``.
"""

from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

import pytest
from pydantic import BaseModel

from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveContext,
    CognitiveWorker,
    ActionCall,
    HumanCall,
    EnterAgent,
    LLMCall,
    StepOutputType,
    StepToolCall,
    ToolArgument,
    RunMode,
    ThinkUnit,
    think_unit,
)
from bridgic.core.model.types import Message, Response, Role, Tool, ToolCall
from bridgic.core.model.protocols import PydanticModel


# ---------------------------------------------------------------------------
# Helpers + mock LLMs
# ---------------------------------------------------------------------------


class MyResponseSchema(BaseModel):
    answer: str
    confidence: float


ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)


class FullProtocolLLM:
    """Mock LLM that supports all three protocols, scriptable per protocol."""

    def __init__(
        self,
        chat_responses: Optional[List[Response]] = None,
        structured_responses: Optional[List[Any]] = None,
        tool_selector_responses: Optional[List[Tuple[List[ToolCall], Optional[str]]]] = None,
        chat_raises: Optional[BaseException] = None,
        finish_steps: Optional[List[Any]] = None,
    ):
        self.chat_responses = list(chat_responses or [])
        self.structured_responses = list(structured_responses or [])
        self.tool_selector_responses = list(tool_selector_responses or [])
        self.chat_raises = chat_raises
        self.finish_steps = list(finish_steps or [])

        self.last_chat_messages: Optional[List[Message]] = None
        self.last_structured_messages: Optional[List[Message]] = None
        self.last_structured_constraint: Any = None
        self.last_tool_selector_messages: Optional[List[Message]] = None
        self.last_tool_selector_tools: Optional[List[Tool]] = None

        self._chat_idx = 0
        self._structured_idx = 0
        self._tool_selector_idx = 0
        self._finish_idx = 0

    async def achat(self, messages, **kwargs):
        if self.chat_raises is not None:
            raise self.chat_raises
        self.last_chat_messages = list(messages)
        if not self.chat_responses:
            raise RuntimeError("no chat responses configured")
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs):
        # ``astructured_output`` is invoked from two places: explicit
        # LLMCall.structure_output (scripted via ``structured_responses``)
        # and CognitiveWorker think-step calls inside on_agent fallback
        # (scripted via ``finish_steps``). We try ``structured_responses``
        # first so the explicit-LLMCall tests stay deterministic, and
        # fall through to ``finish_steps`` for the fallback path.
        self.last_structured_messages = list(messages)
        self.last_structured_constraint = constraint
        if self.structured_responses:
            resp = self.structured_responses[
                self._structured_idx % len(self.structured_responses)
            ]
            self._structured_idx += 1
            return resp
        if self.finish_steps:
            resp = self.finish_steps[self._finish_idx % len(self.finish_steps)]
            self._finish_idx += 1
            return resp
        raise RuntimeError("no structured responses configured")

    async def aselect_tool(self, messages, tools, **kwargs):
        self.last_tool_selector_messages = list(messages)
        self.last_tool_selector_tools = list(tools)
        if not self.tool_selector_responses:
            raise RuntimeError("no tool_selector responses configured")
        resp = self.tool_selector_responses[
            self._tool_selector_idx % len(self.tool_selector_responses)
        ]
        self._tool_selector_idx += 1
        return resp

    # Sync counterparts required by the runtime-checkable
    # StructuredOutput / ToolSelection protocols (isinstance() checks
    # attribute presence). Tests never invoke these.
    def structured_output(self, messages, constraint, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...

    # BaseLlm contract — not used by tests, but needed for completeness.
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...


class ChatOnlyLLM:
    """Mock LLM that ONLY exposes ``achat`` — no structured / tool-selector.

    Used to verify the runtime-checkable protocol guard: structure_output
    and tool_selector dispatch must raise TypeError when the LLM does not
    implement the relevant protocol.
    """

    def __init__(self, chat_responses: Optional[List[Response]] = None):
        self.chat_responses = list(chat_responses or [])
        self._chat_idx = 0

    async def achat(self, messages, **kwargs):
        if not self.chat_responses:
            raise RuntimeError("no chat responses configured")
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...


def _make_text_response(text: str) -> Response:
    return Response(message=Message.from_text(text, role=Role.AI))


def _make_ctx() -> CognitiveContext:
    return CognitiveContext(goal="LLMCall test")


def _finish_decision() -> ThinkDecision:
    return ThinkDecision(step_content="Done", output=[], finish=True)


# ---------------------------------------------------------------------------
# 1. Dataclass / factory shape
# ---------------------------------------------------------------------------


class TestLLMCallDataclass:

    def test_chat_factory_minimal(self):
        c = LLMCall.chat("hi")
        assert c.protocol == "chat"
        assert c.prompt == "hi"
        assert c.history is None
        assert c.constraint is None
        assert c.tools is None

    def test_structure_output_requires_constraint(self):
        with pytest.raises(ValueError, match="constraint"):
            LLMCall(protocol="structure_output", prompt="x")

    def test_tool_selector_requires_tools(self):
        with pytest.raises(ValueError, match="tools"):
            LLMCall(protocol="tool_selector", prompt="x")

    def test_chat_rejects_constraint_or_tools(self):
        with pytest.raises(ValueError, match="does not accept"):
            LLMCall(
                protocol="chat",
                prompt="x",
                constraint=PydanticModel(model=MyResponseSchema),
            )

    def test_structure_output_factory_round_trip(self):
        constraint = PydanticModel(model=MyResponseSchema)
        c = LLMCall.structure_output("x", constraint=constraint)
        assert c.protocol == "structure_output"
        assert c.constraint is constraint
        assert c.tools is None

    def test_tool_selector_factory_round_trip(self):
        tools = [Tool(name="t", description="d", parameters={})]
        c = LLMCall.tool_selector("x", tools=tools)
        assert c.protocol == "tool_selector"
        assert c.tools is tools
        assert c.constraint is None


# ---------------------------------------------------------------------------
# 2. Dispatch — chat protocol
# ---------------------------------------------------------------------------


class TestLLMCallDispatchChat:

    @pytest.mark.asyncio
    async def test_chat_returns_text(self):
        llm = FullProtocolLLM(chat_responses=[_make_text_response("4")])
        captured: List[str] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                text = yield LLMCall.chat("What is 2+2?")
                captured.append(text)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert captured == ["4"]

    @pytest.mark.asyncio
    async def test_chat_history_prepended(self):
        llm = FullProtocolLLM(chat_responses=[_make_text_response("ok")])
        history = [
            Message.from_text("system note", role=Role.SYSTEM),
            Message.from_text("earlier user", role=Role.USER),
        ]

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("now", history=history)

        await Agent().arun(llm=llm, context=_make_ctx())

        msgs = llm.last_chat_messages
        assert msgs is not None
        assert len(msgs) == 3
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "system note"
        assert msgs[1].role == Role.USER
        assert msgs[1].content == "earlier user"
        assert msgs[2].role == Role.USER
        assert msgs[2].content == "now"

    @pytest.mark.asyncio
    async def test_chat_response_with_no_message_falls_back_to_str(self):
        llm = FullProtocolLLM(chat_responses=[Response(message=None)])
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                v = yield LLMCall.chat("x")
                captured.append(v)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert len(captured) == 1
        assert isinstance(captured[0], str)
        assert captured[0]  # non-empty fallback string


# ---------------------------------------------------------------------------
# 3. Dispatch — structure_output protocol
# ---------------------------------------------------------------------------


class TestLLMCallDispatchStructured:

    @pytest.mark.asyncio
    async def test_structure_output_returns_pydantic_instance(self):
        instance = MyResponseSchema(answer="42", confidence=0.95)
        llm = FullProtocolLLM(structured_responses=[instance])
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                v = yield LLMCall.structure_output(
                    "extract",
                    constraint=PydanticModel(model=MyResponseSchema),
                )
                captured.append(v)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert captured == [instance]

    @pytest.mark.asyncio
    async def test_structure_output_constraint_passed_through(self):
        instance = MyResponseSchema(answer="42", confidence=0.95)
        constraint = PydanticModel(model=MyResponseSchema)
        llm = FullProtocolLLM(structured_responses=[instance])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.structure_output("x", constraint=constraint)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert llm.last_structured_constraint is constraint

    @pytest.mark.asyncio
    async def test_structure_output_typeerror_when_protocol_unsupported(self):
        llm = ChatOnlyLLM(chat_responses=[_make_text_response("never used")])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.structure_output(
                    "x", constraint=PydanticModel(model=MyResponseSchema)
                )

        # Pure WORKFLOW mode (no on_agent) → will_fallback=False → raise.
        with pytest.raises(TypeError, match="StructuredOutput"):
            await Agent().arun(llm=llm, context=_make_ctx())


# ---------------------------------------------------------------------------
# 4. Dispatch — tool_selector protocol
# ---------------------------------------------------------------------------


class TestLLMCallDispatchToolSelector:

    @pytest.mark.asyncio
    async def test_tool_selector_returns_tuple(self):
        tool_calls = [
            ToolCall(id="call_1", name="search", arguments={"q": "weather"})
        ]
        reply = "I'll look that up."
        llm = FullProtocolLLM(tool_selector_responses=[(tool_calls, reply)])
        tools = [Tool(name="search", description="search the web", parameters={})]
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                v = yield LLMCall.tool_selector("pick", tools=tools)
                captured.append(v)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert len(captured) == 1
        result = captured[0]
        assert isinstance(result, tuple)
        assert result[0] is tool_calls
        assert result[1] == reply

    @pytest.mark.asyncio
    async def test_tool_selector_typeerror_when_protocol_unsupported(self):
        llm = ChatOnlyLLM(chat_responses=[_make_text_response("never used")])
        tools = [Tool(name="search", description="d", parameters={})]

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.tool_selector("x", tools=tools)

        with pytest.raises(TypeError, match="ToolSelection"):
            await Agent().arun(llm=llm, context=_make_ctx())


# ---------------------------------------------------------------------------
# 5. asend round-trip + generator exhaustion
# ---------------------------------------------------------------------------


class TestLLMCallSendRoundTrip:

    @pytest.mark.asyncio
    async def test_send_value_threads_through_subsequent_yields(self):
        """The value returned from ``yield LLMCall(...)`` is the LLM's result,
        and the workflow continues running afterwards."""
        llm = FullProtocolLLM(
            chat_responses=[_make_text_response("first"), _make_text_response("second")]
        )
        captured: List[str] = []

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                a = yield LLMCall.chat("q1")
                captured.append(a)
                b = yield LLMCall.chat("q2")
                captured.append(b)

        await Agent().arun(llm=llm, context=_make_ctx())

        assert captured == ["first", "second"]

    @pytest.mark.asyncio
    async def test_workflow_exhausts_cleanly_after_final_llm_call(self):
        """A workflow whose last yield is an LLMCall must not deadlock or raise."""
        llm = FullProtocolLLM(chat_responses=[_make_text_response("done")])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("last")

        await Agent().arun(llm=llm, context=_make_ctx())  # no exception → pass


# ---------------------------------------------------------------------------
# 6. Error paths
# ---------------------------------------------------------------------------


class TestLLMCallErrorPaths:

    @pytest.mark.asyncio
    async def test_no_llm_configured_raises(self):

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("x")

        with pytest.raises(RuntimeError, match="self._llm"):
            await Agent().arun(context=_make_ctx())

    @pytest.mark.asyncio
    async def test_llm_internal_failure_falls_back_to_on_agent(self):
        """AMPHIFLOW: an LLMCall that raises hands control to ``on_agent(ctx)``."""
        on_agent_calls: List[str] = []
        llm = FullProtocolLLM(
            chat_raises=RuntimeError("simulated provider failure"),
            finish_steps=[_finish_decision()],
        )

        class Agent(AmphibiousAutoma[CognitiveContext]):
            recoverer = think_unit(CognitiveWorker.inline("Recover."), max_attempts=1)

            async def on_agent(self, ctx) -> AsyncGenerator[Any, Any]:
                on_agent_calls.append(ctx.goal)
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("doomed")

        await Agent().arun(llm=llm, context=_make_ctx())

        assert len(on_agent_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_internal_failure_propagates_in_workflow_mode(self):
        """Pure WORKFLOW (no on_agent override): the exception escapes arun."""
        llm = FullProtocolLLM(chat_raises=RuntimeError("provider down"))

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("doomed")

        with pytest.raises(RuntimeError, match="provider down"):
            await Agent().arun(llm=llm, context=_make_ctx())


# ---------------------------------------------------------------------------
# 7. Trace recording
# ---------------------------------------------------------------------------


class TestLLMCallTrace:

    @pytest.mark.asyncio
    async def test_llm_call_recorded_in_trace(self):
        llm = FullProtocolLLM(chat_responses=[_make_text_response("traced")])

        class Agent(AmphibiousAutoma[CognitiveContext]):
            async def on_workflow(self, ctx) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("captured prompt")

        agent = Agent()
        await agent.arun(llm=llm, context=_make_ctx(), trace_running=True)

        assert agent._agent_trace is not None
        trace = agent._agent_trace.build()
        steps = trace["steps"]
        llm_steps = [s for s in steps if s.output_type == StepOutputType.LLM_CALL]
        assert len(llm_steps) == 1
        recorded = llm_steps[0]
        assert recorded.llm_call_protocol == "chat"
        assert recorded.observation == "captured prompt"
        assert recorded.step_content == "LLMCall(chat)"
