"""Consolidated integration tests for the workflow-side primitives.

This file collapses the former ``test_llm_call``, ``test_builtin_tools``,
``test_unified_fallback``, the full-fallback ``request_human`` slice of
``test_human_in_loop``, ``test_workflow_helper_error``, ``test_arun_workdir``
and ``test_scaffold`` suites into a small set of end-to-end flows driven
through ``await agent.arun(...)`` / the ``on_workflow`` yield primitives.

Coverage map:

* ``LLMCall`` factory guards, three-protocol dispatch + asend round-trip,
  error paths (no LLM / unsupported protocol / pure-WORKFLOW propagation)
  and trace recording.
* The seven built-in tools: the registered tool-spec set, ``bash`` capture
  envelope (+ timeout), the filesystem lifecycle (write→read→edit→glob→grep)
  with its secondary modes, the read-before-modify invariant + validation
  contract + read-tracker lifecycle, and the tool-exception → workflow
  re-raise path.
* Declared-tool resolution (no auto-injection / no opt-out kwarg) and the
  inheritance dedupe seam.
* AMPHIFLOW two-tier fallback (step-level resume vs full fallback across
  ``LLMCall`` / ``HumanCall`` / LLM-selected ``request_human``), recovery
  value shaping, and the consecutive-failure counter reset.
* ``on_workflow`` generator-internal exception handling across the three
  modes.
* ``arun(trace=, workdir=)`` artifacts / metadata / shape (incl. a
  ``THINK_AGENT`` trace step) and the scaffolding CLI.

Isolation: every test (or test class) defines its OWN
``AmphibiousAutoma`` / ``Context`` subclass — the per-class
``__init_subclass__`` registries (``_human_channels`` / ``_declared_tools``)
must never be shared across tests.
"""

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, Tuple, Union

import pytest
from pydantic import BaseModel

from bridgic.amphibious import (
    ActionCall,
    AmphibiousAutoma,
    AgentRequest,
    AgentResult,
    AgentWorker,
    BaseAgent,
    CognitiveWorker,
    Context,
    EnterAgent,
    HumanCall,
    LLMCall,
    OTAContext,
    RETURN,
    RunMode,
    StepOutputType,
    ThinkAgent,
    ThinkUnit,
    ToolResult,
    bash_tool,
    human_channel,
    read_file_tool,
    think_agent,
    think_unit,
)
from bridgic.amphibious.builtin_tools import (
    ALL_BUILTIN_TOOLS,
    current_agent,
    request_human_tool,
)
from bridgic.amphibious.builtin_tools.filesystem._shared import track_read
from bridgic.amphibious.builtin_tools.filesystem.edit_file import edit_file
from bridgic.amphibious.builtin_tools.filesystem.glob import glob
from bridgic.amphibious.builtin_tools.filesystem.grep import grep
from bridgic.amphibious.builtin_tools.filesystem.read_file import read_file
from bridgic.amphibious.builtin_tools.filesystem.write_file import write_file
from bridgic.amphibious.builtin_tools.shell.bash import bash
from bridgic.amphibious.scaffold import _AMPHI_FILENAME, create_project
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import Message, Response, Role, Tool, ToolCall


# ===========================================================================
# Shared helpers / mock LLMs (ported from the subsumed suites)
# ===========================================================================


class MyResponseSchema(BaseModel):
    answer: str
    confidence: float


def _make_text_response(text: str) -> Response:
    return Response(message=Message.from_text(text, role=Role.AI))


def _make_ctx() -> Context:
    """Big-loop (knowledge) context; the goal is seeded into the per-run OTA
    context via the ``user_input=`` arun kwarg."""
    return Context()


_GOAL = "workflow-primitive test"


class FullProtocolLLM:
    """Mock LLM supporting all three protocols, scriptable per protocol.

    ``aselect_tool`` doubles as the recovery worker's think-step source via
    ``finish_steps`` (each a ``(tool_calls, content)`` pair; an empty list
    finishes the worker). Explicit ``tool_selector`` responses take priority.
    """

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
        self.chat_message_log: List[List[Message]] = []
        self.last_structured_constraint: Any = None

        self._chat_idx = 0
        self._structured_idx = 0
        self._tool_selector_idx = 0
        self._finish_idx = 0

    async def achat(self, messages, **kwargs):
        if self.chat_raises is not None:
            raise self.chat_raises
        self.last_chat_messages = list(messages)
        self.chat_message_log.append(list(messages))
        if not self.chat_responses:
            raise RuntimeError("no chat responses configured")
        resp = self.chat_responses[self._chat_idx % len(self.chat_responses)]
        self._chat_idx += 1
        return resp

    async def astructured_output(self, messages, constraint, **kwargs):
        self.last_structured_constraint = constraint
        if self.structured_responses:
            resp = self.structured_responses[
                self._structured_idx % len(self.structured_responses)
            ]
            self._structured_idx += 1
            return resp
        raise RuntimeError("no structured responses configured")

    async def aselect_tool(self, messages, tools, **kwargs):
        if self.tool_selector_responses:
            resp = self.tool_selector_responses[
                self._tool_selector_idx % len(self.tool_selector_responses)
            ]
            self._tool_selector_idx += 1
            return resp
        if self.finish_steps:
            resp = self.finish_steps[self._finish_idx % len(self.finish_steps)]
            self._finish_idx += 1
            return resp
        raise RuntimeError("no tool_selector responses configured")

    # Sync counterparts required by the runtime-checkable protocols
    # (isinstance() checks attribute presence). Tests never invoke these.
    def structured_output(self, messages, constraint, **kwargs): ...
    def select_tool(self, messages, tools, **kwargs): ...
    def chat(self, messages, **kwargs): ...
    def stream(self, messages, **kwargs): ...
    async def astream(self, messages, **kwargs): ...


class ChatOnlyLLM:
    """Mock LLM exposing ONLY ``achat`` — used to verify the protocol guard:
    structure_output / tool_selector dispatch must raise TypeError when the
    LLM does not implement the relevant runtime-checkable protocol."""

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


class RecovererWorker(CognitiveWorker):
    """Minimal tool-selecting recovery worker (replaces the removed ``inline``).

    Its ``thinking`` calls ``aselect_tool`` with the OTA context's tools; the
    scripted LLM drives it to finish (empty tool calls), whose ``content``
    becomes the recovery sub-run's final answer when on_agent does not
    ``RETURN``.
    """

    async def thinking(
        self, ota_context: OTAContext, context: Optional[Context] = None
    ) -> Any:
        return await self._llm.aselect_tool(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )


def _finish_step(content: str = "Recovered"):
    """Scripted ``aselect_tool`` reply with no tool calls → worker finishes."""
    return ([], content)


def _tool_call_step(tool: str, content: str, **arguments):
    """Scripted ``aselect_tool`` reply with one tool call (worker continues)."""
    return ([{"name": tool, "arguments": arguments}], content)


# ===========================================================================
# 1. LLMCall primitive
# ===========================================================================


class TestLLMCallPrimitive:

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(protocol="structure_output", prompt="x"), "constraint"),
            (dict(protocol="tool_selector", prompt="x"), "tools"),
            (
                dict(
                    protocol="chat",
                    prompt="x",
                    constraint=PydanticModel(model=MyResponseSchema),
                ),
                "does not accept",
            ),
            (
                dict(
                    protocol="chat",
                    prompt="x",
                    tools=[Tool(name="t", description="d", parameters={})],
                ),
                "does not accept",
            ),
        ],
    )
    def test_factory_guards(self, kwargs, match):
        """__post_init__ guards: structure_output requires a constraint,
        tool_selector requires tools, chat rejects either. The happy ``chat``
        factory yields a minimal, fully-defaulted call."""
        with pytest.raises(ValueError, match=match):
            LLMCall(**kwargs)

        c = LLMCall.chat("hi")
        assert c.protocol == "chat"
        assert c.prompt == "hi"
        assert c.history is None
        assert c.constraint is None
        assert c.tools is None

    @pytest.mark.asyncio
    async def test_protocol_dispatch_and_roundtrip(self):
        """All three protocols dispatch correctly through a single workflow:
        chat returns text (with history prepended + a str fallback when the
        Response carries no message), structure_output returns the pydantic
        instance (constraint transparently passed through), tool_selector
        returns the ``(tool_calls, reply)`` tuple. Each yielded value is
        asend-ed back to the yield site, and the generator exhausts cleanly
        after its final LLMCall."""
        instance = MyResponseSchema(answer="42", confidence=0.95)
        constraint = PydanticModel(model=MyResponseSchema)
        tool_calls = [ToolCall(id="c1", name="search", arguments={"q": "weather"})]
        llm = FullProtocolLLM(
            chat_responses=[
                _make_text_response("4"),
                Response(message=None),  # → str fallback
            ],
            structured_responses=[instance],
            tool_selector_responses=[(tool_calls, "I'll look that up.")],
        )
        history = [
            Message.from_text("system note", role=Role.SYSTEM),
            Message.from_text("earlier user", role=Role.USER),
        ]
        captured: List[Any] = []
        tools = [Tool(name="search", description="search the web", parameters={})]

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                captured.append((yield LLMCall.chat("first", history=history)))
                captured.append((yield LLMCall.chat("no-message")))
                captured.append(
                    (yield LLMCall.structure_output("extract", constraint=constraint))
                )
                captured.append((yield LLMCall.tool_selector("pick", tools=tools)))

        await Agent().arun(llm=llm, context=_make_ctx(), user_input=_GOAL)

        # chat #1 → verbatim text + history prepended ahead of the prompt.
        assert captured[0] == "4"
        first_msgs = llm.chat_message_log[0]
        assert [m.role for m in first_msgs] == [Role.SYSTEM, Role.USER, Role.USER]
        assert first_msgs[0].content == "system note"
        assert first_msgs[-1].content == "first"
        # chat #2 → no history, just the prompt.
        assert [m.content for m in llm.chat_message_log[1]] == ["no-message"]
        # chat #2 → non-empty str fallback when Response.message is None.
        assert isinstance(captured[1], str) and captured[1]
        # structure_output → the pydantic instance, constraint passed through.
        assert captured[2] is instance
        assert llm.last_structured_constraint is constraint
        # tool_selector → the (tool_calls, reply) tuple.
        assert isinstance(captured[3], tuple)
        assert captured[3][0] is tool_calls
        assert captured[3][1] == "I'll look that up."

    @pytest.mark.parametrize(
        "make_call, llm_factory, exc, match",
        [
            (
                lambda: LLMCall.structure_output(
                    "x", constraint=PydanticModel(model=MyResponseSchema)
                ),
                lambda: ChatOnlyLLM(chat_responses=[_make_text_response("nope")]),
                TypeError,
                "StructuredOutput",
            ),
            (
                lambda: LLMCall.tool_selector(
                    "x", tools=[Tool(name="s", description="d", parameters={})]
                ),
                lambda: ChatOnlyLLM(chat_responses=[_make_text_response("nope")]),
                TypeError,
                "ToolSelection",
            ),
            (
                lambda: LLMCall.chat("x"),
                lambda: None,  # no LLM configured
                RuntimeError,
                "self._llm",
            ),
            (
                lambda: LLMCall.chat("doomed"),
                lambda: FullProtocolLLM(chat_raises=RuntimeError("provider down")),
                RuntimeError,
                "provider down",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_error_paths(self, make_call, llm_factory, exc, match):
        """In pure WORKFLOW mode (no on_agent override → no fallback) an
        LLMCall surfaces its failure to the caller: an unsupported protocol
        raises TypeError, a missing LLM raises RuntimeError, and a provider
        exception propagates verbatim."""
        call = make_call()

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield call

        with pytest.raises(exc, match=match):
            await Agent().arun(llm=llm_factory(), context=_make_ctx(), user_input=_GOAL)


# ===========================================================================
# 2. Built-in tools
# ===========================================================================


class _StubAgent:
    """Minimal stand-in for AmphibiousAutoma in filesystem-tool tests — the
    tools only read a ``_read_tracker`` dict off ``current_agent``."""

    def __init__(self) -> None:
        self._read_tracker: dict = {}


@pytest.fixture
def stub_agent():
    """Bind ``current_agent`` to a fresh stub for the duration of the test."""
    stub = _StubAgent()
    token = current_agent.set(stub)
    try:
        yield stub
    finally:
        current_agent.reset(token)


def _tracker_has(stub: _StubAgent, path: Path) -> bool:
    """Tolerant tracker membership check (macOS /tmp ↔ /private/tmp etc.)."""
    return any(
        key in stub._read_tracker
        for key in (str(path), str(path.resolve()), os.path.abspath(str(path)))
    )


class TestBuiltinTools:

    def test_all_tool_specs_registered(self):
        """All seven built-ins are present and expose the documented JSON
        schema parameters."""
        by_name = {t.tool_name: t for t in ALL_BUILTIN_TOOLS}
        expected = {
            "request_human": {"prompt", "channel"},
            "bash": {"command", "timeout", "cwd"},
            "read_file": {"file_path", "offset", "limit"},
            "write_file": {"file_path", "content"},
            "edit_file": {"file_path", "old_string", "new_string", "replace_all"},
            "glob": {"pattern", "path"},
            "grep": {
                "pattern",
                "path",
                "glob",
                "output_mode",
                "case_insensitive",
                "head_limit",
            },
        }
        assert set(by_name) == set(expected)
        for name, params in expected.items():
            actual = set(by_name[name].tool_parameters["properties"].keys())
            assert actual == params, f"{name}: {actual} != {params}"

    @pytest.mark.asyncio
    async def test_bash_capture_and_failures(self, tmp_path: Path):
        """Successful exit returns raw ``stdout`` (no envelope tags, stderr
        dropped unless redirected); a non-zero exit raises RuntimeError
        carrying the code + stderr; a timeout raises TimeoutError; an empty
        command is rejected up front."""
        result = await bash("pwd && echo line2", cwd=str(tmp_path))
        resolved = os.path.realpath(str(tmp_path))
        assert resolved in result or str(tmp_path) in result
        assert "line2" in result
        for tag in ("<stdout>", "</stdout>", "<stderr>", "<exit_code>"):
            assert tag not in result, f"raw output must not carry {tag!r}"

        # stderr on a successful command is dropped; 2>&1 merges it back.
        assert (await bash("printf 'oops' 1>&2")).strip() == ""
        assert "kept" in await bash("printf 'oops' 1>&2; printf 'kept'")

        with pytest.raises(ValueError, match="command is required"):
            await bash("   ")

        with pytest.raises(RuntimeError) as ei:
            await bash("printf 'kaboom' 1>&2; exit 7")
        assert "exit code 7" in str(ei.value) and "kaboom" in str(ei.value)
        with pytest.raises(RuntimeError, match="exit code 2"):
            await bash("echo only-stdout && exit 2")

        with pytest.raises(TimeoutError, match="timed out"):
            await bash("sleep 5", timeout=200)

    @pytest.mark.asyncio
    async def test_filesystem_lifecycle(self, tmp_path: Path, stub_agent):
        """End-to-end happy path stitching every filesystem tool together,
        plus the secondary modes (replace_all, offset/limit slicing,
        mtime-desc glob sort, grep glob/case filters + hidden-dir skip)."""
        target = tmp_path / "module.py"

        # write (no prior read) → read (line numbers + tracker) → unique edit
        assert "Created" in await write_file(
            str(target), "def foo():\n    return 'foo'\n"
        )
        content = await read_file(str(target))
        assert "1\tdef foo():" in content
        assert "2\t    return 'foo'" in content
        assert _tracker_has(stub_agent, target)
        await edit_file(str(target), "return 'foo'", "return 'bar'")
        assert "return 'bar'" in target.read_text()
        # consecutive edit only works because track_read refreshed the mtime.
        await edit_file(str(target), "def foo", "def baz")
        assert target.read_text() == "def baz():\n    return 'bar'\n"

        # glob discovery + grep across all three output modes.
        assert "module.py" in await glob("*.py", path=str(tmp_path))
        assert "module.py" in await grep("baz", path=str(tmp_path))
        assert ":1" in await grep("baz", path=str(tmp_path), output_mode="count")
        assert ":1:def baz" in await grep(
            "baz", path=str(tmp_path), output_mode="content"
        )

        # --- secondary modes -------------------------------------------------
        many = tmp_path / "many.txt"
        many.write_text("foo\n" * 5)
        await read_file(str(many))
        await edit_file(str(many), "foo", "bar", replace_all=True)
        assert many.read_text() == "bar\n" * 5

        sliced = await read_file(str(many), offset=2, limit=2)
        assert "2\tbar" in sliced and "3\tbar" in sliced
        assert "1\tbar" not in sliced and "4\tbar" not in sliced
        assert "past the end" in await read_file(str(many), offset=999)

        old = tmp_path / "old.py"; old.write_text("o")
        new = tmp_path / "new.py"; new.write_text("n")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        sort_result = await glob("*.py", path=str(tmp_path))
        assert sort_result.index("new.py") < sort_result.index("old.py")
        assert "No files matched" in await glob("*.no-such-ext", path=str(tmp_path))

        (tmp_path / "ignore.md").write_text("FOO\n")
        hidden = tmp_path / ".secret"
        hidden.mkdir()
        (hidden / "leak.py").write_text("FOO\n")
        py_only = await grep(
            "foo", path=str(tmp_path), glob="*.py", case_insensitive=True
        )
        assert "ignore.md" not in py_only and ".secret" not in py_only
        assert "No matches" in await grep("FOO_NOT_PRESENT", path=str(tmp_path))

    @pytest.mark.asyncio
    async def test_read_before_modify_and_validation(
        self, tmp_path: Path, monkeypatch, stub_agent
    ):
        """The read-before-modify invariant (+ external-mtime detection), the
        full validation/error contract for every tool, and the read-tracker
        lifecycle (reset at arun entry + ``track_read`` swallowing OSError)."""
        # --- read-before-modify invariant -----------------------------------
        target = tmp_path / "x.txt"
        target.write_text("alpha\n")
        with pytest.raises(RuntimeError, match="read_file"):
            await write_file(str(target), "y")
        assert target.read_text() == "alpha\n"  # untouched
        with pytest.raises(RuntimeError, match="read_file"):
            await edit_file(str(target), "alpha", "beta")
        await read_file(str(target))
        await edit_file(str(target), "alpha", "beta")
        assert target.read_text() == "beta\n"
        future = time.time() + 60
        os.utime(target, (future, future))
        with pytest.raises(RuntimeError, match="modified externally"):
            await edit_file(str(target), "beta", "gamma")

        # --- validation contract (paths / filesystem) -----------------------
        with pytest.raises(ValueError, match="absolute"):
            await read_file("rel/x.txt")
        with pytest.raises(ValueError, match="absolute"):
            await write_file("rel.txt", "x")
        with pytest.raises(ValueError, match="absolute"):
            await glob("*.py", path="rel")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            await read_file(str(tmp_path / "missing.txt"))
        with pytest.raises(FileNotFoundError, match="Parent directory"):
            await write_file(str(tmp_path / "no_dir" / "x.txt"), "y")
        existing_file = tmp_path / "f.txt"
        existing_file.write_text("x")
        with pytest.raises(NotADirectoryError):
            await glob("*.py", path=str(existing_file))
        with pytest.raises(NotADirectoryError):
            await grep("x", path=str(existing_file))
        with pytest.raises(ValueError, match="Not a regular file"):
            await read_file(str(tmp_path))

        # --- edit_file input checks (file must stay untouched) ---------------
        e_target = tmp_path / "e.txt"
        e_target.write_text("foo\nfoo\nfoo\n")
        await read_file(str(e_target))
        with pytest.raises(ValueError, match="must not be empty"):
            await edit_file(str(e_target), "", "y")
        with pytest.raises(ValueError, match="identical"):
            await edit_file(str(e_target), "foo", "foo")
        with pytest.raises(ValueError, match="not found"):
            await edit_file(str(e_target), "missing", "y")
        with pytest.raises(ValueError, match="occurs 3 times"):
            await edit_file(str(e_target), "foo", "bar")
        assert e_target.read_text() == "foo\nfoo\nfoo\n"

        # --- grep / glob input checks ----------------------------------------
        with pytest.raises(ValueError, match="pattern is required"):
            await grep("")
        with pytest.raises(ValueError, match="pattern is required"):
            await glob("")
        with pytest.raises(ValueError, match="Unknown output_mode"):
            await grep("x", path=str(tmp_path), output_mode="weird")
        with pytest.raises(re.error):
            await grep("[unterminated", path=str(tmp_path))

        # --- read-tracker lifecycle: reset across runs + OSError tolerance ---
        class _TrackerProbeAgent(AmphibiousAutoma[OTAContext, Context]):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tracker_size_at_entry: int = -1

            async def on_workflow(self, ota_context, context=None):
                self.tracker_size_at_entry = len(self._read_tracker)
                if False:  # pragma: no cover — async generator stub
                    yield

        probe = _TrackerProbeAgent()
        probe._read_tracker[str(tmp_path / "ghost.txt")] = 12345.0
        await probe.arun(user_input="x")
        assert probe.tracker_size_at_entry == 0

        def boom(*a, **k):
            raise OSError("simulated")

        monkeypatch.setattr("os.stat", boom)
        before = dict(stub_agent._read_tracker)
        track_read("/tmp/anything")  # best-effort hook must not raise
        # OSError is swallowed → no new tracker entry recorded.
        assert stub_agent._read_tracker == before


# ===========================================================================
# 3. Declared-tool resolution + tool-exception → workflow re-raise
# ===========================================================================


class _CaptureToolsMixin:
    """Snapshot the per-run OTA ``ctx.tools`` in ``on_workflow``. Kept off
    ``AmphibiousAutoma`` so each concrete agent parametrizes its own context."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_tools: List[str] = []

    async def on_workflow(self, ota_context, context=None):
        self.captured_tools = sorted(t.tool_name for t in ota_context.tools)
        if False:  # pragma: no cover — async generator stub
            yield


class _AllBuiltinsOTAContext(OTAContext):
    """Declares the whole built-in set on its class."""


for _spec in ALL_BUILTIN_TOOLS:
    _AllBuiltinsOTAContext.tool(_spec)


class _SubsetOTAContext(OTAContext):
    """Declares only a two-tool subset (request_human + bash)."""


_SubsetOTAContext.tool(ALL_BUILTIN_TOOLS[0])  # request_human
_SubsetOTAContext.tool(bash_tool)


class _ChildInheritsOTAContext(_SubsetOTAContext):
    """Inherits request_human + bash, declares nothing new."""


class _ChildRedeclaresOTAContext(_SubsetOTAContext):
    """Inherits request_human + bash, then re-declares ``bash``."""


_ChildRedeclaresOTAContext.tool(bash_tool)


class _AllBuiltinsAgent(_CaptureToolsMixin, AmphibiousAutoma[_AllBuiltinsOTAContext, Context]):
    pass


class _SubsetAgent(_CaptureToolsMixin, AmphibiousAutoma[_SubsetOTAContext, Context]):
    pass


class _NoToolsAgent(_CaptureToolsMixin, AmphibiousAutoma[OTAContext, Context]):
    pass


class _InheritAgent(_CaptureToolsMixin, AmphibiousAutoma[_ChildInheritsOTAContext, Context]):
    pass


class _RedeclareAgent(_CaptureToolsMixin, AmphibiousAutoma[_ChildRedeclaresOTAContext, Context]):
    pass


class _ReadFileOTAContext(OTAContext):
    """Declares ``read_file`` so a deterministic ``ActionCall`` resolves."""


_ReadFileOTAContext.tool(read_file_tool)


class _FailingToolAgent(AmphibiousAutoma[_ReadFileOTAContext, Context]):
    def __init__(self, target: str, **kwargs):
        super().__init__(**kwargs)
        self.target = target

    async def on_workflow(self, ota_context, context=None):
        yield ActionCall("read_file", file_path=self.target)


class TestDeclaredToolResolution:

    @pytest.mark.asyncio
    async def test_declared_tools_resolution_and_dedupe(self):
        """A context carries EXACTLY what it declares — no auto-injection, no
        ``builtin_tools=`` kwarg, no opt-out. Inheritance dedupes by name to a
        single entry; re-declaring an inherited spec on the child appends a
        second copy (the only dedupe seam left)."""
        expected_all = {t.tool_name for t in ALL_BUILTIN_TOOLS}

        a = _AllBuiltinsAgent()
        await a.arun(user_input="x")
        assert set(a.captured_tools) == expected_all

        sub = _SubsetAgent()
        await sub.arun(user_input="x")
        assert set(sub.captured_tools) == {"request_human", "bash"}

        none = _NoToolsAgent()
        await none.arun(user_input="x")
        assert none.captured_tools == []

        inherit = _InheritAgent()
        await inherit.arun(user_input="x")
        assert inherit.captured_tools == ["bash", "request_human"]

        redeclare = _RedeclareAgent()
        await redeclare.arun(user_input="x")
        assert redeclare.captured_tools.count("bash") == 2

        # A declared tool that raises produces ActionStepResult(success=False);
        # in pure WORKFLOW mode (no fallback) the workflow re-raises with the
        # formatted ``Tool execution failed`` message.
        failing = _FailingToolAgent(target="/nonexistent-dir/missing.txt")
        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await failing.arun(user_input="x", mode=RunMode.WORKFLOW)


# ===========================================================================
# 4. AMPHIFLOW two-tier fallback
# ===========================================================================


async def _always_fails() -> str:
    raise RuntimeError("simulated failure")


_always_fails_tool = FunctionToolSpec.from_raw(_always_fails)


class TestAmphiflowFallback:

    @pytest.mark.parametrize(
        "call, threshold, use_return, return_val",
        [
            # Step-level (failure < threshold → resume); recovery answer is the
            # recoverer's finishing step_content ("Recovered").
            ("llm", 2, False, None),
            ("human", 2, False, None),
            # Full fallback (failure == threshold → no resume).
            ("llm", 1, False, None),
            ("human", 1, False, None),
            # Recovery value shaping (step-level): explicit RETURN vs the
            # recoverer's last step_content, wrapped per the failed Call type.
            ("action", 2, True, "recovered_data"),   # RETURN → ToolResult.result
            ("action", 2, False, None),              # no RETURN → step_content
            ("human", 2, True, "yes please"),        # RETURN → answer or ""
        ],
    )
    @pytest.mark.asyncio
    async def test_two_tier_fallback_and_shaping(
        self, call, threshold, use_return, return_val
    ):
        """All three atomic Call types share two-tier AMPHIFLOW semantics: a
        failure below the threshold runs a bounded recovery on_agent and
        resumes the workflow with the recovery conclusion shaped to the failed
        Call's return type (ActionCall → one ToolResult carrying the original
        name/arguments + success=True; HumanCall / chat → ``answer or ""``);
        at/over the threshold the workflow is closed (full fallback, no
        resume). The conclusion is the explicit ``RETURN`` value when present,
        else the recoverer's last think ``step_content``. on_agent runs once."""
        expect_resume = threshold >= 2
        # When no RETURN, the recoverer's finishing step_content is the answer.
        step_content = "agent-conclusion" if not use_return else "ignored"
        llm = FullProtocolLLM(finish_steps=[_finish_step(step_content)])

        class _OTA(OTAContext):
            pass

        if call == "action":
            _OTA.tool(_always_fails_tool)

        agent_invocations: List[str] = []
        captured: List[Any] = []

        class Agent(AmphibiousAutoma[_OTA, Context]):
            recoverer = think_unit(RecovererWorker(), max_attempts=1)

            @human_channel
            async def broken(self, prompt: str) -> str:
                raise RuntimeError("channel broken")

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                agent_invocations.append(ota_context.user_input)
                yield ThinkUnit("recoverer")
                if use_return:
                    yield RETURN(return_val)

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                if call == "llm":
                    captured.append((yield LLMCall.chat("will fail")))
                elif call == "human":
                    captured.append((yield HumanCall(prompt="confirm?")))
                else:
                    captured.append((yield ActionCall("_always_fails", arg1="x")))

        await Agent().arun(
            llm=llm,
            user_input="trigger fallback",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=threshold,
        )

        assert len(agent_invocations) == 1
        if not expect_resume:
            assert captured == []  # full fallback → workflow did not resume
            return

        assert "[Workflow fallback]" in agent_invocations[0]
        assert len(captured) == 1
        # Recovery conclusion = explicit RETURN value, else the recoverer's
        # finishing step_content.
        expected = return_val if use_return else step_content
        if call == "action":
            result_list = captured[0]
            assert isinstance(result_list, list) and len(result_list) == 1
            rec = result_list[0]
            assert isinstance(rec, ToolResult)
            assert rec.tool_name == "_always_fails"
            assert rec.tool_arguments == {"arg1": "x"}
            assert rec.success is True
            assert rec.result == expected
        else:
            # chat / human shaped to ``answer or ""``.
            assert captured == [expected]

    @pytest.mark.asyncio
    async def test_successful_call_resets_counter(self):
        """A successful atomic Call after a fallback resets
        consecutive_failures. With threshold=2, four LLMCalls
        (fail/succeed/fail/succeed) complete via two step-level fallbacks — the
        second fail would breach if the counter were not reset."""

        class _AlternatingLLM:
            def __init__(self):
                self.chat_count = 0

            async def aselect_tool(self, messages, tools, **kwargs):
                return _finish_step()

            async def achat(self, messages, **kwargs):
                self.chat_count += 1
                if self.chat_count in {1, 3}:
                    raise RuntimeError(f"fail on attempt {self.chat_count}")
                return _make_text_response(f"ok-{self.chat_count}")

            async def astructured_output(self, messages, constraint, **kwargs): ...
            async def astream(self, messages, **kwargs): ...
            def chat(self, messages, **kwargs): ...
            def select_tool(self, messages, tools, **kwargs): ...
            def structured_output(self, messages, constraint, **kwargs): ...
            def stream(self, messages, **kwargs): ...

        llm = _AlternatingLLM()
        agent_invocations: List[str] = []
        results: List[Any] = []

        class Agent(AmphibiousAutoma[OTAContext, Context]):
            recoverer = think_unit(RecovererWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                agent_invocations.append(ota_context.user_input)
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall, RETURN], None
            ]:
                results.append((yield LLMCall.chat("c1")))  # fail → fallback
                results.append((yield LLMCall.chat("c2")))  # ok → reset
                results.append((yield LLMCall.chat("c3")))  # fail → fallback (not full)
                results.append((yield LLMCall.chat("c4")))  # ok

        await Agent().arun(
            llm=llm,
            user_input="counter-reset test",
            mode=RunMode.AMPHIFLOW,
            max_consecutive_fallbacks=2,
        )

        assert len(agent_invocations) == 2  # two step-level, not one full
        assert results == ["Recovered", "ok-2", "Recovered", "ok-4"]

    @pytest.mark.asyncio
    async def test_request_human_resolves_channel_in_full_fallback(self):
        """When AMPHIFLOW escalates to full fallback, an LLM-selected
        ``request_human`` tool inside on_agent still resolves to the lone
        ``@human_channel`` (channel registry survives the fallback handoff)."""

        class _FallbackOTA(OTAContext):
            pass

        _FallbackOTA.tool(request_human_tool)
        _FallbackOTA.tool(_always_fails_tool)

        # First the recoverer selects request_human, then it finishes.
        llm = FullProtocolLLM(
            finish_steps=[
                _tool_call_step("request_human", "Ask for rescue", prompt="help?"),
                _finish_step("Got help"),
            ]
        )
        captured: List[str] = []

        class Agent(AmphibiousAutoma[_FallbackOTA, Context]):
            recoverer = think_unit(RecovererWorker(), max_attempts=5)

            @human_channel
            async def stdin(self, prompt: str) -> str:
                captured.append(prompt)
                return "here is help"

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield ActionCall("_always_fails")

        await Agent().arun(
            llm=llm,
            user_input="trigger full fallback path",
            max_consecutive_fallbacks=1,
        )

        assert captured == ["help?"]


# ===========================================================================
# 5. on_workflow generator-internal exceptions
# ===========================================================================


class _GenErrRecoverWorker(CognitiveWorker):
    async def thinking(
        self, ota_context: OTAContext, context: Optional[Context] = None
    ) -> Any:
        return await self._llm.aselect_tool(
            messages=[Message.from_text(ota_context.summary(), role=Role.USER)],
            tools=[t.to_tool() for t in ota_context.tools],
        )


class TestWorkflowGeneratorError:
    """A helper raising between yields inside ``on_workflow`` cannot resume the
    generator; the handling is mode-dependent."""

    @pytest.mark.asyncio
    async def test_workflow_generator_exception_modes(self):
        # 1. Pure WORKFLOW mode (no on_agent override): exception propagates.
        class WorkflowOnly(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                raise ValueError("boom-from-helper")
                yield  # pragma: no cover

        with pytest.raises(ValueError, match="boom-from-helper"):
            await WorkflowOnly().arun(user_input="trigger helper failure")

        # 2. AMPHIFLOW (on_agent overridden): exception routes to on_agent
        #    exactly once; arun does not raise.
        on_agent_calls: List[str] = []
        llm = FullProtocolLLM(finish_steps=[_finish_step()])

        class Amphi(AmphibiousAutoma[OTAContext, Context]):
            recoverer = think_unit(_GenErrRecoverWorker(), max_attempts=1)

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                on_agent_calls.append(ota_context.user_input)
                yield ThinkUnit("recoverer")

            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                raise RuntimeError("simulated helper failure")
                yield  # pragma: no cover

        await Amphi().arun(llm=llm, user_input="trigger amphiflow fallback")
        assert len(on_agent_calls) == 1

        # 3. Explicit RunMode.AMPHIFLOW without an on_agent override → rejected
        #    at ``_resolve_mode`` before any driver runs.
        class MissingAgent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent], None
            ]:
                raise KeyError("missing key")
                yield  # pragma: no cover

        with pytest.raises(
            RuntimeError,
            match=r"requested mode=RunMode\.AMPHIFLOW but does not override on_agent\(\)",
        ):
            await MissingAgent().arun(user_input="x", mode=RunMode.AMPHIFLOW)


# ===========================================================================
# 6. arun trace / workdir artifacts (+ THINK_AGENT trace step)
# ===========================================================================


class _WorkflowAgent(AmphibiousAutoma[OTAContext, Context]):
    """A minimal workflow-only agent — no LLM needed."""

    async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
        yield RETURN("done")


class _NoopAgent(BaseAgent):
    """A ``BaseAgent`` stub — ``run()`` never spawns a subprocess."""

    async def run(self, request: AgentRequest) -> AgentResult:  # pragma: no cover
        return AgentResult(output="ok", exit_code=0, completion="agent_done")


@pytest.fixture
def mock_run_body(monkeypatch):
    """Stub ``_run_think_agent_body`` so no subprocess is spawned; the
    envelope still writes a complete THINK_AGENT trace step."""

    async def fake_body(self_agent, worker):
        return AgentResult(output="ok", exit_code=0, completion="agent_done")

    monkeypatch.setattr(AmphibiousAutoma, "_run_think_agent_body", fake_body)


class TestArunTraceAndWorkdir:

    @pytest.mark.asyncio
    async def test_arun_trace_and_workdir(self, tmp_path: Path, mock_run_body):
        """``trace`` activates the in-memory ``AgentTrace``; ``workdir``
        materialises ``<workdir>/runs/<id>/`` and only persists ``trace.json``
        (shape ``{goal, metadata, history}``) when ``trace=True`` too. The
        trace records one step per primitive — an ``LLM_CALL`` step and one
        ``THINK_AGENT`` step per ThinkAgent yield. Per-run state resets in
        ``finally``."""
        # --- artifact / activation truth table -------------------------------
        # workdir only → run dir materialised, no trace, state reset.
        bare = _WorkflowAgent()
        await bare.arun(user_input="x", workdir=tmp_path)
        assert bare._agent_trace is None
        assert bare._current_run_dir is None
        run_dirs = list((tmp_path / "runs").iterdir())
        assert len(run_dirs) == 1
        assert run_dirs[0].name.count("-") >= 1  # timestamp-prefixed id
        assert {p.name for p in run_dirs[0].iterdir() if p.is_file()} == set()

        # neither flag → no run dir created anywhere, no trace.
        clean = tmp_path / "clean"
        clean.mkdir()
        no_wd = _WorkflowAgent()
        await no_wd.arun(user_input="x")
        assert list(clean.iterdir()) == []
        assert no_wd._agent_trace is None

        # trace only → in-memory trace, no disk.
        mem = _WorkflowAgent()
        await mem.arun(user_input="x", trace=True)
        assert mem._agent_trace is not None

        # trace + workdir (string path accepted) → single trace.json, unified
        # {goal, metadata, history} shape with the expected metadata fields.
        persisted = _WorkflowAgent()
        wd2 = tmp_path / "wd2"
        await persisted.arun(user_input="hello-goal", trace=True, workdir=str(wd2))
        assert persisted._agent_trace is not None
        assert persisted._current_run_dir is None
        run_dir = next((wd2 / "runs").iterdir())
        assert {p.name for p in run_dir.iterdir() if p.is_file()} == {"trace.json"}
        trace = json.loads((run_dir / "trace.json").read_text())
        assert set(trace) == {"goal", "metadata", "history"}
        assert trace["goal"] == "hello-goal"
        meta = trace["metadata"]
        for field in (
            "agent_class", "mode", "run_id", "start_time",
            "end_time", "spent_time", "cost_time",
        ):
            assert field in meta
        assert meta["mode"] == "workflow"

        # --- LLM_CALL step recorded (in-memory trace) ------------------------
        llm = FullProtocolLLM(chat_responses=[_make_text_response("traced")])

        class LlmAgent(AmphibiousAutoma[OTAContext, Context]):
            async def on_workflow(self, ota_context, context=None) -> AsyncGenerator[
                Union[ActionCall, HumanCall, EnterAgent, LLMCall], None
            ]:
                yield LLMCall.chat("captured prompt")

        llm_agent = LlmAgent()
        await llm_agent.arun(llm=llm, context=_make_ctx(), trace=True)
        steps = llm_agent._agent_trace.build()["history"]
        llm_steps = [s for s in steps if s.output_type == StepOutputType.LLM_CALL]
        assert len(llm_steps) == 1
        assert llm_steps[0].llm_call_protocol == "chat"
        assert llm_steps[0].observation == "captured prompt"
        assert llm_steps[0].step_content == "LLMCall(chat)"

        # --- THINK_AGENT step per yield (persisted, run body stubbed) --------
        class DelegAgent(AmphibiousAutoma[OTAContext, Context]):
            do_thing = think_agent(AgentWorker(_NoopAgent()))

            async def on_agent(self, ota_context, context=None) -> AsyncGenerator[Any, Any]:
                yield ThinkAgent("do_thing", goal="g1")
                yield ThinkAgent("do_thing", goal="g2")

        wd3 = tmp_path / "wd3"
        await DelegAgent().arun(user_input="x", trace=True, workdir=wd3)
        ta_run = next((wd3 / "runs").iterdir())
        ta_trace = json.loads((ta_run / "trace.json").read_text())
        ta_steps = [s for s in ta_trace["history"] if s.get("name") == "think_agent"]
        assert [s["structured_output"]["goal"] for s in ta_steps] == ["g1", "g2"]
        first = ta_steps[0]
        assert first["output_type"] == "think_agent"
        assert first["think_agent_name"] == "do_thing"
        assert first["structured_output"]["result"] == "ok"
        assert first["structured_output"]["exit_code"] == 0
        assert first["structured_output"]["completion_signal"] == "agent_done"


# ===========================================================================
# 7. Scaffolding CLI
# ===========================================================================


class TestScaffold:

    def test_create_project(self, tmp_path: Path):
        """``create_project`` writes exactly one compilable ``amphi.py`` (no
        __main__/asyncio/dotenv, no legacy files), omits the ``# Task:`` comment
        without ``--task`` and injects it with one, surfaces the two-loop
        skeleton, creates a missing base_dir on demand, and refuses to clobber
        an existing file."""
        # Default (no --task): single compilable file, no legacy, no comment.
        no_task_dir = tmp_path / "plain"
        path = create_project(base_dir=str(no_task_dir))
        assert sorted(os.listdir(no_task_dir)) == [_AMPHI_FILENAME]
        assert path == no_task_dir / _AMPHI_FILENAME and path.is_file()
        source = path.read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError on failure
        for forbidden in ("__main__", "asyncio", "dotenv", "# Task:"):
            assert forbidden not in source
        for legacy in (
            "task.md", "config.py", "tools.py", "workers.py", "agents.py",
            "skills", "result", "log", ".env", ".env.example",
        ):
            assert not (no_task_dir / legacy).exists(), f"{legacy} should not exist"

        # Repeat invocation must not clobber the existing file.
        with pytest.raises(FileExistsError):
            create_project(base_dir=str(no_task_dir))

        # --task injects a top-of-file comment; the template exposes the
        # two-loop skeleton. The nested base_dir is created on demand.
        nested = tmp_path / "nested" / "deep"
        assert not nested.exists()
        task_path = create_project(base_dir=str(nested), task="Navigate to example.com")
        assert task_path.is_file() and task_path.parent == nested
        task_source = task_path.read_text(encoding="utf-8")
        assert task_source.startswith("# Task: Navigate to example.com\n")
        assert "class AmphiOTAContext(OTAContext):" in task_source
        assert "class AmphiBigContext(Context):" in task_source
        assert (
            "class Amphi(AmphibiousAutoma[AmphiOTAContext, AmphiBigContext]):"
            in task_source
        )
        assert "think_unit(" in task_source
        assert "class MainThink(CognitiveWorker):" in task_source
        assert "async def thinking(self, ota_context" in task_source
        assert "async def on_agent(self, ota_context" in task_source
        assert "async def on_workflow(self, ota_context" in task_source
        for name in ("ActionCall", "EnterAgent", "HumanCall"):
            assert name in task_source, f"{name} should be visible in the template"
