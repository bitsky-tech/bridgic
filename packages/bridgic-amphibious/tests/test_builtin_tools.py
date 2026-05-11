"""Integration tests for the built-in tools shipped with bridgic-amphibious.

Each test exercises a meaningful slice of behaviour rather than a single
assertion — favouring realistic flows (write→read→edit→glob→grep, the
read-before-modify chain, the builtin-tool injection filter resolution)
over isolated unit checks.

Where a single concern doesn't compose with neighbours (bash timeout,
end-to-end exception propagation through ``_action``), it stays in its
own test for diagnostic clarity.
"""

import os
import re
import time
from pathlib import Path
from typing import List

import pytest

from bridgic.amphibious import (
    ActionCall,
    AmphibiousAutoma,
    CognitiveContext,
    RunMode,
    bash_tool,
)
from bridgic.amphibious.builtin_tools import ALL_BUILTIN_TOOLS, current_agent
from bridgic.amphibious.builtin_tools.filesystem._shared import track_read
from bridgic.amphibious.builtin_tools.filesystem.edit_file import edit_file
from bridgic.amphibious.builtin_tools.filesystem.glob import glob
from bridgic.amphibious.builtin_tools.filesystem.grep import grep
from bridgic.amphibious.builtin_tools.filesystem.read_file import read_file
from bridgic.amphibious.builtin_tools.filesystem.write_file import write_file
from bridgic.amphibious.builtin_tools.shell.bash import bash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAgent:
    """Minimal stand-in for AmphibiousAutoma in tests of filesystem tools.

    The real agent only contributes a ``_read_tracker`` dict to these tools,
    so we set the ContextVar to a stub holding just that dict.
    """

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


# ---------------------------------------------------------------------------
# 1. Smoke: every built-in tool exposes the expected JSON-schema parameters
# ---------------------------------------------------------------------------


def test_all_tool_specs_registered():
    """All seven built-ins are present and expose the documented parameters."""
    by_name = {t.tool_name: t for t in ALL_BUILTIN_TOOLS}
    expected = {
        "request_human": {"prompt", "channel"},
        "bash": {"command", "timeout", "cwd"},
        "read_file": {"file_path", "offset", "limit"},
        "write_file": {"file_path", "content"},
        "edit_file": {"file_path", "old_string", "new_string", "replace_all"},
        "glob": {"pattern", "path"},
        "grep": {"pattern", "path", "glob", "output_mode",
                 "case_insensitive", "head_limit"},
    }
    assert set(by_name.keys()) == set(expected.keys())
    for name, params in expected.items():
        actual = set(by_name[name].tool_parameters["properties"].keys())
        assert actual == params, f"{name}: {actual} != {params}"


# ---------------------------------------------------------------------------
# 2. Bash — full capture envelope in one shot, timeout in isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_full_capture_envelope(tmp_path: Path):
    """One run exercises stdout, exit_code and cwd; another covers stderr.

    Also asserts the empty-command guard so the validation path is verified
    alongside the happy path.
    """
    # stdout + cwd + non-zero exit code in a single command
    result = await bash("pwd && echo line2 && exit 3", cwd=str(tmp_path))
    assert "<exit_code>3</exit_code>" in result
    assert "line2" in result
    resolved = os.path.realpath(str(tmp_path))
    assert resolved in result or str(tmp_path) in result

    # stderr separate so the envelope clearly contains both tags
    result_err = await bash("printf 'oops' 1>&2")
    assert "<stderr>" in result_err and "oops" in result_err
    assert "<exit_code>0</exit_code>" in result_err

    # Empty command rejected up front
    with pytest.raises(ValueError, match="command is required"):
        await bash("   ")


@pytest.mark.asyncio
async def test_bash_timeout_raises():
    """``asyncio.TimeoutError`` is converted to a ``TimeoutError`` with a
    descriptive message, and the killed process is awaited (no zombie)."""
    with pytest.raises(TimeoutError, match="timed out"):
        await bash("sleep 5", timeout=200)


# ---------------------------------------------------------------------------
# 3. Filesystem lifecycle — write → read → edit → glob → grep on real files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filesystem_lifecycle(tmp_path: Path, stub_agent):
    """End-to-end happy path stitching every filesystem tool together.

    Covers: write_file create, read_file line numbers + tracker side-effect,
    edit_file unique replace, consecutive edits (tracker refresh), glob
    discovery, grep across all three output modes.
    """
    target = tmp_path / "module.py"

    # 1. write a new file — no prior read needed
    msg = await write_file(str(target), "def foo():\n    return 'foo'\n")
    assert "Created" in msg

    # 2. read it — line numbers + tracker recorded
    content = await read_file(str(target))
    assert "1\tdef foo():" in content
    assert "2\t    return 'foo'" in content
    assert _tracker_has(stub_agent, target)

    # 3. unique replacement edit
    await edit_file(str(target), "return 'foo'", "return 'bar'")
    assert "return 'bar'" in target.read_text()

    # 4. consecutive edit — only works if track_read refreshed the mtime
    #    after the previous write; otherwise this would trip the staleness
    #    check.
    await edit_file(str(target), "def foo", "def baz")
    assert target.read_text() == "def baz():\n    return 'bar'\n"

    # 5. glob discovers the file
    listing = await glob("*.py", path=str(tmp_path))
    assert "module.py" in listing

    # 6. grep across all three output modes
    files = await grep("baz", path=str(tmp_path))
    assert "module.py" in files

    counts = await grep("baz", path=str(tmp_path), output_mode="count")
    assert ":1" in counts

    contents = await grep("baz", path=str(tmp_path), output_mode="content")
    assert ":1:def baz" in contents


# ---------------------------------------------------------------------------
# 4. Filesystem extras — replace_all, offset/limit, glob sorting, grep filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filesystem_extras(tmp_path: Path, stub_agent):
    """Secondary modes that don't fit into the linear lifecycle test."""
    # replace_all
    target = tmp_path / "many.txt"
    target.write_text("foo\n" * 5)
    await read_file(str(target))
    await edit_file(str(target), "foo", "bar", replace_all=True)
    assert target.read_text() == "bar\n" * 5

    # offset + limit slicing
    sliced = await read_file(str(target), offset=2, limit=2)
    assert "2\tbar" in sliced and "3\tbar" in sliced
    assert "1\tbar" not in sliced and "4\tbar" not in sliced

    # offset past the end returns informational text, not an exception
    oob = await read_file(str(target), offset=999)
    assert "past the end" in oob

    # glob: mtime-desc sort + no-match informational return
    old = tmp_path / "old.py"; old.write_text("o")
    new = tmp_path / "new.py"; new.write_text("n")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    sort_result = await glob("*.py", path=str(tmp_path))
    assert sort_result.index("new.py") < sort_result.index("old.py")
    assert "No files matched" in await glob("*.no-such-ext", path=str(tmp_path))

    # grep: glob filter + case-insensitive + hidden directory skipped
    (tmp_path / "ignore.md").write_text("FOO\n")
    hidden = tmp_path / ".secret"
    hidden.mkdir()
    (hidden / "leak.py").write_text("FOO\n")

    py_only = await grep(
        "foo", path=str(tmp_path), glob="*.py", case_insensitive=True
    )
    assert "ignore.md" not in py_only
    assert ".secret" not in py_only

    no_match = await grep("FOO_NOT_PRESENT", path=str(tmp_path))
    assert "No matches" in no_match


# ---------------------------------------------------------------------------
# 5. Read-before-modify invariant — comprehensive chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_before_modify_invariant(tmp_path: Path, stub_agent):
    """Edit/Write must refuse without a prior read, and must detect when
    the file changes between read and modify."""
    target = tmp_path / "x.txt"
    target.write_text("alpha\n")

    # 1. write to existing file without prior read → RuntimeError
    with pytest.raises(RuntimeError, match="read_file"):
        await write_file(str(target), "y")
    assert target.read_text() == "alpha\n"  # untouched

    # 2. edit existing file without prior read → RuntimeError
    with pytest.raises(RuntimeError, match="read_file"):
        await edit_file(str(target), "alpha", "beta")

    # 3. after read_file, edit succeeds
    await read_file(str(target))
    await edit_file(str(target), "alpha", "beta")
    assert target.read_text() == "beta\n"

    # 4. external modification (mtime advanced) → RuntimeError
    future = time.time() + 60
    os.utime(target, (future, future))
    with pytest.raises(RuntimeError, match="modified externally"):
        await edit_file(str(target), "beta", "gamma")


# ---------------------------------------------------------------------------
# 6. Validation contract — every "raise" path collected for documentary value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_contract(tmp_path: Path, stub_agent):
    """One test enumerating every validation error each tool surfaces.

    Reading the body tells you what bad inputs to expect; we also verify
    files aren't mutated when an edit-related validation fires.
    """
    # --- path / filesystem validation --------------------------------------
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

    # --- edit_file input checks (all should leave the file untouched) ------
    target = tmp_path / "e.txt"
    target.write_text("foo\nfoo\nfoo\n")
    await read_file(str(target))
    with pytest.raises(ValueError, match="must not be empty"):
        await edit_file(str(target), "", "y")
    with pytest.raises(ValueError, match="identical"):
        await edit_file(str(target), "foo", "foo")
    with pytest.raises(ValueError, match="not found"):
        await edit_file(str(target), "missing", "y")
    with pytest.raises(ValueError, match="occurs 3 times"):
        await edit_file(str(target), "foo", "bar")
    assert target.read_text() == "foo\nfoo\nfoo\n"

    # --- grep / glob input checks ------------------------------------------
    with pytest.raises(ValueError, match="pattern is required"):
        await grep("")
    with pytest.raises(ValueError, match="pattern is required"):
        await glob("")
    with pytest.raises(ValueError, match="Unknown output_mode"):
        await grep("x", path=str(tmp_path), output_mode="weird")
    with pytest.raises(re.error):
        await grep("[unterminated", path=str(tmp_path))


# ---------------------------------------------------------------------------
# 7. Built-in tool injection filter — every branch in one chain
# ---------------------------------------------------------------------------


class _CapturingAgent(AmphibiousAutoma[CognitiveContext]):
    """Workflow agent that snapshots ``ctx.tools`` then exits."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_tools: List[str] = []

    async def on_workflow(self, ctx):
        self.captured_tools = sorted(t.tool_name for t in ctx.tools.get_all())
        if False:  # pragma: no cover — async generator stub
            yield


class _SubsetAgent(_CapturingAgent):
    builtin_tools = frozenset({"request_human", "bash"})


class _OptOutAgent(_CapturingAgent):
    builtin_tools = frozenset()


class _TypoAgent(_CapturingAgent):
    builtin_tools = frozenset({"read_files"})  # typo: trailing 's'


@pytest.mark.asyncio
async def test_builtin_tools_filter_resolution():
    """Walk every branch: default-all, class-subset, runtime-override,
    opt-out, user-tool dedup, unknown-name raise."""
    expected_all = {t.tool_name for t in ALL_BUILTIN_TOOLS}

    # 1. None (default) → inject every built-in
    a = _CapturingAgent()
    await a.arun(goal="x")
    assert set(a.captured_tools) == expected_all

    # 2. Class-level subset
    sub = _SubsetAgent()
    await sub.arun(goal="x")
    assert set(sub.captured_tools) == {"request_human", "bash"}

    # 3. arun(builtin_tools=...) overrides the class attribute
    sub2 = _SubsetAgent()
    await sub2.arun(goal="x", builtin_tools=["read_file"])
    assert sub2.captured_tools == ["read_file"]

    # 4. Empty frozenset opts out entirely
    opt = _OptOutAgent()
    await opt.arun(goal="x")
    assert opt.captured_tools == []

    # 5. User-supplied tool with the same name wins (built-in skipped)
    a2 = _CapturingAgent()
    await a2.arun(goal="x", tools=[bash_tool], builtin_tools=["bash"])
    assert a2.captured_tools.count("bash") == 1

    # 6. Unknown name in arun kwarg fails loudly
    a3 = _CapturingAgent()
    with pytest.raises(ValueError, match="Unknown built-in tool name"):
        await a3.arun(goal="x", builtin_tools=["bahs"])

    # 7. Unknown name in class attr fails loudly too
    typo = _TypoAgent()
    with pytest.raises(ValueError, match="Unknown built-in tool name"):
        await typo.arun(goal="x")


# ---------------------------------------------------------------------------
# 8. Read-tracker lifecycle — reset across runs + stat-failure resilience
# ---------------------------------------------------------------------------


class _TrackerProbeAgent(AmphibiousAutoma[CognitiveContext]):
    """Workflow agent that records its tracker size at workflow entry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker_size_at_entry: int = -1

    async def on_workflow(self, ctx):
        self.tracker_size_at_entry = len(self._read_tracker)
        if False:
            yield


@pytest.mark.asyncio
async def test_read_tracker_lifecycle(tmp_path: Path, monkeypatch, stub_agent):
    """Tracker is reset at every arun() entry, and ``track_read`` is a
    best-effort hook that never propagates stat errors."""
    # Reset across runs: pre-populate, run, observe empty.
    agent = _TrackerProbeAgent()
    agent._read_tracker[str(tmp_path / "ghost.txt")] = 12345.0
    await agent.arun(goal="x")
    assert agent.tracker_size_at_entry == 0

    # track_read swallows OSError so a successful read never gets masked.
    def boom(*a, **k):
        raise OSError("simulated")
    monkeypatch.setattr("os.stat", boom)
    track_read("/tmp/anything")  # must not raise
    assert stub_agent._read_tracker == {}


# ---------------------------------------------------------------------------
# 9. End-to-end: tool exception → ActionStepResult(success=False) → workflow
# ---------------------------------------------------------------------------


class _FailingToolAgent(AmphibiousAutoma[CognitiveContext]):
    """Workflow agent that yields a single read_file ActionCall."""

    def __init__(self, target: str, **kwargs):
        super().__init__(**kwargs)
        self.target = target

    async def on_workflow(self, ctx):
        yield ActionCall("read_file", file_path=self.target)


@pytest.mark.asyncio
async def test_tool_exception_surfaces_via_action_step_result(tmp_path: Path):
    """A tool that raises produces ``ActionStepResult(success=False)``;
    in WORKFLOW mode (no fallback) the workflow re-raises with the formatted
    error message — proving the framework's existing exception path is what
    the new tools rely on."""
    agent = _FailingToolAgent(target=str(tmp_path / "missing.txt"))
    with pytest.raises(RuntimeError, match="Tool execution failed"):
        await agent.arun(goal="x", mode=RunMode.WORKFLOW)
