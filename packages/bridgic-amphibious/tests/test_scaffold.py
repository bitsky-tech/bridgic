"""Tests for the bridgic-amphibious project scaffolding CLI."""

import ast
import os

import pytest

from bridgic.amphibious.scaffold import _AMPHI_FILENAME, create_project


class TestCreateProject:

    def test_generates_only_amphi_py(self, tmp_path):
        """The scaffold writes exactly one file and creates no subdirectories."""
        path = create_project(base_dir=str(tmp_path))

        entries = sorted(os.listdir(tmp_path))
        assert entries == [_AMPHI_FILENAME]
        assert path == tmp_path / _AMPHI_FILENAME
        assert path.is_file()

        # Sanity: none of the legacy files / dirs should appear
        for legacy in ("task.md", "config.py", "tools.py", "workers.py",
                       "agents.py", "skills", "result", "log",
                       ".env", ".env.example"):
            assert not (tmp_path / legacy).exists(), f"{legacy} should not be generated"

    def test_amphi_py_is_compilable(self, tmp_path):
        """The generated file must be syntactically valid Python."""
        path = create_project(base_dir=str(tmp_path))
        source = path.read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError on failure

    def test_amphi_py_has_no_main_block(self, tmp_path):
        """Per design: the template must not contain a __main__ entry point."""
        path = create_project(base_dir=str(tmp_path))
        source = path.read_text(encoding="utf-8")
        assert "__main__" not in source
        assert "asyncio" not in source
        assert "dotenv" not in source

    def test_task_arg_injected_as_comment(self, tmp_path):
        """--task injects a top-of-file '# Task: ...' comment."""
        path = create_project(base_dir=str(tmp_path), task="Navigate to example.com")
        source = path.read_text(encoding="utf-8")
        assert source.startswith("# Task: Navigate to example.com\n")

    def test_no_task_arg_omits_comment(self, tmp_path):
        """Without --task, no '# Task:' comment is emitted."""
        path = create_project(base_dir=str(tmp_path))
        source = path.read_text(encoding="utf-8")
        assert "# Task:" not in source

    def test_fails_when_amphi_py_already_exists(self, tmp_path):
        """A second invocation in the same directory must not clobber an existing file."""
        create_project(base_dir=str(tmp_path))
        with pytest.raises(FileExistsError):
            create_project(base_dir=str(tmp_path))

    def test_creates_base_dir_if_missing(self, tmp_path):
        """A non-existent base_dir is created on demand."""
        target = tmp_path / "nested" / "deep"
        assert not target.exists()
        path = create_project(base_dir=str(target))
        assert path.is_file()
        assert path.parent == target

    def test_template_surfaces_core_architecture(self, tmp_path):
        """The template should expose the framework's main building blocks so a
        user discovers the architecture by reading it."""
        path = create_project(base_dir=str(tmp_path))
        source = path.read_text(encoding="utf-8")

        # Custom Context subclass
        assert "class AmphiContext(CognitiveContext):" in source
        # AmphibiousAutoma subclass parameterized on the custom context
        assert "class Amphi(AmphibiousAutoma[AmphiContext]):" in source
        # think_unit declaration with a CognitiveWorker
        assert "think_unit(" in source
        assert "CognitiveWorker.inline(" in source
        # Both lifecycle methods present (defines AMPHIFLOW resolution)
        assert "async def on_agent(self, ctx" in source
        assert "async def on_workflow(self, ctx" in source
        # Workflow primitives imported (referenced in commented hints)
        for name in ("ActionCall", "AgentCall", "HumanCall"):
            assert name in source, f"{name} should be visible in the template"
