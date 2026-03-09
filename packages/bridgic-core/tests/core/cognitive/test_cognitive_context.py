"""Tests for CognitiveContext: summary, format_summary, disclosed_details, skills.add()."""
import os
import pytest

from bridgic.core.cognitive import CognitiveContext, CognitiveSkills, Skill, Step
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")
TRAVEL_SKILL_FILE = os.path.join(SKILLS_DIR, "travel-planning", "SKILL.md")


def _make_context() -> CognitiveContext:
    """Create a context with tools, skills and some history."""
    ctx = CognitiveContext(goal="Plan a trip to Tokyo")
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)
    ctx.skills.load_from_directory(SKILLS_DIR)
    ctx.add_info(Step(content="Search flights", status=True, result="Found 3 flights"))
    ctx.add_info(Step(content="Book flight CA123", status=True, result="Booking confirmed"))
    return ctx


class TestCognitiveContext:

    def test_summary_and_format(self):
        """summary() structure and all format_summary() modes."""
        ctx = _make_context()

        # --- summary() returns dict with all expected keys ---
        summary = ctx.summary()
        assert isinstance(summary, dict)
        assert all(key in summary for key in ["goal", "tools", "skills", "cognitive_history"])
        assert "Plan a trip to Tokyo" in summary["goal"]
        assert isinstance(summary["tools"], str)
        assert isinstance(summary["skills"], str)
        assert isinstance(summary["cognitive_history"], str)

        # --- format_summary(include=...) ---
        result = ctx.format_summary(include=["goal", "tools"])
        assert "Plan a trip to Tokyo" in result
        assert "Available Tools" in result
        assert "Available Skills" not in result
        assert "Available Skills" not in result

        # --- format_summary(exclude=...) ---
        result = ctx.format_summary(exclude=["tools"])
        assert "Plan a trip to Tokyo" in result
        assert "Available Tools" not in result
        assert "Skills" in result or "skills" in result.lower()

        # --- format_summary() default includes everything ---
        ctx2 = _make_context()
        result = ctx2.format_summary()
        assert "Plan a trip to Tokyo" in result
        assert "Tools" in result or "tools" in result.lower()
        assert "History" in result or "history" in result.lower()

    def test_disclosed_details(self):
        """get_details() persists disclosed details in subsequent summary()."""
        ctx = _make_context()

        # Before requesting details, no disclosed_details key
        summary_before = ctx.summary()
        assert "disclosed_details" not in summary_before

        # Request details for skills[0]
        detail = ctx.get_details("skills", 0)
        assert detail is not None

        # After requesting details, disclosed_details should appear
        summary_after = ctx.summary()
        assert "disclosed_details" in summary_after
        assert "Previously Disclosed Details" in summary_after["disclosed_details"]

    def test_skills_add_unified(self):
        """CognitiveSkills.add() accepts Skill object, file path, or markdown text."""
        skills = CognitiveSkills()

        # --- Skill object ---
        skill = Skill(name="test", description="A test skill", content="Do things.")
        idx = skills.add(skill)
        assert idx == 0
        assert skills[0].name == "test"

        # --- File path string ---
        idx = skills.add(TRAVEL_SKILL_FILE)
        assert idx == 1
        assert skills[1].name == "travel-planning"

        # --- Markdown text ---
        markdown = (
            "---\n"
            "name: inline-skill\n"
            "description: An inline skill\n"
            "---\n"
            "# Instructions\nDo something inline."
        )
        idx = skills.add(markdown)
        assert idx == 2
        assert skills[2].name == "inline-skill"

        # --- Wrong type → TypeError ---
        with pytest.raises(TypeError, match="expected Skill or str"):
            skills.add(123)


class TestLayeredExposureReveal:
    """Tests for Feature 1: LayeredExposure owns disclosure state."""

    def test_reveal_caches_result(self):
        """reveal() returns detail and caches it in _revealed."""
        ctx = _make_context()
        skills = ctx.skills

        # _revealed starts empty
        assert len(skills._revealed) == 0

        # reveal() fetches and caches the detail
        detail = skills.reveal(0)
        assert detail is not None
        assert 0 in skills._revealed
        assert skills._revealed[0] == detail

    def test_reveal_caches_and_returns_same(self):
        """reveal() returns the same cached result on repeat calls."""
        ctx = _make_context()
        skills = ctx.skills

        d1 = skills.reveal(0)
        d2 = skills.reveal(0)
        assert d1 == d2
        assert len(skills._revealed) == 1  # Only cached once

    def test_reset_revealed_clears_cache(self):
        """reset_revealed() clears all cached reveals."""
        ctx = _make_context()
        skills = ctx.skills

        skills.reveal(0)
        assert 0 in skills._revealed

        skills.reset_revealed()
        assert len(skills._revealed) == 0

    def test_context_get_details_uses_reveal(self):
        """context.get_details() delegates to field.reveal() and populates _revealed."""
        ctx = _make_context()

        # get_details via context
        detail = ctx.get_details("skills", 0)
        assert detail is not None

        # Internally the skill field's _revealed was updated
        assert 0 in ctx.skills._revealed
        assert ctx.skills._revealed[0] == detail

    def test_context_reset_revealed(self):
        """ctx.reset_revealed() clears all LayeredExposure fields' reveal caches."""
        ctx = _make_context()
        ctx.get_details("skills", 0)
        assert len(ctx.skills._revealed) > 0

        ctx.reset_revealed()
        assert len(ctx.skills._revealed) == 0

    def test_context_get_revealed_items(self):
        """get_revealed_items() returns (field, idx) tuples for all revealed items."""
        ctx = _make_context()

        # Nothing revealed yet
        assert ctx.get_revealed_items() == []

        ctx.get_details("skills", 0)
        revealed = ctx.get_revealed_items()
        assert ("skills", 0) in revealed

    def test_summary_disclosed_from_revealed(self):
        """summary() builds 'disclosed_details' from _revealed dicts, not _disclosed_details."""
        ctx = _make_context()

        # Use ctx.get_details to trigger reveal
        ctx.get_details("skills", 0)

        summary = ctx.summary()
        assert "disclosed_details" in summary
        assert "skills[0]" in summary["disclosed_details"]

    def test_skill_reset_revealed_clears_from_summary(self):
        """After reset_revealed(), disclosed_details disappears from summary."""
        ctx = _make_context()
        ctx.get_details("skills", 0)
        assert "disclosed_details" in ctx.summary()

        ctx.skills.reset_revealed()
        assert "disclosed_details" not in ctx.summary()


class TestContextSnapshot:
    """Tests for Context.snapshot() three-mode revealed management."""

    @pytest.mark.asyncio
    async def test_snapshot_clear_all_clears_revealed_on_enter(self):
        """Default mode (keep_revealed=None): _revealed cleared on enter, restored on exit."""
        ctx = _make_context()
        # Reveal skill[0] before entering snapshot
        ctx.get_details("skills", 0)
        assert 0 in ctx.skills._revealed

        async with ctx.snapshot(goal="sub-goal") as c:
            assert c is ctx
            # Inside: revealed cleared
            assert 0 not in ctx.skills._revealed
            # Reveal skill[1] inside
            ctx.get_details("skills", 1)
            assert 1 in ctx.skills._revealed

        # After exit: skill[0] restored, skill[1] gone
        assert 0 in ctx.skills._revealed
        assert 1 not in ctx.skills._revealed

    @pytest.mark.asyncio
    async def test_snapshot_clear_all_restores_on_exception(self):
        """Revealed state is restored even when exception is raised inside snapshot block."""
        ctx = _make_context()
        ctx.get_details("skills", 0)

        try:
            async with ctx.snapshot(goal="sub-goal"):
                ctx.get_details("skills", 1)
                raise ValueError("test error")
        except ValueError:
            pass

        # skill[0] restored, skill[1] gone
        assert 0 in ctx.skills._revealed
        assert 1 not in ctx.skills._revealed

    @pytest.mark.asyncio
    async def test_snapshot_custom_keep_revealed(self):
        """Custom keep_revealed dict preserves specified indices on enter."""
        ctx = _make_context()
        # Reveal skills 0, 1 before entering
        ctx.get_details("skills", 0)
        ctx.get_details("skills", 1)
        assert 0 in ctx.skills._revealed
        assert 1 in ctx.skills._revealed

        # Enter with keep_revealed={"skills": [0]} — keep only index 0
        async with ctx.snapshot(goal="sub-goal", keep_revealed={"skills": [0]}):
            assert 0 in ctx.skills._revealed
            assert 1 not in ctx.skills._revealed

        # After exit: both 0 and 1 restored
        assert 0 in ctx.skills._revealed
        assert 1 in ctx.skills._revealed

    @pytest.mark.asyncio
    async def test_snapshot_goal_field_restored(self):
        """snapshot() also restores field values like override() does."""
        ctx = CognitiveContext(goal="original goal")
        async with ctx.snapshot(goal="sub-goal"):
            assert ctx.goal == "sub-goal"
        assert ctx.goal == "original goal"

    @pytest.mark.asyncio
    async def test_override_is_alias_for_snapshot_clear_all(self):
        """override() is a backward-compatible alias that clears revealed on enter."""
        ctx = _make_context()
        ctx.get_details("skills", 0)

        async with ctx.override(goal="sub"):
            assert 0 not in ctx.skills._revealed

        assert 0 in ctx.skills._revealed


class TestContextOverride:
    """Tests for Feature 2: Context.override() async context manager."""

    @pytest.mark.asyncio
    async def test_override_applies_fields(self):
        """override() applies new values during the async with block."""
        ctx = CognitiveContext(goal="original goal")

        async with ctx.override(goal="overridden goal") as c:
            assert c is ctx
            assert ctx.goal == "overridden goal"

    @pytest.mark.asyncio
    async def test_override_restores_on_success(self):
        """Fields are restored after normal exit from override block."""
        ctx = CognitiveContext(goal="original goal")

        async with ctx.override(goal="temp goal"):
            pass

        assert ctx.goal == "original goal"

    @pytest.mark.asyncio
    async def test_override_restores_on_exception(self):
        """Fields are restored even when an exception is raised inside the block."""
        ctx = CognitiveContext(goal="original goal")

        try:
            async with ctx.override(goal="temp goal"):
                assert ctx.goal == "temp goal"
                raise ValueError("test error")
        except ValueError:
            pass

        assert ctx.goal == "original goal"

    @pytest.mark.asyncio
    async def test_override_multiple_fields(self):
        """override() can temporarily set multiple fields at once."""
        ctx = CognitiveContext(goal="original", last_step_has_tools=False)

        async with ctx.override(goal="phase goal", last_step_has_tools=True):
            assert ctx.goal == "phase goal"
            assert ctx.last_step_has_tools is True

        assert ctx.goal == "original"
        assert ctx.last_step_has_tools is False
