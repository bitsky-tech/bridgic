"""Tests for CognitiveContext: summary, format_summary, disclosed_details, skills.add()."""
import os
import pytest

from bridgic.amphibious import CognitiveContext, CognitiveSkills, Skill, Step
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
