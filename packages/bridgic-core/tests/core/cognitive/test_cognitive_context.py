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
        """summary() structure, status transitions, and all format_summary() modes."""
        ctx = _make_context()

        # --- summary() returns dict with all expected keys ---
        summary = ctx.summary()
        assert isinstance(summary, dict)
        assert all(key in summary for key in ["goal", "status", "tools", "skills", "cognitive_history"])
        assert "Plan a trip to Tokyo" in summary["goal"]
        assert isinstance(summary["tools"], str)
        assert isinstance(summary["skills"], str)
        assert isinstance(summary["cognitive_history"], str)

        # --- Status transitions ---
        assert summary["status"] == "Status: In Progress"

        ctx.set_finish()
        summary = ctx.summary()
        assert summary["status"] == "Status: Completed"

        # --- format_summary(include=...) ---
        result = ctx.format_summary(include=["goal", "status"])
        assert "Plan a trip to Tokyo" in result
        assert "Status:" in result
        assert "Available Tools" not in result
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
        assert "Status:" in result
        assert "Tools" in result or "tools" in result.lower()

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
