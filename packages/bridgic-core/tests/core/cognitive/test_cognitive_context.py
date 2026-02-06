"""Tests for CognitiveContext: summary, format_summary, disclosed_details."""
import os
import pytest

from bridgic.core.cognitive import CognitiveContext, Step
from .tools import get_travel_planning_tools

SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def _make_context() -> CognitiveContext:
    """Create a context with tools, skills and some history."""
    ctx = CognitiveContext(goal="Plan a trip to Tokyo")

    # Add tools
    for tool in get_travel_planning_tools():
        ctx.tools.add(tool)

    # Add skills
    ctx.skills.load_from_directory(SKILLS_DIR)

    # Add history
    ctx.add_info(Step(content="Search flights", status=True, result="Found 3 flights"))
    ctx.add_info(Step(content="Book flight CA123", status=True, result="Booking confirmed"))

    return ctx


class TestSummaryBasic:
    def test_summary_basic(self):
        ctx = _make_context()
        summary = ctx.summary()

        assert isinstance(summary, dict)
        assert "goal" in summary
        assert "status" in summary
        assert "tools" in summary
        assert "skills" in summary
        assert "cognitive_history" in summary

        # Values are formatted strings
        assert "Plan a trip to Tokyo" in summary["goal"]
        assert isinstance(summary["tools"], str)
        assert isinstance(summary["skills"], str)
        assert isinstance(summary["cognitive_history"], str)


class TestSummaryStatus:
    def test_status_in_progress(self):
        ctx = CognitiveContext(goal="test")
        summary = ctx.summary()
        assert summary["status"] == "Status: In Progress"

    def test_status_completed(self):
        ctx = CognitiveContext(goal="test")
        ctx.set_finish()
        summary = ctx.summary()
        assert summary["status"] == "Status: Completed"


class TestFormatSummaryInclude:
    def test_format_summary_include(self):
        ctx = _make_context()
        result = ctx.format_summary(include=["goal", "status"])

        assert "Plan a trip to Tokyo" in result
        assert "Status:" in result
        # Should NOT contain tools/skills/history
        assert "Available Tools" not in result
        assert "Available Skills" not in result


class TestFormatSummaryExclude:
    def test_format_summary_exclude(self):
        ctx = _make_context()
        result = ctx.format_summary(exclude=["tools"])

        assert "Plan a trip to Tokyo" in result
        assert "Status:" in result
        # tools should be excluded
        assert "Available Tools" not in result
        # skills/history should remain
        assert "Skills" in result or "skills" in result.lower()


class TestFormatSummaryDefault:
    def test_format_summary_default(self):
        ctx = _make_context()
        result = ctx.format_summary()

        # All fields should be present
        assert "Plan a trip to Tokyo" in result
        assert "Status:" in result
        assert "Tools" in result or "tools" in result.lower()


class TestDisclosedDetails:
    def test_disclosed_details(self):
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
