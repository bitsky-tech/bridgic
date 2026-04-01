"""
Human request built-in tool.

Uses ``contextvars.ContextVar`` for late-binding to the running agent,
so the exported ``human_request_tool`` is a plain ``FunctionToolSpec``
that can be used exactly like any other tool — no factory call needed.

Usage::

    from bridgic.amphibious.buildin_tools import human_request_tool

    await agent.arun(goal="...", tools=[search_tool, human_request_tool])
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING

from bridgic.core.agentic.tool_specs import FunctionToolSpec

if TYPE_CHECKING:
    from bridgic.amphibious._amphibious_automa import AmphibiousAutoma

# Task-scoped variable — each asyncio Task gets its own value,
# so concurrent agents never interfere with each other.
current_agent: ContextVar["AmphibiousAutoma"] = ContextVar("current_agent")


async def ask_human(prompt: str) -> str:
    """Ask the human operator a question and wait for their response.

    Use this tool when you need clarification, confirmation, or any
    information that only the human can provide.

    Parameters
    ----------
    prompt : str
        The question or message to present to the human.
    """
    agent = current_agent.get(None)
    if agent is None:
        raise RuntimeError(
            "ask_human can only be called during agent execution. "
            "Ensure the tool is used within an AmphibiousAutoma.arun() context."
        )
    return await agent.request_human(prompt)


human_request_tool: FunctionToolSpec = FunctionToolSpec.from_raw(ask_human)
