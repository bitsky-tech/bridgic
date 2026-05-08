"""
Human request built-in tool.

Uses ``contextvars.ContextVar`` for late-binding to the running agent,
so the exported ``request_human_tool`` is a plain ``FunctionToolSpec``
that can be used exactly like any other tool — no factory call needed.

``AmphibiousAutoma.arun()`` auto-injects this tool into every agent's
``context.tools``, so the LLM can call ``request_human`` in any mode
(AGENT, WORKFLOW fallback, AMPHIFLOW) with no extra wiring::

    # No need to pass request_human_tool — it is already available as `request_human`.
    await agent.arun(goal="...", tools=[search_tool])

If you want to be explicit, importing and passing ``request_human_tool``
still works — the injection step deduplicates by tool name::

    from bridgic.amphibious.builtin_tools import request_human_tool

    await agent.arun(goal="...", tools=[search_tool, request_human_tool])  # also fine
"""

from bridgic.core.agentic.tool_specs import FunctionToolSpec

from bridgic.amphibious.builtin_tools._agent_state import current_agent


async def request_human(prompt: str) -> str:
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
            "request_human can only be called during agent execution. "
            "Ensure the tool is used within an AmphibiousAutoma.arun() context."
        )
    # Route through the framework's @human_channel registry. With no
    # channels registered, this falls through to the stdin handler;
    # with one registered, that channel is the implicit default; with
    # multiple, the agent must register a default channel for the tool
    # to route deterministically.
    return await agent._dispatch_human_channel(prompt, channel=None)


request_human_tool: FunctionToolSpec = FunctionToolSpec.from_raw(request_human)
