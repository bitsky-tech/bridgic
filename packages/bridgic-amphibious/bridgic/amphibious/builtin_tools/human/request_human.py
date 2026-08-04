"""
Human request built-in tool.

Uses ``contextvars.ContextVar`` for late-binding to the running agent,
so the exported ``request_human_tool`` is a plain ``FunctionToolSpec``
that can be used exactly like any other tool — no factory call needed.

Nothing is auto-injected by the framework anymore; a context that wants
human-in-the-loop **declares** this tool on its class via the
``OTAContext.tool`` registry, so the LLM can call ``request_human`` in any
mode (AGENT, WORKFLOW fallback, AMPHIFLOW)::

    from bridgic.amphibious.builtin_tools import request_human_tool

    MyOTAContext.tool(request_human_tool)
"""

from typing import Optional

from bridgic.core.agentic.tool_specs import FunctionToolSpec

from bridgic.amphibious.builtin_tools._agent_state import current_agent


async def request_human(prompt: str, channel: Optional[str] = None) -> str:
    """Ask the human operator a question and wait for their response.

    Use this tool when you need clarification, confirmation, or any
    information that only the human can provide.

    Parameters
    ----------
    prompt : str
        The question or message to present to the human.
    channel : Optional[str], default None
        Name of a registered ``@human_channel`` to route this question
        through (for example ``"feishu"``, ``"slack"``, ``"terminal"``).
        Leave as ``None`` to use the implicit default: the framework's
        stdin fallback when no channels are registered, or the sole
        registered channel. **When multiple channels are registered this
        argument is required** — omitting it raises an ambiguity error.
        The accepted names match the keys passed to ``@human_channel(...)``
        on the agent class.
    """
    agent = current_agent.get(None)
    if agent is None:
        raise RuntimeError(
            "request_human can only be called during agent execution. "
            "Ensure the tool is used within an AmphibiousAutoma.arun() context."
        )
    # Route through the framework's @human_channel registry by
    # constructing a HumanCall item and handing off to the canonical
    # ``_run_human_call`` driver (which also emits the
    # ``[Human Interaction]`` header + ``-> result:`` arrow).
    from bridgic.amphibious._type import HumanCall
    return await agent._run_human_call(
        HumanCall(prompt=prompt, channel=channel),
    )


request_human_tool: FunctionToolSpec = FunctionToolSpec.from_raw(request_human)
