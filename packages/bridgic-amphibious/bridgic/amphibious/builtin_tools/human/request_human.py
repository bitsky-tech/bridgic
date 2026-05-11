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

from copy import deepcopy
from typing import Iterable, Optional

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
    # Route through the framework's @human_channel registry. With no
    # channels registered, this falls through to the stdin handler;
    # with one registered, that channel is the implicit default; with
    # multiple, the LLM must pass an explicit ``channel`` matching one
    # of the registered ``@human_channel`` names for routing to be
    # deterministic.
    return await agent._dispatch_human_channel(prompt, channel=channel)


request_human_tool: FunctionToolSpec = FunctionToolSpec.from_raw(request_human)


def build_request_human_tool(
    channel_names: Optional[Iterable[str]] = None,
) -> FunctionToolSpec:
    """Build a ``request_human`` ``FunctionToolSpec`` specialised for an
    agent's registered ``@human_channel`` keys.

    When an agent class registers one or more channels, the JSON schema
    advertised to the LLM is customised so that:

    * the top-level tool description lists the agent's registered
      channel names verbatim, and
    * the ``channel`` parameter's schema is constrained to an ``enum``
      of those exact names — so the LLM cannot hallucinate a name that
      will be rejected by ``_dispatch_human_channel``.

    Parameters
    ----------
    channel_names : Optional[Iterable[str]]
        Sorted/unsorted iterable of channel keys from the agent class's
        ``_human_channels`` registry. ``None`` or empty → returns the
        generic module-level ``request_human_tool`` unchanged (the
        stdin-fallback case, where no channels are registered).

    Returns
    -------
    FunctionToolSpec
        A new ``FunctionToolSpec`` (does not mutate the shared static
        spec) bound to the same underlying ``request_human`` function.
    """
    names = sorted({n for n in (channel_names or []) if n})
    if not names:
        return request_human_tool

    base_params = deepcopy(request_human_tool.tool_parameters or {})
    props = base_params.setdefault("properties", {})
    channel_prop = props.get("channel")
    if channel_prop is None:
        # Defensive: schema lost the channel field somehow — fall back
        # to the generic spec rather than silently shipping a broken one.
        return request_human_tool
    channel_prop["enum"] = names
    channel_prop["description"] = (
        "Name of the @human_channel to route this question through. "
        f"Registered on this agent: {names}. "
        "Required when 2+ channels are registered; with a single channel "
        "it is optional (defaults to that channel)."
    )

    base_desc = request_human_tool.tool_description or ""
    custom_desc = (
        f"{base_desc}\n\n"
        f"Registered @human_channel keys on this agent: {names}. "
        "Pass `channel=\"<name>\"` to target one explicitly; omit it "
        "only when exactly one channel is registered."
    )

    return FunctionToolSpec.from_raw(
        request_human,
        tool_description=custom_desc,
        tool_parameters=base_params,
    )
