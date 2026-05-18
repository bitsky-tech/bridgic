"""In-process FastMCP HTTP host used by the ThinkAgent runtime.

Exposes a set of callables as MCP tools over HTTP. Runs in the **same
asyncio event loop** as the parent ``AmphibiousAutoma`` (mounted via
``uvicorn.Server.serve()`` as a task) so tool handlers have full closure
access to the agent's ``self`` and ``ctx``. The ``agent_done`` completion
signal is owned by the host as a first-class future.

This module is itself imported lazily from ``_think_agent.py`` (only on
the first ``ThinkAgent`` dispatch).

NOTE: ``from __future__ import annotations`` is intentionally NOT used
here. ``_build_handler`` materialises a function via ``exec`` and needs
the generated signature's annotations as real type objects (not strings)
so ``inspect.signature(handler)`` sees the proper types.
"""

import asyncio
import socket
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastmcp import FastMCP
import uvicorn


ToolCallback = Callable[[str, Dict[str, Any]], Awaitable[Any]]
AgentDoneCallback = Callable[[str], None]


@dataclass
class MCPToolBinding:
    """One project tool to expose via MCP.

    Attributes
    ----------
    name : str
        Tool name as claude will see it (the local name; FastMCP will
        prefix with ``mcp__<server_name>__``).
    description : str
        Free-text description shown to claude.
    parameters : Dict[str, Any]
        JSON Schema object describing the tool's arguments. Pass the
        bridgic ``ToolSpec.tool_parameters`` directly — it is already
        in JSON-Schema form.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


class MCPHost:
    """A FastMCP host bound to the running asyncio loop.

    Lifecycle: ``start()`` → ``url`` is populated and the server is
    listening; ``stop()`` → server shuts down gracefully and the
    background task is joined.
    """

    def __init__(
        self,
        *,
        server_name: str,
        bindings: List[MCPToolBinding],
        on_tool_call: ToolCallback,
        on_agent_done: AgentDoneCallback,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.server_name = server_name
        self.bindings = bindings
        self.on_tool_call = on_tool_call
        self.on_agent_done = on_agent_done
        self.host = host
        self._desired_port = port

        self.port: Optional[int] = None
        self.url: Optional[str] = None

        self._uv_server: Optional[uvicorn.Server] = None
        self._serve_task: Optional[asyncio.Task] = None
        self._mcp: Optional[FastMCP] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        app = self._build_app()
        self.port = self._reserve_port()

        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="on",
        )
        self._uv_server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._uv_server.serve())

        await self._await_started(timeout=15.0)
        self.url = f"http://{self.host}:{self.port}/mcp"

    async def stop(self) -> None:
        if self._uv_server is not None:
            self._uv_server.should_exit = True
        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._serve_task.cancel()
                try:
                    await self._serve_task
                except (asyncio.CancelledError, Exception):
                    pass
        self._uv_server = None
        self._serve_task = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_app(self):
        mcp = FastMCP(self.server_name)
        self._mcp = mcp

        on_tool_call = self.on_tool_call
        on_agent_done = self.on_agent_done

        # Project tool handlers — one per binding, generated with a
        # signature that mirrors the binding's parameter list so FastMCP
        # can extract the schema by introspection.
        for binding in self.bindings:
            handler = _build_handler(binding, on_tool_call)
            mcp.tool(
                name=binding.name,
                description=binding.description,
            )(handler)

        # Completion signal tool.
        @mcp.tool(
            name="agent_done",
            description=(
                "Call this exactly once when the goal is fully complete. "
                "Pass the final answer / summary as the `result` argument. "
                "The host will resume the parent automa with the value you pass."
            ),
        )
        async def agent_done(result: str) -> str:
            on_agent_done(result)
            return "Acknowledged. Goal recorded as complete; you may finish now."

        return mcp.http_app()

    def _reserve_port(self) -> int:
        if self._desired_port != 0:
            return self._desired_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            return s.getsockname()[1]

    async def _await_started(self, *, timeout: float) -> None:
        assert self._uv_server is not None
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._uv_server.started:
                return
            await asyncio.sleep(0.1)
        raise RuntimeError(
            f"MCP server did not start within {timeout}s (port={self.port})"
        )


# ----------------------------------------------------------------------
# Handler factory — builds a handler whose signature matches the
# binding's parameter list so FastMCP's introspection picks up the
# schema correctly.
# ----------------------------------------------------------------------

# Map JSON-Schema primitive types to Python builtin type names. Anything
# unrecognised (or absent) falls back to ``str``, which is the most
# permissive choice for FastMCP's schema inference and matches the
# claude-side stringification of MCP arguments.
_PYTHON_TYPE_FOR_JSON: Dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _python_type_name(schema: Any) -> str:
    """Resolve a JSON-Schema property's ``type`` into a Python type name."""
    if not isinstance(schema, dict):
        return "str"
    json_type = schema.get("type")
    if isinstance(json_type, list):
        # Union type (e.g. ``["string", "null"]``) — pick the first
        # non-null entry. Good enough for the common Optional[T] case.
        for t in json_type:
            if t != "null":
                return _PYTHON_TYPE_FOR_JSON.get(t, "str")
        return "str"
    if not isinstance(json_type, str):
        return "str"
    return _PYTHON_TYPE_FOR_JSON.get(json_type, "str")


def _default_literal_for(type_name: str) -> str:
    """Python literal expression for the default value of a given type."""
    return {
        "str": "''",
        "int": "0",
        "float": "0.0",
        "bool": "False",
        "list": "[]",
        "dict": "{}",
    }.get(type_name, "None")


def _build_handler(
    binding: MCPToolBinding,
    on_tool_call: ToolCallback,
):
    """Construct an async handler whose Python signature mirrors the
    binding's JSON-Schema parameters (types + required-vs-optional).

    FastMCP introspects the handler's signature to build the MCP tool
    schema it advertises to claude. To honour the binding's declared
    schema, we materialise a real Python function via ``exec``:

    * **Required** parameters appear without default values, so FastMCP
      marks them ``required`` in its advertised schema and the model
      can be relied on to send them.
    * **Optional** parameters get a default of the right shape for
      their type (e.g. ``int = 0``, ``str = ''``).
    * **Type annotations** mirror the JSON-Schema ``type`` so the
      model sees ``integer`` / ``number`` / ``boolean`` instead of
      everything stringified.

    FastMCP 3.x does not offer an "imperative add_tool with an explicit
    JSON-Schema dict" form that plugs into the same place the decorator
    does, so this is the cleanest path: we trust FastMCP's signature
    introspection but feed it a signature that mirrors the bridgic
    ToolSpec.
    """
    if not isinstance(binding.parameters, dict):
        properties: Dict[str, Any] = {}
        required: set[str] = set()
    else:
        properties = binding.parameters.get("properties") or {}
        required = set(binding.parameters.get("required") or [])

    # Required-first: Python syntax forbids non-default args after
    # default args, so we order the signature accordingly.
    required_names = [n for n in properties.keys() if n in required]
    optional_names = [n for n in properties.keys() if n not in required]
    arg_names = required_names + optional_names

    sig_parts: List[str] = []
    for n in required_names:
        tname = _python_type_name(properties.get(n))
        sig_parts.append(f"{n}: {tname}")
    for n in optional_names:
        tname = _python_type_name(properties.get(n))
        default = _default_literal_for(tname)
        sig_parts.append(f"{n}: {tname} = {default}")

    arg_sig = ", ".join(sig_parts)
    arg_packing = ", ".join(f"{n!r}: {n}" for n in arg_names)

    src = (
        f"async def handler({arg_sig}) -> str:\n"
        f"    args = {{{arg_packing}}}\n"
        f"    result = await _cb({binding.name!r}, args)\n"
        f"    return '' if result is None else str(result)\n"
    )
    ns: Dict[str, Any] = {"_cb": on_tool_call}
    exec(src, ns)
    handler = ns["handler"]
    handler.__name__ = binding.name
    handler.__doc__ = binding.description
    return handler


__all__ = [
    "MCPHost",
    "MCPToolBinding",
    "ToolCallback",
    "AgentDoneCallback",
]
