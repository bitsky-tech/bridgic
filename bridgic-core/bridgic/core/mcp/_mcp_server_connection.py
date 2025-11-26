from typing import Any
from abc import ABC, abstractmethod
from contextlib import _AsyncGeneratorContextManager, AsyncExitStack
from typing import List, Dict, Optional, Union, Any
from datetime import timedelta
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client
from bridgic.core.types._error import McpServerConnectionError

class McpServerConnection(ABC):
    """
    The abstract base class for Connection to an MCP server.

    This class is responsible for establishing a connection to an MCP server and providing a 
    session to interact with the server. The connection can be established using different 
    transport protocols, which depends on the specific implementation.

    Methods
    -------
    connect
        Establish a connection to an MCP server.
    get_mcp_client
        Get a MCP client to interact with the server.
    """
    name: str
    """The name of the connected MCP server."""

    request_timeout: int
    """The timeout in seconds for the requests to the MCP server."""

    encoding: str
    """The encoding to use for the connection."""

    client_kwargs: Dict[str, Any]
    """The keyword arguments to pass to the MCP client."""

    session: ClientSession
    """The session that is used to interact with the MCP server."""

    _exit_stack: AsyncExitStack
    """The exit stack for the connection."""

    def __init__(
        self,
        name: str,
        *,
        request_timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        self.name = name
        self.request_timeout = request_timeout or 60
        self.client_kwargs = kwargs

        self.session = None
        self.is_connected = False

        self._exit_stack = AsyncExitStack()

    async def connect(self):
        """
        Establish a connection to the MCP server.
        """
        if not self.session:
            # Try to establish a connection to the specified server.
            try:
                transport = await self._exit_stack.enter_async_context(self.get_mcp_client())
                session = await self._exit_stack.enter_async_context(
                    ClientSession(
                        read_stream=transport[0],
                        write_stream=transport[1],
                        read_timeout_seconds=timedelta(seconds=self.request_timeout),
                        # TODO : Callback will be added in the future to support more advanced features.
                        # message_handler=...,
                        # logging_callback=...,
                        # sampling_callback=...,
                    )
                )
                await session.initialize()
                self.is_connected = True
            except Exception as ex:
                await self._exit_stack.aclose()
                raise McpServerConnectionError(f"Failed to create session to MCP server: name={self.name}, error={ex}") from ex
            # Hold the connected session for later use.
            self.session = session
        elif self.session._request_id == 0:
            await self.session.initialize()
            self.is_connected = True

    async def close(self):
        """
        Close the connection to the MCP server.
        """
        await self._exit_stack.aclose()
        self.session = None
        self.is_connected = False

    @abstractmethod
    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """
        Get an MCP client.

        Returns
        -------
        _AsyncGeneratorContextManager[Any, None]
            An async context manager for the MCP client transport.
        """
        ...


class McpServerConnectionStdio(McpServerConnection):
    """
    The connection to an MCP server using stdio.
    """
    def __init__(
        self,
        name: str,
        command: str,
        *,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        encoding: Optional[str] = None,
        request_timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            request_timeout=request_timeout,
            **kwargs,
        )
        self.command = command
        self.encoding = encoding or "utf-8"
        self.args = args
        self.env = env

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """
        Get an MCP client transport for stdio.

        Returns
        -------
        _AsyncGeneratorContextManager[Any, None]
            An async context manager for the stdio client transport.
        """
        start_args = {
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }
        if self.encoding:
            start_args["encoding"] = self.encoding
        if self.client_kwargs:
            start_args.update(self.client_kwargs)

        return stdio_client(server=StdioServerParameters(**start_args))

class McpServerConnectionStreamableHttp(McpServerConnection):
    """
    The connection to an MCP server using streamable http.
    """
    url: str
    """The URL of the MCP server."""

    def __init__(
        self,
        name: str,
        url: str,
        *,
        headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        terminate_on_close: Optional[bool] = None,
        sse_read_timeout: Optional[float] = None,
        request_timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name,
            request_timeout=request_timeout,
            **kwargs,
        )
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.terminate_on_close = terminate_on_close
        self.sse_read_timeout = sse_read_timeout

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """
        Get an MCP client transport for streamable http.

        Returns
        -------
        _AsyncGeneratorContextManager[Any, None]
            An async context manager for the streamable HTTP client transport.
        """
        start_args = {
            "url": self.url,
        }
        if self.headers:
            start_args["headers"] = self.headers
        if self.timeout is not None:
            start_args["timeout"] = self.timeout
        if self.terminate_on_close is not None:
            start_args["terminate_on_close"] = self.terminate_on_close
        if self.sse_read_timeout is not None:
            start_args["sse_read_timeout"] = self.sse_read_timeout
        if self.client_kwargs:
            start_args.update(self.client_kwargs)

        return streamablehttp_client(**start_args)