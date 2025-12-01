from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Dict, Optional, Union, Any, TYPE_CHECKING
from contextlib import _AsyncGeneratorContextManager, AsyncExitStack
from mcp.client.session import ClientSession
from mcp.types import ListPromptsResult, GetPromptResult, ListToolsResult, CallToolResult
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client

from bridgic.core.types._error import McpServerConnectionError
from bridgic.core.mcp._mcp_server_connection_manager import McpServerConnectionManager

if TYPE_CHECKING:
    from bridgic.core.prompt._mcp_template import McpPromptTemplate
    from bridgic.core.agentic.tool_specs._mcp_tool_spec import McpToolSpec

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
    """The timeout in seconds for the requests to the MCP server. Default is 30 seconds."""

    encoding: str
    """The encoding to use for the connection."""

    client_kwargs: Dict[str, Any]
    """The keyword arguments to pass to the MCP client."""

    _manager: McpServerConnectionManager
    """The manager that is used to manage the connection."""

    _session: ClientSession
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
        self.request_timeout = request_timeout or 30
        self.client_kwargs = kwargs

        self._manager = None
        self._session = None

        self.is_connected = False

    def _get_manager(self) -> McpServerConnectionManager:
        if self._manager is None:
            manager = McpServerConnectionManager.get_instance()
            manager.register_connection(self)
            assert manager is self._manager
        return self._manager

    def connect(self):
        """
        Establishe a connection to the MCP server. Call this method once before using the connection.

        If the connection is not registered in a specific manager explicitly, it will be registered
        in the default manager (manager_name="default-mcp-manager"). If the connection needs to be
        registered in a specific manager, the `connect` method should be called after the registration.

        Notes
        -----
        The event loop responsible for managing the session is determined at the time when `connect()` is called.
        Therefore, it is required to register the connection to the desired manager *before* calling `connect()`.
        Otherwise, the connection will be registered to the default manager. All registrations could not be changed later.

        Example
        -------
        >>> connection = McpServerConnectionStreamableHttp(
        ...     name="streamable-http-server-connection",
        ...     url="http://localhost:8000",
        ...     request_timeout=5,
        ... )
        >>> manager = McpServerConnectionManager.get_instance("my-manager")
        >>> manager.register_connection(connection)
        >>> connection.connect()
        """
        self._get_manager().run_sync(
            coro=self._connect_unsafe(),
            timeout=self.request_timeout + 1,
        )

    def close(self):
        """
        Close the connection to the MCP server.
        """
        if self._manager is None:
            return

        manager = self._get_manager()
        manager.run_sync(
            coro=self._close_unsafe(),
            timeout=self.request_timeout + 1,
        )
        manager.unregister_connection(self)

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

    ###########################################################################
    # Protected methods that should be called within the dedicated event loop.
    ###########################################################################

    async def _connect_unsafe(self):
        """
        Asynchronously establish a connection to the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self._session:
            # Try to establish a connection to the specified server.
            try:
                self._exit_stack = AsyncExitStack()
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
            except Exception as ex:
                await self._exit_stack.aclose()
                raise McpServerConnectionError(f"Failed to create session to MCP server: name={self.name}, error={ex}") from ex
            # Hold the connected session for later use.
            self._session = session
            self.is_connected = True
        elif self._session._request_id == 0:
            await self._session.initialize()
            self.is_connected = True

    async def _close_unsafe(self):
        """
        Asynchronously close the connection to the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        await self._exit_stack.aclose()
        self._session = None
        self.is_connected = False

    async def _list_prompts_unsafe(self) -> ListPromptsResult:
        """
        Asynchronously list the prompts from the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected:
            await self._connect_unsafe()
        return await self._session.list_prompts()

    async def _get_prompt_unsafe(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> GetPromptResult:
        """
        Asynchronously get a prompt from the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected:
            await self._connect_unsafe()
        return await self._session.get_prompt(name=prompt_name, arguments=arguments or {})

    async def _list_tools_unsafe(self) -> ListToolsResult:
        """
        Asynchronously list the tools from the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected:
            await self._connect_unsafe()
        return await self._session.list_tools()

    async def _call_tool_unsafe(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> CallToolResult:
        """
        Asynchronously call a tool on the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected:
            await self._connect_unsafe()
        return await self._session.call_tool(name=tool_name, arguments=arguments or {})

    ###########################################################################
    # Public methods that are safely wrapped and could be called anywhere.
    ###########################################################################

    def list_prompts(self) -> List["McpPromptTemplate"]:
        """
        List the prompts from the MCP server.

        Returns
        -------
        List[McpPromptTemplate]
            The list of prompt template instances from the server.
        """
        from bridgic.core.prompt._mcp_template import McpPromptTemplate

        result = self._get_manager().run_sync(
            coro=self._list_prompts_unsafe(),
            timeout=self.request_timeout + 1,
        )

        return [
            McpPromptTemplate(
                prompt_name=prompt.name,
                prompt_info=prompt,
                server_connection=self
            )
            for prompt in result.prompts
        ]

    async def alist_prompts(self) -> List["McpPromptTemplate"]:
        """
        Asynchronously list the prompts from the MCP server.

        Returns
        -------
        List[McpPromptTemplate]
            The list of prompt template instances from the server.
        """
        from bridgic.core.prompt._mcp_template import McpPromptTemplate

        result = await self._get_manager().run_async(
            coro=self._list_prompts_unsafe(),
            timeout=self.request_timeout + 1,
        )

        return [
            McpPromptTemplate(
                prompt_name=prompt.name,
                prompt_info=prompt,
                server_connection=self
            )
            for prompt in result.prompts
        ]

    def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> GetPromptResult:
        """
        Synchronously get a prompt from the MCP server.

        Parameters
        ----------
        prompt_name : str
            The name of the prompt to retrieve.
        arguments : Optional[Dict[str, Any]]
            Arguments to pass to the prompt.

        Returns
        -------
        GetPromptResult
            The prompt result from the server.

        Raises
        ------
        RuntimeError
            If the connection is not established.
        """
        return self._get_manager().run_sync(
            coro=self._get_prompt_unsafe(prompt_name=prompt_name, arguments=arguments),
            timeout=self.request_timeout + 1,
        )

    async def aget_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> GetPromptResult:
        """
        Asynchronously get a prompt from the MCP server.

        Parameters
        ----------
        prompt_name : str
            The name of the prompt to retrieve.
        arguments : Optional[Dict[str, Any]]
            Arguments to pass to the prompt.

        Returns
        -------
        GetPromptResult
            The prompt result from the server.

        Raises
        ------
        RuntimeError
            If the connection is not established.
        """
        return await self._get_manager().run_async(
            coro=self._get_prompt_unsafe(prompt_name=prompt_name, arguments=arguments),
            timeout=self.request_timeout + 1,
        )

    def list_tools(self) -> List["McpToolSpec"]:
        """
        List the tools from the MCP server.

        This method synchronously retrieves the list of tools available from the connected
        MCP server and wraps each tool in an `McpToolSpec` instance for use within the
        bridgic framework.

        Returns
        -------
        List[McpToolSpec]
            The list of tool specification instances from the server.

        Raises
        ------
        RuntimeError
            If the connection is not established and cannot be established.
        """
        from bridgic.core.agentic.tool_specs._mcp_tool_spec import McpToolSpec

        result = self._get_manager().run_sync(
            coro=self._list_tools_unsafe(),
            timeout=self.request_timeout + 1,
        )

        return [
            McpToolSpec(
                tool_name=tool.name,
                tool_info=tool,
                server_connection=self
            )
            for tool in result.tools
        ]

    async def alist_tools(self) -> List["McpToolSpec"]:
        """
        Asynchronously list the tools from the MCP server.

        This method asynchronously retrieves the list of tools available from the connected
        MCP server and wraps each tool in an `McpToolSpec` instance for use within the
        bridgic framework.

        Returns
        -------
        List[McpToolSpec]
            The list of tool specification instances from the server.

        Raises
        ------
        RuntimeError
            If the connection is not established and cannot be established.
        """
        from bridgic.core.agentic.tool_specs._mcp_tool_spec import McpToolSpec

        result = await self._get_manager().run_async(
            coro=self._list_tools_unsafe(),
            timeout=self.request_timeout + 1,
        )

        return [
            McpToolSpec(
                tool_name=tool.name,
                tool_info=tool,
                server_connection=self
            )
            for tool in result.tools
        ]

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> CallToolResult:
        """
        Synchronously call a tool on the MCP server.

        This method synchronously invokes a tool on the connected MCP server with the
        specified arguments and returns the result.

        Parameters
        ----------
        tool_name : str
            The name of the tool to call.
        arguments : Optional[Dict[str, Any]]
            The arguments to pass to the tool. If None, an empty dictionary will be used.

        Returns
        -------
        CallToolResult
            The result of the tool call from the server, containing content and optionally
            structured content.

        Raises
        ------
        RuntimeError
            If the connection is not established and cannot be established.
        """
        return self._get_manager().run_sync(
            coro=self._call_tool_unsafe(tool_name=tool_name, arguments=arguments),
            timeout=self.request_timeout + 1,
        )

    async def acall_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> CallToolResult:
        """
        Asynchronously call a tool on the MCP server.

        This method asynchronously invokes a tool on the connected MCP server with the
        specified arguments and returns the result.

        Parameters
        ----------
        tool_name : str
            The name of the tool to call.
        arguments : Optional[Dict[str, Any]]
            The arguments to pass to the tool. If None, an empty dictionary will be used.

        Returns
        -------
        CallToolResult
            The result of the tool call from the server, containing content and optionally
            structured content.

        Raises
        ------
        RuntimeError
            If the connection is not established and cannot be established.
        """
        return await self._get_manager().run_async(
            coro=self._call_tool_unsafe(tool_name=tool_name, arguments=arguments),
            timeout=self.request_timeout + 1,
        )


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