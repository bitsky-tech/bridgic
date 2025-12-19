import asyncio
import warnings
import httpx
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Dict, Optional, Union, Any, TYPE_CHECKING
from contextlib import _AsyncGeneratorContextManager, AsyncExitStack
from mcp.client.session import ClientSession
from mcp.types import ListPromptsResult, GetPromptResult, ListToolsResult, CallToolResult
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from bridgic.protocols.mcp._error import McpServerConnectionError
from bridgic.protocols.mcp._mcp_server_connection_manager import McpServerConnectionManager

if TYPE_CHECKING:
    from bridgic.protocols.mcp._mcp_template import McpPromptTemplate
    from bridgic.protocols.mcp._mcp_tool_spec import McpToolSpec

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
    _session: ClientSession
    _exit_stack: AsyncExitStack

    _lifecycle_task: asyncio.Task
    _connection_ready_event: asyncio.Event
    _connection_stop_event: asyncio.Event
    _connection_error: Exception

    is_connected: bool
    """Whether the connection is established."""

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
        self._exit_stack = None

        self._lifecycle_task = None
        self._connection_stop_event = asyncio.Event()
        self._connection_ready_event = asyncio.Event()

        self.is_connected = False
        self._connection_error = None

    def _get_manager(self) -> McpServerConnectionManager:
        if self._manager is None:
            manager = McpServerConnectionManager.get_instance()
            manager.register_connection(self)
            assert manager is self._manager
        return self._manager

    def connect(self):
        """
        Establish a connection to the MCP server. Call this method once before using the connection.

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
        if self.is_connected:
            return

        # Create events and lifecycle task in the manager's event loop.
        async def setup_and_wait():
            # Create events in the manager's event loop.
            self._connection_stop_event.clear()
            self._connection_ready_event.clear()
            self._connection_error = None

            # Create the lifecycle task.
            self._lifecycle_task = asyncio.create_task(self._lifecycle_task_coro())

            # Wait for connection to be ready
            await self._connection_ready_event.wait()

            # Check if there was an error
            if self._connection_error:
                raise self._connection_error

        # Setup the connection and make sure it could be closed within the same lifecycle task.
        self._get_manager().run_sync(
            coro=setup_and_wait(),
            timeout=self.request_timeout + 1,
        )

    def close(self):
        """
        Close the connection to the MCP server.
        """
        if not self.is_connected:
            return

        # Signal the lifecycle task to stop and wait for it to complete.
        async def stop_and_wait():
            if self._connection_stop_event is not None:
                self._connection_stop_event.set()

            if self._lifecycle_task is not None:
                try:
                    await self._lifecycle_task
                except Exception as e:
                    warnings.warn(f"Exception occurred while waiting for the lifecycle task to complete: {e}")
                finally:
                    self.is_connected = False
                    self._lifecycle_task = None
                    self._manager.unregister_connection(self)

        self._get_manager().run_sync(
            coro=stop_and_wait(),
            timeout=self.request_timeout + 1,
        )

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

    async def _lifecycle_task_coro(self):
        """
        The lifecycle task coroutine that manages the connection lifecycle.

        This coroutine runs in the manager's dedicated event loop and is responsible for:
        1. Establishing the connection to the MCP server
        2. Waiting for the stop event
        3. Closing the connection when the stop event is set

        This ensures that the connection is created and closed within the same Task,
        which is required by anyio's cancel scope mechanism.
        """
        try:
            # Establish connection
            self._exit_stack = AsyncExitStack()
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
            except Exception as ex:
                session = None
                self._connection_error = McpServerConnectionError(
                    f"Failed to create session to MCP server: name={self.name}"
                ).with_traceback(ex.__traceback__)

            # Hold the connected session for later use.
            self._session = session
            self.is_connected = True if session is not None else False

            # Signal that the connection is ready.
            self._connection_ready_event.set()

            # Wait for the stop signal.
            await self._connection_stop_event.wait()
        finally:
            # At the final moment, the exit stack must be closed.
            if self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

            self._session = None
            self.is_connected = False

    async def _list_prompts_unsafe(self) -> ListPromptsResult:
        """
        Asynchronously list the prompts from the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected or self._session is None:
            raise McpServerConnectionError(
                f"Connection to MCP server is not established: name={self.name}"
            )
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
        if not self.is_connected or self._session is None:
            raise McpServerConnectionError(
                f"Connection to MCP server is not established: name={self.name}"
            )
        return await self._session.get_prompt(name=prompt_name, arguments=arguments or {})

    async def _list_tools_unsafe(self) -> ListToolsResult:
        """
        Asynchronously list the tools from the MCP server.

        Since the session used to communicate with the MCP server is bound to a specific event 
        loop, this method should be called within the designated event loop for the connection.
        """
        if not self.is_connected or self._session is None:
            raise McpServerConnectionError(
                f"Connection to MCP server is not established: name={self.name}"
            )
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
        if not self.is_connected or self._session is None:
            raise McpServerConnectionError(
                f"Connection to MCP server is not established: name={self.name}"
            )
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
        from bridgic.protocols.mcp._mcp_template import McpPromptTemplate

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
        from bridgic.protocols.mcp._mcp_template import McpPromptTemplate

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
        from bridgic.protocols.mcp._mcp_tool_spec import McpToolSpec

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
        from bridgic.protocols.mcp._mcp_tool_spec import McpToolSpec

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

    command: str
    """The command to use for the connection."""

    encoding: str
    """The encoding to use for the connection."""

    args: List[str]
    """The arguments to use for the connection."""

    env: Dict[str, str]
    """The environment variables to use for the connection."""

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

    http_client: httpx.AsyncClient
    """The HTTP client to use for the connection."""

    terminate_on_close: bool
    """Whether to terminate the session when the connection is closed."""

    def __init__(
        self,
        name: str,
        url: str,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        terminate_on_close: Optional[bool] = None,
        request_timeout: Optional[int] = None,
    ):
        """
        Initialize a streamable HTTP connection to an MCP server.

        Parameters
        ----------
        name : str
            The name of the connection.
        url : str
            The URL of the MCP server.
        http_client : Optional[httpx.AsyncClient]
            Optional pre-configured httpx.AsyncClient. If None, a default
            client with recommended MCP timeouts will be created. To configure headers,
            authentication, or other HTTP settings, create an httpx.AsyncClient and pass it here.
        terminate_on_close : Optional[bool]
            If True, send a DELETE request to terminate the session when the connection
            is closed. Defaults to True.
        request_timeout : Optional[int]
            The timeout in seconds for MCP requests. Default is 30 seconds.
        """
        super().__init__(
            name,
            request_timeout=request_timeout,
        )
        self.url = url
        self.http_client = http_client
        self.terminate_on_close = terminate_on_close or True

    def get_mcp_client(self) -> _AsyncGeneratorContextManager[Any, None]:
        """
        Get an MCP client transport for streamable http.

        Returns
        -------
        _AsyncGeneratorContextManager[Any, None]
            An async context manager for the streamable HTTP client transport.
        """
        return streamable_http_client(
            url=self.url,
            http_client=self.http_client,
            terminate_on_close=self.terminate_on_close,
        )

