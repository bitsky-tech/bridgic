import uuid

from typing import Optional, List, Dict, Any, Literal
from typing_extensions import override
from mcp.shared._httpx_utils import MCP_DEFAULT_TIMEOUT, MCP_DEFAULT_SSE_READ_TIMEOUT

from bridgic.core.agentic.tool_specs._base_tool_spec import ToolSetBuilder, ToolSetResponse
from bridgic.core.config import HttpClientConfig, create_http_client_from_config
from bridgic.protocols.mcp._mcp_server_connection import (
    McpServerConnection,
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
    McpServerConnectionType,
)
from bridgic.protocols.mcp._mcp_server_connection_manager import McpServerConnectionManager


class McpToolSetBuilder(ToolSetBuilder):
    """
    A builder for creating exclusive `McpToolSpec` instances from a newly created MCP server connection.

    This builder creates a new MCP server connection from scratch, ensuring that the connection is 
    exclusive and not shared with other instances. This is important for stateful connections that 
    should not be shared to different owners.

    Example
    -------
    >>> from bridgic.protocols.mcp import McpToolSetBuilder
    >>> from bridgic.core.agentic.recent import ReCentAutoma
    >>> 
    >>> # Create a builder for stdio connection
    >>> builder = McpToolSetBuilder.stdio(
    ...     command="npx",
    ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    ...     tool_names=["read_file", "write_file"]  # Optional: filter specific tools
    ... )
    >>> 
    >>> # Or create a builder for streamable HTTP connection
    >>> builder = McpToolSetBuilder.streamable_http(
    ...     url="http://localhost:8000",
    ...     tool_names=["get_weather"]  # Optional: filter specific tools
    ... )
    """

    _connection_type: Literal[McpServerConnectionType.STDIO, McpServerConnectionType.STREAMABLE_HTTP]
    """The type of connection: 'stdio' or 'streamable_http'."""

    _connection_config: Dict[str, Any]
    """Configuration parameters for creating the connection."""
    
    _tool_names: Optional[List[str]]
    """Optional list of tool names to include. If None, all tools will be included."""
    
    def __init__(
        self,
        connection_type: str,
        connection_config: Dict[str, Any],
        tool_names: Optional[List[str]] = None,
    ):
        """
        Initialize the McpToolSetBuilder.

        Parameters
        ----------
        connection_type : str
            The type of connection: 'stdio' or 'streamable_http'.
        connection_config : Dict[str, Any]
            Configuration parameters for creating the connection.
            - For stdio connections, this should include 'command', and optionally 'args', 'env', 'encoding', 'request_timeout'.
            - For streamable_http connections, this should include 'url', and optionally 'http_client_config', 'terminate_on_close', 'request_timeout'.
        tool_names : Optional[List[str]]
            Optional list of tool names to include.
            If None, all available tools from the connection will be included.
        """
        if connection_type not in (McpServerConnectionType.STDIO, McpServerConnectionType.STREAMABLE_HTTP):
            raise ValueError(
                f"Invalid connection_type: {connection_type}. "
                f"Expected 'stdio' or 'streamable_http'."
            )

        self._connection_type = connection_type
        self._connection_config = connection_config
        self._tool_names = tool_names
    
    @classmethod
    def stdio(
        cls,
        command: str,
        *,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        encoding: Optional[str] = None,
        request_timeout: Optional[int] = None,
        tool_names: Optional[List[str]] = None,
    ) -> "McpToolSetBuilder":
        """
        Create a builder for a stdio-based MCP server connection.

        Parameters
        ----------
        command : str
            The command to use for the connection (e.g., "npx", "python").
        args : Optional[List[str]]
            The arguments to pass to the command.
        env : Optional[Dict[str, str]]
            Environment variables to set for the process.
        encoding : Optional[str]
            The encoding to use for the connection. Defaults to "utf-8".
        request_timeout : Optional[int]
            The timeout in seconds for MCP requests. Default is 30 seconds.
        tool_names : Optional[List[str]]
            Optional list of tool names to include. If None, all available tools will be included.

        Returns
        -------
        McpToolSetBuilder
            A builder instance configured for stdio connection.
        """
        connection_config = {
            "command": command,
        }
        if args is not None:
            connection_config["args"] = args
        if env is not None:
            connection_config["env"] = env
        if encoding is not None:
            connection_config["encoding"] = encoding
        if request_timeout is not None:
            connection_config["request_timeout"] = request_timeout
        
        return cls(
            connection_type="stdio",
            connection_config=connection_config,
            tool_names=tool_names,
        )

    @classmethod
    def streamable_http(
        cls,
        url: str,
        *,
        http_client_config: Optional[HttpClientConfig] = None,
        terminate_on_close: Optional[bool] = None,
        request_timeout: Optional[int] = None,
        tool_names: Optional[List[str]] = None,
    ) -> "McpToolSetBuilder":
        """
        Create a builder for a streamable HTTP-based MCP server connection.

        Parameters
        ----------
        url : str
            The URL of the MCP server.
        http_client_config : Optional[HttpClientConfig]
            Optional configuration for creating the HTTP client.
            If None, a default client will be created with MCP defaults.
        terminate_on_close : Optional[bool]
            If True, send a DELETE request to terminate the session when the connection
            is closed. Defaults to True.
        request_timeout : Optional[int]
            The timeout in seconds for MCP requests. Default is 30 seconds.
        tool_names : Optional[List[str]]
            Optional list of tool names to include. If None, all available tools will be included.

        Returns
        -------
        McpToolSetBuilder
            A builder instance configured for streamable HTTP connection.
        """
        connection_config = {
            "url": url,
        }
        if http_client_config is not None:
            connection_config["http_client_config"] = http_client_config
        if terminate_on_close is not None:
            connection_config["terminate_on_close"] = terminate_on_close
        if request_timeout is not None:
            connection_config["request_timeout"] = request_timeout

        return cls(
            connection_type="streamable_http",
            connection_config=connection_config,
            tool_names=tool_names,
        )

    @override
    def build(self) -> ToolSetResponse:
        """
        Build and return McpToolSpec instances from a newly created server connection.
        
        This method creates a new connection on each call, ensuring that each call gets its own 
        exclusive connection instance. The builder acts as a factory for creating connections.
        
        Returns
        -------
        ToolSetResponse
            A response containing the list of McpToolSpec instances with `_from_builder=True`,
            along with optional extras.
        
        Raises
        ------
        McpServerConnectionError
            If the connection cannot be created or accessed.
        RuntimeError
            If the connection fails to establish.
        """
        # Create a new connection for this build call.
        connection = self._create_connection()


        # Get all available tools from the connection.
        all_tool_specs = connection.list_tools()
        
        # Filter by tool_names if specified.
        if self._tool_names is not None:
            tool_specs = [
                tool_spec for tool_spec in all_tool_specs
                if tool_spec.tool_name in self._tool_names
            ]
            
            # Check if all requested tools were found.
            found_tool_names = {tool_spec.tool_name for tool_spec in tool_specs}
            missing_tool_names = set(self._tool_names) - found_tool_names
            if missing_tool_names:
                raise ValueError(
                    f"The following tools were not found in the MCP server connection "
                    f"'{connection.name}': {sorted(missing_tool_names)}"
                )
        else:
            tool_specs = all_tool_specs

        # Mark all tool specs as from builder.
        for tool_spec in tool_specs:
            tool_spec._from_builder = True

        return ToolSetResponse(
            tool_specs=tool_specs,
            extras={
                "mcp_server_connection_name": connection.name,
            }
        )

    def _create_connection(self) -> McpServerConnection:
        """
        Create and connect a new MCP server connection.

        Each call to this method creates a new connection with a unique name, allowing the 
        builder to act as a factory for creating multiple connection instances.

        Returns
        -------
        McpServerConnection
            The created and connected connection instance.
        """
        # Generate a unique connection name for each connection instance
        connection_name = f"mcp-builder-{uuid.uuid4().hex[:8]}"

        # Create connection based on type
        if self._connection_type == McpServerConnectionType.STDIO:
            connection = McpServerConnectionStdio(
                name=connection_name,
                **self._connection_config,
            )
        elif self._connection_type == McpServerConnectionType.STREAMABLE_HTTP:
            # Copy connection config to avoid modifying the original config.
            connection_config = self._connection_config.copy()

            # Pop http_client_config because the real initialization of the connection doesn't need it.
            http_client_config: HttpClientConfig = connection_config.pop("http_client_config", None)

            if not http_client_config:
                http_client = None
            else:
                if "timeout" not in http_client_config or http_client_config["timeout"] is None:
                    http_client_config["timeout"] = {
                        "default": MCP_DEFAULT_TIMEOUT,
                        "read": MCP_DEFAULT_SSE_READ_TIMEOUT,
                    }
                http_client = create_http_client_from_config(http_client_config.copy(), is_async=True)

            connection = McpServerConnectionStreamableHttp(
                name=connection_name,
                http_client=http_client,
                **connection_config,
            )
        else:
            raise ValueError(f"Unknown connection type: {self._connection_type}")

        # Register and connect
        manager = McpServerConnectionManager.get_instance()
        manager.register_connection(connection)
        connection.connect()

        return connection

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {
            "connection_type": self._connection_type,
            "connection_config": self._connection_config,
        }
        if self._tool_names is not None:
            state_dict["tool_names"] = self._tool_names
        return state_dict
    
    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self._connection_type = state_dict["connection_type"]
        self._connection_config = state_dict["connection_config"]
        self._tool_names = state_dict.get("tool_names")
