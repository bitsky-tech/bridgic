from typing import Optional, Union, Dict, Any, TYPE_CHECKING
from typing_extensions import override
from mcp.types import Tool as McpTool

from bridgic.core.agentic.tool_specs._base_tool_spec import ToolSpec
from bridgic.core.model.types import Tool
from bridgic.core.automa.worker import Worker
from bridgic.protocols.mcp._mcp_server_connection import McpServerConnection
from bridgic.protocols.mcp._mcp_server_connection_manager import McpServerConnectionManager
from bridgic.protocols.mcp._error import McpServerConnectionError

if TYPE_CHECKING:
    from bridgic.protocols.mcp._mcp_tool_worker import McpToolWorker


class McpToolSpec(ToolSpec):
    """
    A tool specification that represents an MCP tool from a connected MCP server.

    This class provides a bridge between MCP tools and the Bridgic framework,
    allowing MCP tools to be used seamlessly within Bridgic agentic systems.
    """

    tool_info: McpTool
    """The raw MCP tool definition from the server."""

    _server_connection: McpServerConnection
    """The connection to the MCP server that provides this tool."""

    _server_connection_name: Optional[str]
    """The name of the server connection, used for lookup after deserialization."""

    def __init__(
        self,
        tool_name: str,
        tool_info: McpTool,
        server_connection: Union[str, McpServerConnection],
    ):
        super().__init__(
            tool_name=tool_name,
            tool_description=tool_info.description or "",
            tool_parameters=tool_info.inputSchema or {},
        )

        # Store the corresponding McpTool object.
        self.tool_info = tool_info

        # Try to associate with the real connection object.
        if isinstance(server_connection, str):
            self._server_connection = McpServerConnectionManager.get_connection(server_connection)
            self._server_connection_name = server_connection
        elif isinstance(server_connection, McpServerConnection):
            self._server_connection = server_connection
            self._server_connection_name = server_connection.name
        else:
            raise TypeError(f"Invalid type for server connection: {type(server_connection)}")

    @classmethod
    def from_raw(
        cls,
        tool_name: str,
        server_connection: Union[str, McpServerConnection],
    ) -> "McpToolSpec":
        """
        Create a McpToolSpec from a specified server connection and tool name.
        """
        # Try to associate with the real connection object.
        connection = None
        if isinstance(server_connection, str):
            connection = McpServerConnectionManager.get_connection(server_connection)
        elif isinstance(server_connection, McpServerConnection):
            connection = server_connection
        else:
            raise TypeError(f"Invalid type for server connection: {type(server_connection)}")

        # Use the connection to get the McpToolSpec object corresponding to tool_name
        all_tools = connection.list_tools()
        tool_spec = next((tool for tool in all_tools if tool.tool_name == tool_name), None)
        if tool_spec is None:
            raise ValueError(f"Cannot find tool '{tool_name}' in the provided server connection.")
        return tool_spec

    @property
    def server_connection(self) -> McpServerConnection:
        """
        Get the server connection, loading it from the server connection manager if necessary.

        This property implements lazy loading of the server connection. If the connection 
        is not available (e.g., after deserialization), it will be retrieved from the 
        server connection manager by its name.

        Returns
        -------
        McpServerConnection
            The server connection instance.

        Raises
        ------
        McpServerConnectionError
            If the connection cannot be found in the manager.
        """
        if self._server_connection is None:
            if self._server_connection_name is None:
                raise McpServerConnectionError(
                    f"Cannot load server connection for McpToolSpec '{self._tool_name}': "
                    f"connection name is not available."
                )

            try:
                connection = McpServerConnectionManager.get_connection(self._server_connection_name)
            except KeyError as e:
                raise McpServerConnectionError(
                    f"Failed to load the server connection for McpToolSpec \"{self._tool_name}\", because the "
                    f"connection named \"{self._server_connection_name}\" was not found in any connection manager. "
                    f"You must create a McpServerConnection with name \"{self._server_connection_name}\" and "
                    f"ensure it is properly registered in a connection manager before using this tool spec."
                ) from e

            self._server_connection = connection

        return self._server_connection

    @override
    def to_tool(self) -> Tool:
        """
        Transform this McpToolSpec to a `Tool` object used by LLM.

        Returns
        -------
        Tool
            A `Tool` object that can be used by LLM for tool selection.
        """
        return Tool(
            name=self._tool_name,
            description=self._tool_description,
            parameters=self._tool_parameters
        )

    @override
    def create_worker(self) -> Worker:
        """
        Create a Worker from the information included in this McpToolSpec.

        Returns
        -------
        Worker
            A new `McpToolWorker` object that can be added to an Automa to execute the tool.
        """
        from bridgic.protocols.mcp._mcp_tool_worker import McpToolWorker
        return McpToolWorker(
            tool_name=self._tool_name,
            server_connection=self.server_connection,
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["server_connection_name"] = self._server_connection_name
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        connection_name = state_dict.get("server_connection_name", None)

        if connection_name is None:
            raise McpServerConnectionError(
                f"Cannot load server connection for McpToolSpec \"{self._tool_name}\", because "
                f"its connection name is not available."
            )

        # Try to reload the server connection from the global manager.
        self._server_connection = McpServerConnectionManager.get_connection(connection_name)
        self._server_connection_name = connection_name

        # Load tool_info from the real connection.
        all_tools = self.server_connection.list_tools()
        reload_tool_spec = next((tool for tool in all_tools if tool.tool_name == self._tool_name), None)
        if reload_tool_spec is None:
            raise RuntimeError(
                f"Cannot restore McpToolSpec because the tool is not found on server: "
                f"server_connection_name=\"{connection_name}\", tool_name=\"{self._tool_name}\"."
            )
        self.tool_info = reload_tool_spec.tool_info

