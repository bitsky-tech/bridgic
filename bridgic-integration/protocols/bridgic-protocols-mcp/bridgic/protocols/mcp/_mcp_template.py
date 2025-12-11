from typing import List, Union
from mcp.types import PromptMessage, Prompt

from bridgic.core.prompt._base_template import BasePromptTemplate
from bridgic.protocols.mcp._mcp_server_connection import McpServerConnection
from bridgic.protocols.mcp._mcp_server_connection_manager import McpServerConnectionManager
from bridgic.core.model.types import Message

class McpPromptTemplate(BasePromptTemplate):
    """
    This template implementation is used to generate a prompt from a connected MCP server.
    """

    prompt_name: str
    """The name of the prompt template."""

    prompt_info: Prompt
    """The raw information of the prompt."""

    _server_connection: McpServerConnection
    """The connection to the MCP server."""

    def __init__(
        self,
        prompt_name: str,
        prompt_info: Prompt,
        server_connection: Union[str, McpServerConnection],
    ):
        super().__init__(prompt_name=prompt_name, prompt_info=prompt_info)
        
        # Try to associate with the real connection object.
        if isinstance(server_connection, str):
            self._server_connection = McpServerConnectionManager.get_connection(server_connection)
        elif isinstance(server_connection, McpServerConnection):
            self._server_connection = server_connection
        else:
            raise TypeError(f"Invalid type for server connection: {type(server_connection)}")

    def format_messages(self, **kwargs) -> List[Message]:
        """
        Format the prompt template from a connected MCP server into messages.

        Parameters
        ----------
        **kwargs : Any
            The keyword arguments to pass to the prompt template.

        Returns
        -------
        List[Message]
            The list of messages.
        """
        if not self._server_connection or not self._server_connection.is_connected:
            raise RuntimeError("MCP session is not connected, unable to render prompt.")

        mcp_result = self._server_connection.get_prompt(
            prompt_name=self.prompt_name,
            arguments=kwargs,
        )

        mcp_messages: List[PromptMessage] = mcp_result.messages

        messages: List[Message] = []
        if mcp_messages:
            for msg in mcp_messages:
                messages.append(Message.from_text(text=msg.content.text, role=msg.role))

        return messages

