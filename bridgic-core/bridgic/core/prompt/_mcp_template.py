from typing import List, Union
from mcp.types import PromptMessage

from bridgic.core.prompt._base_template import BasePromptTemplate
from bridgic.core.mcp._mcp_server_connection import McpServerConnection
from bridgic.core.model.types import Message

class McpPromptTemplate(BasePromptTemplate):
    """
    This template implementation is used to generate a prompt from a connected MCP server.
    """

    prompt_name: str
    """The name of the prompt template."""

    _server_connection: McpServerConnection
    """The connection to the MCP server."""

    def __init__(self, prompt_name: str, server_connection: McpServerConnection):
        super().__init__(prompt_name=prompt_name)
        self._server_connection = server_connection

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