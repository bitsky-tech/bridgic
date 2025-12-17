import pytest
import pytest_asyncio
from mcp.types import Prompt

from bridgic.protocols.mcp import McpPromptTemplate
from bridgic.core.model.types import Message, Role


@pytest.mark.asyncio
async def test_mcp_prompt_template_stdio(mock_writer_stdio_connection):
    template = McpPromptTemplate(
        prompt_name="ask_for_creative",
        prompt_info=Prompt(name="ask_for_creative", description="Ask for a creative idea"),
        server_connection=mock_writer_stdio_connection,
    )

    messages = template.format_messages(
        topic="Product Launch",
        description="A new innovative product that combines AI and design",
    )

    assert isinstance(messages, list)
    assert len(messages) > 0

    for msg in messages:
        assert isinstance(msg, Message)
        assert msg.role in [Role.SYSTEM, Role.USER, Role.AI]
        assert len(msg.blocks) > 0

@pytest.mark.asyncio
async def test_mcp_prompt_template_streamable_http(mock_writer_streamable_http_connection):
    template = McpPromptTemplate(
        prompt_name="ask_for_creative",
        prompt_info=Prompt(name="ask_for_creative", description="Ask for a creative idea"),
        server_connection=mock_writer_streamable_http_connection,
    )

    messages = template.format_messages(
        topic="Product Launch",
        description="A new innovative product that combines AI and design",
    )

    assert isinstance(messages, list)
    assert len(messages) > 0

    for msg in messages:
        assert isinstance(msg, Message)
        assert msg.role in [Role.SYSTEM, Role.USER, Role.AI]
        assert len(msg.blocks) > 0

