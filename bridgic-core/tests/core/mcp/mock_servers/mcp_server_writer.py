import click
from typing import List
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

mcp = FastMCP("writer-mcp")


@mcp.tool()
def write_advertising(topic: str) -> str:
    return f"This is advertising about {topic}."

@mcp.prompt()
def ask_for_creative(topic: str, description: str) -> List[PromptMessage]:
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": (
                    f"You are a creativity expert specializing in the topic: {topic}.\n\n"
                    f"Based on the following description, please provide creative suggestions: {description}"
                ),
            },
        },
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": (
                    f"For the topic '{topic}', I suggest:\n"
                    f"1. Leverage unique features of the topic\n"
                    f"2. Highlight the key aspects of the description\n"
                    f"3. Propose an innovative integrated solution"
                ),
            },
        }
    ]


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable_http"], case_sensitive=False),
    default="stdio",
    help="Transport type to use (default: stdio)",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host address for SSE or Streamable HTTP transport (default: 127.0.0.1)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port number for SSE or Streamable HTTP transport (default: 8000)",
)
def main(transport: str, host: str, port: int):
    transport = transport.lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable_http":
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        click.echo(f"Unsupported transport: {transport}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()

