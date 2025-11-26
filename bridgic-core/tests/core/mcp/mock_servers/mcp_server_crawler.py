import click
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("crawler-mcp")


@mcp.tool()
def get_weather(date: str, city: str) -> Dict[str, Any]:
    city_temps = {
        "New York": 15,
        "London": 10,
        "Shanghai": 19,
        "Shenzhen": 26,
    }
    base_temp = city_temps.get(city, 20)
    
    temp_variation = hash(date) % 3
    temperature = base_temp + temp_variation
    
    conditions = ["sunny", "cloudy", "rainy", "windy"]
    condition = conditions[hash(city + date) % len(conditions)]
    
    return {
        "date": date,
        "city": city,
        "temperature": f"{temperature}°C",
        "condition": condition,
    }

@mcp.tool()
def get_news(date: str, country: str) -> List[str]:
    news_templates = [
        f"{date} {country} [Technology]: major progress in innovation",
        f"{date} {country} [Social]: continuous optimization of people's livehood policies",
        f"{date} {country} [Culture]: traditional culture is revitalized",
    ]
    return news_templates


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
