import pytest
import os
from bridgic.core.mcp._mcp_server_connection import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
)

# GitHub MCP server 配置
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_MCP_HTTP_URL = os.environ.get("GITHUB_MCP_HTTP_URL", "http://localhost:3000/sse")


@pytest.mark.asyncio
async def test_github_mcp_server_stdio_connection():
    """
    测试通过 stdio 方式连接 GitHub MCP server。
    
    需要设置环境变量 GITHUB_TOKEN 才能运行此测试。
    需要安装 @modelcontextprotocol/server-github 包。
    """
    if not GITHUB_TOKEN:
        pytest.skip("GITHUB_TOKEN 环境变量未设置，跳过 stdio 连接测试")
    
    # 创建 stdio 连接
    connection = McpServerConnectionStdio(
        name="github-mcp-stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": GITHUB_TOKEN},
        request_timeout=30,
    )
    
    try:
        # 建立连接
        await connection.connect()
        
        # 验证连接成功
        assert connection.session is not None
        assert connection.session._request_id > 0  # 初始化后 request_id 应该大于 0
        
        # 可以尝试列出可用的工具来验证连接
        tools = await connection.session.list_tools()
        assert tools is not None
        
    finally:
        # 清理资源
        await connection._exit_stack.aclose()


@pytest.mark.asyncio
async def test_github_mcp_server_streamable_http_connection():
    """
    测试通过 streamable http 方式连接 GitHub MCP server。
    
    需要设置环境变量 GITHUB_MCP_HTTP_URL 指向运行中的 GitHub MCP HTTP 服务器。
    如果未设置，默认使用 http://localhost:3000/sse
    
    注意：如果服务器未运行，此测试会失败。可以通过设置环境变量或跳过测试来避免。
    """
    # 创建 streamable http 连接
    connection = McpServerConnectionStreamableHttp(
        name="github-mcp-streamable-http",
        url=GITHUB_MCP_HTTP_URL,
        request_timeout=30,
    )
    
    try:
        # 建立连接
        await connection.connect()
        
        # 验证连接成功
        assert connection.session is not None
        assert connection.session._request_id > 0  # 初始化后 request_id 应该大于 0
        
        # 可以尝试列出可用的工具来验证连接
        tools = await connection.session.list_tools()
        assert tools is not None
        
    finally:
        # 清理资源
        await connection._exit_stack.aclose()
