"""
Shared pytest fixtures for integration tests.

This module provides shared fixtures for integration tests that require
real LLM services, MCP servers, and other external dependencies.
"""
import os
import platform
import pytest
import pytest_asyncio
import shutil

from bridgic.core.config import HttpClientConfig, HttpClientAuthConfig, HttpClientTimeoutConfig
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration
from bridgic.llms.vllm import VllmServerLlm, VllmServerConfiguration
from bridgic.protocols.mcp import (
    McpServerConnectionStdio,
    McpServerConnectionStreamableHttp,
    McpToolSetBuilder,
)


# ============================================================================
# Environment variable fixtures
# ============================================================================

@pytest.fixture
def openai_api_key():
    """OpenAI API key from environment variable."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def openai_model_name():
    """OpenAI model name from environment variable."""
    return os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")


@pytest.fixture
def openai_api_base():
    """OpenAI API base URL from environment variable."""
    return os.environ.get("OPENAI_BASE_URL")


@pytest.fixture
def vllm_api_base():
    """VLLM server API base URL from environment variable."""
    return os.environ.get("VLLM_SERVER_API_BASE")


@pytest.fixture
def vllm_api_key():
    """VLLM server API key from environment variable."""
    return os.environ.get("VLLM_SERVER_API_KEY", "EMPTY")


@pytest.fixture
def vllm_model_name():
    """VLLM server model name from environment variable."""
    return os.environ.get("VLLM_SERVER_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")


@pytest.fixture
def github_token():
    """GitHub token from environment variable."""
    return os.environ.get("GITHUB_TOKEN")


@pytest.fixture
def github_mcp_url():
    """GitHub MCP HTTP URL from environment variable."""
    return os.environ.get("GITHUB_MCP_HTTP_URL")


@pytest.fixture
def has_npx():
    """Check if npx is available."""
    return shutil.which("npx") is not None

@pytest.fixture
def has_chrome():
    """Check if Chrome is available (cross-platform)."""
    system = platform.system()

    # Check common Chrome executable names via PATH
    if system == "Linux":
        chrome_names = ["google-chrome", "chrome", "chromium", "chromium-browser"]
        for name in chrome_names:
            if shutil.which(name):
                return True

    # macOS: Check for Chrome in Applications folder
    if system == "Darwin":
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        for path in chrome_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return True

    # Windows: Check common installation paths
    elif system == "Windows":
        chrome_paths = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles%\Chromium\Application\chromium.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Chromium\Application\chromium.exe"),
        ]
        for path in chrome_paths:
            if os.path.exists(path):
                return True

    return False


# ============================================================================
# LLM fixtures
# ============================================================================

@pytest.fixture
def openai_llm(openai_api_key, openai_model_name, openai_api_base):
    """OpenAI LLM fixture."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    return OpenAILlm(
        api_key=openai_api_key,
        configuration=OpenAIConfiguration(model=openai_model_name),
        timeout=30,
        api_base=openai_api_base,
    )


@pytest.fixture
def vllm_llm(vllm_api_base, vllm_api_key, vllm_model_name):
    """VLLM server LLM fixture."""
    if not vllm_api_base:
        pytest.skip("VLLM_SERVER_API_BASE environment variable not set")
    
    return VllmServerLlm(
        api_key=vllm_api_key,
        api_base=vllm_api_base,
        configuration=VllmServerConfiguration(model=vllm_model_name),
        timeout=30,
    )


# ============================================================================
# MCP Server Connection fixtures
# ============================================================================

@pytest_asyncio.fixture
async def github_mcp_stdio_connection(has_npx, github_token):
    """GitHub MCP server connection via stdio (requires npx)."""
    if not has_npx:
        pytest.skip("npx is not available")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable not set")

    connection = McpServerConnectionStdio(
        name="github-mcp-stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": github_token},
        request_timeout=5,
    )

    connection.connect()
    yield connection
    connection.close()


@pytest_asyncio.fixture
async def github_mcp_streamable_http_connection(github_mcp_url, github_token):
    """GitHub MCP server connection via streamable HTTP."""
    if not github_mcp_url:
        pytest.skip("GITHUB_MCP_HTTP_URL environment variable not set")
    if not github_token:
        pytest.skip("GITHUB_TOKEN environment variable not set")
    
    import httpx
    from mcp.shared._httpx_utils import create_mcp_http_client
    
    http_client = create_mcp_http_client(
        headers={"Authorization": f"Bearer {github_token}"}
    )
    
    connection = McpServerConnectionStreamableHttp(
        name="github-mcp-streamable-http",
        url=github_mcp_url,
        http_client=http_client,
        request_timeout=15,
    )
    
    connection.connect()
    yield connection
    connection.close()


@pytest_asyncio.fixture
async def playwright_mcp_stdio_connection(has_npx, has_chrome):
    """Playwright MCP server connection via stdio (requires npx)."""
    if not has_npx:
        pytest.skip("npx is not available")
    if not has_chrome:
        pytest.skip("Chrome is not available")

    connection = McpServerConnectionStdio(
        name="playwright-mcp-stdio",
        command="npx",
        args=[
            "@playwright/mcp@latest",
        ],
        request_timeout=60,  # Browser operations may take longer
    )

    connection.connect()
    yield connection
    connection.close()


# ============================================================================
# MCP Tool Set Builder fixtures
# ============================================================================

@pytest_asyncio.fixture
def github_mcp_streamable_http_connection_toolset_builder(github_mcp_url, github_token):
    return McpToolSetBuilder.streamable_http(
        url=github_mcp_url,
        http_client_config=HttpClientConfig(
            # timeout=HttpClientTimeoutConfig(read=15),
            auth=HttpClientAuthConfig(
                type="bearer",
                token=github_token,
            ),
        ),
        request_timeout=15,
    )


@pytest_asyncio.fixture
async def playwright_mcp_stdio_connection_toolset_builder():
    return McpToolSetBuilder.stdio(
        command="npx",
        args=["@playwright/mcp@latest", "--isolated"],
        request_timeout=60,
    )


# ============================================================================
# Shared test utilities
# ============================================================================

@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    """Base path for storing test data (snapshots, etc.)."""
    return tmp_path_factory.mktemp("integration_test_data")
