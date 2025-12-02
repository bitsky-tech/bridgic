import asyncio
import pytest
import os
import warnings
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.langwatch import LangWatchTraceCallback, start_langwatch_trace


# Check environment variables, set LANGWATCH_API_KEY=your-api-key-here
# Optionally set LANGWATCH_ENDPOINT (defaults to https://app.langwatch.ai, for self-hosted, set LANGWATCH_ENDPOINT=http://localhost:5560)
_langwatch_api_key = os.environ.get("LANGWATCH_API_KEY", None)
_langwatch_endpoint = os.environ.get("LANGWATCH_ENDPOINT", None)

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        await asyncio.sleep(1)
        return "hello"
    
    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
        await asyncio.sleep(1)
        return f"{step1} world"


@pytest.mark.asyncio
async def test_langwatch_trace_with_start_langwatch_trace():
    """Test using start_langwatch_trace helper function from README."""
    if _langwatch_api_key is None:
        pytest.skip("Set LANGWATCH_API_KEY=\"your-api-key\" to run this test.")

    # Use the higher-level function (this uses GlobalSetting.add() which appends)
    start_langwatch_trace(
        api_key=_langwatch_api_key,
        endpoint_url=_langwatch_endpoint,
    )

    automa = MyAutoma()
    result = await automa.arun()
    assert result == "hello world"


