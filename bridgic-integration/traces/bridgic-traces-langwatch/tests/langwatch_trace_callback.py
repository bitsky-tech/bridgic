import asyncio
import pytest
import os
import warnings
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.langwatch import LangWatchTraceCallback, start_langwatch_trace


def _is_langwatch_configured(api_key: str = None, endpoint_url: str = None) -> bool:
    """Check if LangWatch is configured and ready to use."""
    try:
        # Use warnings.catch_warnings() context manager to temporarily suppress warnings
        # during configuration check. This prevents warning messages from polluting test output
        # when LangWatch is not properly configured (e.g., missing API key).
        # 
        # LangWatchTraceCallback._setup_langwatch() may emit warnings when configuration fails,
        # but we only care about whether the callback is ready (_is_ready flag), not the warning messages.
        with warnings.catch_warnings():
            # Ignore all warnings within this context to keep test output clean
            warnings.simplefilter("ignore")
            callback = LangWatchTraceCallback(
                api_key=api_key,
                endpoint_url=endpoint_url,
            )
            return callback._is_ready
    except Exception:
        return False


# Check environment variables, set LANGWATCH_API_KEY=your-api-key-here
# Optionally set LANGWATCH_ENDPOINT (defaults to https://app.langwatch.ai, for self-hosted, set LANGWATCH_ENDPOINT=http://localhost:5560)
_langwatch_api_key = os.environ.get("LANGWATCH_API_KEY")
_langwatch_endpoint = os.environ.get("LANGWATCH_ENDPOINT")

# Check if LangWatch is configured
_is_langwatch_ready = _is_langwatch_configured(api_key=_langwatch_api_key, endpoint_url=_langwatch_endpoint)

class MyAutoma(GraphAutoma):
    @worker(is_start=True)
    async def step1(self):
        await asyncio.sleep(1)
        return "hello"
    
    @worker(dependencies=["step1"], is_output=True)
    async def step2(self, step1: str):
        await asyncio.sleep(1)
        return f"{step1} world"


@pytest.mark.skipif(
    not _is_langwatch_ready,
    reason="LangWatch is not configured. Set LANGWATCH_API_KEY to run tests.",
)
@pytest.mark.asyncio
async def test_langwatch_trace_with_start_langwatch_trace():
    """Test using start_langwatch_trace helper function from README."""
    # Use the higher-level function (this uses GlobalSetting.add() which appends)
    start_langwatch_trace(
        api_key=_langwatch_api_key,
        endpoint_url=_langwatch_endpoint,
    )

    automa = MyAutoma()
    result = await automa.arun()
    assert result == "hello world"


