import asyncio
import pytest
import os
import warnings
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.opik import OpikTraceCallback, start_opik_trace


# Check environment variables, if is self-hosted, set OPIK_USE_LOCAL=true, else set OPIK_API_KEY=your-api-key-here
_opik_api_key = os.environ.get("OPIK_API_KEY", None)
_opik_use_local = os.environ.get("OPIK_USE_LOCAL", "false").lower() == "true"

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
async def test_opik_trace_with_start_opik_trace():
    """Test using start_opik_trace helper function from README."""
    if _opik_api_key is None and _opik_use_local is False:
        pytest.skip("Set OPIK_API_KEY=\"your-api-key\" or OPIK_USE_LOCAL=true to run this test.")

    # Use the higher-level function (this uses GlobalSetting.add() which appends)
    start_opik_trace(
        project_name="test-project",
        use_local=_opik_use_local,
        api_key=_opik_api_key,
    )

    automa = MyAutoma()
    result = await automa.arun()
    assert result == "hello world"


