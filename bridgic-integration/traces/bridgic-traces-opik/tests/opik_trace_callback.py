import asyncio
import pytest
import os
import warnings
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.opik import OpikTraceCallback, start_opik_trace


def _is_opik_configured(use_local: bool = False, api_key: str = None, host: str = None) -> bool:
    """Check if Opik is configured and ready to use."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            callback = OpikTraceCallback(
                project_name="test-project",
                use_local=use_local,
                api_key=api_key,
            )
            return callback._is_ready
    except Exception:
        return False


# Check environment variables, if is self-hosted, set OPIK_USE_LOCAL=true, else set OPIK_API_KEY=your-api-key-here
_opik_api_key = os.environ.get("OPIK_API_KEY")
_opik_use_local = os.environ.get("OPIK_USE_LOCAL", "false").lower() == "true"

# Check if Opik is configured
_is_opik_ready = _is_opik_configured(use_local=_opik_use_local, api_key=_opik_api_key)

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
    not _is_opik_ready,
    reason="Opik is not configured. Set OPIK_API_KEY or OPIK_USE_LOCAL=true to run tests.",
)
@pytest.mark.asyncio
async def test_opik_trace_with_start_opik_trace():
    """Test using start_opik_trace helper function from README."""
    # Use the higher-level function (this uses GlobalSetting.add() which appends)
    start_opik_trace(
        project_name="test-project",
        use_local=_opik_use_local,
        api_key=_opik_api_key,
    )

    automa = MyAutoma()
    result = await automa.arun()
    assert result == "hello world"


