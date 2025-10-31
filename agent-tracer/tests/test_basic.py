"""Basic tests for agent-tracer."""

import pytest

from agent_tracer import TracingService, TracingConfig, Log


def test_imports():
    """Test that basic imports work."""
    assert TracingService is not None
    assert TracingConfig is not None
    assert Log is not None


def test_tracing_config():
    """Test TracingConfig initialization."""
    config = TracingConfig()
    assert config.deactivate_tracing is False
    assert config.mask_sensitive_data is True
    assert len(config.sensitive_keywords) > 0


def test_tracing_config_custom():
    """Test TracingConfig with custom values."""
    config = TracingConfig(
        deactivate_tracing=True,
        mask_sensitive_data=False,
        sensitive_keywords=["custom_key"]
    )
    assert config.deactivate_tracing is True
    assert config.mask_sensitive_data is False
    assert config.sensitive_keywords == ["custom_key"]


def test_log_schema():
    """Test Log schema."""
    log = Log(name="test", message="Test message", type="info")
    assert log.name == "test"
    assert log.message == "Test message"
    assert log.type == "info"


def test_log_complex_message():
    """Test Log with complex message."""
    log = Log(name="test", message={"key": "value", "number": 42}, type="debug")
    assert log.name == "test"
    assert isinstance(log.message, dict)


def test_tracing_service_init():
    """Test TracingService initialization."""
    tracer = TracingService()
    assert tracer is not None
    assert tracer.deactivated is False


def test_tracing_service_with_config():
    """Test TracingService with config."""
    config = TracingConfig(deactivate_tracing=True)
    tracer = TracingService(config=config)
    assert tracer.deactivated is True


@pytest.mark.asyncio
async def test_tracing_service_deactivated():
    """Test that deactivated tracer doesn't error."""
    config = TracingConfig(deactivate_tracing=True)
    tracer = TracingService(config=config)
    
    # These should not raise errors even when deactivated
    await tracer.start_trace(trace_name="Test")
    
    async with tracer.trace_step("step", {"input": "test"}):
        tracer.set_outputs("step", {"output": "test"})
    
    await tracer.end_trace(outputs={"result": "test"})


@pytest.mark.asyncio
async def test_basic_tracing_flow():
    """Test basic tracing flow without actual backends."""
    config = TracingConfig(deactivate_tracing=False)
    tracer = TracingService(config=config)
    
    # Start trace (no backends configured, should not fail)
    await tracer.start_trace(
        trace_name="Test Trace",
        project_name="Test Project"
    )
    
    # Trace a step
    async with tracer.trace_step(
        step_name="test_step",
        inputs={"test": "input"}
    ):
        tracer.add_log("test_step", Log(
            name="test_log",
            message="Test message",
            type="info"
        ))
        
        tracer.set_outputs("test_step", {"result": "output"})
    
    # End trace
    await tracer.end_trace(outputs={"final": "result"})


@pytest.mark.asyncio
async def test_sensitive_data_masking():
    """Test that sensitive data is masked."""
    from agent_tracer.utils import mask_sensitive_data
    
    data = {
        "api_key": "secret123",
        "password": "pass123",
        "normal_field": "visible",
        "nested": {
            "api_key": "secret456",
            "safe": "visible"
        }
    }
    
    masked = mask_sensitive_data(data, ["api_key", "password"])
    
    assert masked["api_key"] == "*****"
    assert masked["password"] == "*****"
    assert masked["normal_field"] == "visible"
    assert masked["nested"]["api_key"] == "*****"
    assert masked["nested"]["safe"] == "visible"

