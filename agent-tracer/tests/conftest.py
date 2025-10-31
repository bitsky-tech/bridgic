"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Clear all tracer-related environment variables for testing."""
    env_vars = [
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_HOST",
        "LANGWATCH_API_KEY",
        "ARIZE_API_KEY",
        "ARIZE_SPACE_ID",
        "PHOENIX_API_KEY",
        "OPIK_API_KEY",
        "OPIK_WORKSPACE",
        "TRACELOOP_API_KEY",
    ]
    
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

