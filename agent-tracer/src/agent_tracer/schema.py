"""Data schemas for agent tracer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_serializer


class Log(BaseModel):
    """A log entry for a trace."""

    name: str
    message: Any
    type: str

    @field_serializer("message")
    def serialize_message(self, value: Any) -> Any:
        """Serialize message to a safe format."""
        try:
            # Try to serialize as JSON-compatible type
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            if isinstance(value, (dict, list)):
                return value
            # For complex objects, convert to string
            return str(value)
        except Exception:
            return str(value)


class TracingConfig(BaseModel):
    """Configuration for tracing service."""

    deactivate_tracing: bool = False
    mask_sensitive_data: bool = True
    sensitive_keywords: list[str] = ["api_key", "password", "token", "secret", "server_url"]
    enable_console: bool = True  # Enable console output by default

