"""Utility functions for agent tracer."""

from __future__ import annotations

from typing import Any


def serialize_value(value: Any) -> Any:
    """Serialize a value to a JSON-compatible format.

    Args:
        value: The value to serialize

    Returns:
        Serialized value
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if hasattr(value, "model_dump"):  # Pydantic models
        return value.model_dump()
    if hasattr(value, "dict"):  # Older Pydantic models
        return value.dict()
    return str(value)


def mask_sensitive_data(data: dict[str, Any], sensitive_keywords: list[str]) -> dict[str, Any]:
    """Mask sensitive data in a dictionary.

    Args:
        data: Dictionary containing potentially sensitive data
        sensitive_keywords: List of keywords to mask

    Returns:
        Dictionary with sensitive values masked
    """

    def _mask(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "*****" if any(word in k.lower() for word in sensitive_keywords) else _mask(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [_mask(i) for i in obj]
        return obj

    return _mask(data)

