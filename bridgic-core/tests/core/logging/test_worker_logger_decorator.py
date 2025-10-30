import asyncio
import json
import logging

import pytest

from bridgic.core.logging import get_worker_logger
from bridgic.core.logging._decorators import worker_logger


class _DummyBase:
    def __init__(self, logger_name: str):
        # Use the same logger acquisition path as Worker
        self._logger = get_worker_logger(logger_name, source=f"test-{logger_name}")
        self._worker_id = f"{logger_name}_unit"

    def _get_execution_context(self):
        # Minimal context for tests
        return {"worker_key": "dummy", "dependencies": ["a", "b"], "parent_automa_name": "Auto", "nesting_level": 1}

    def _format_result_for_log(self, result, max_length: int = 200) -> str:
        s = str(result)
        return s if len(s) <= max_length else s[:max_length] + "... (truncated)"


class DummyAsync(_DummyBase):
    def __init__(self):
        super().__init__("DummyAsync")

    @worker_logger(lambda self: self.__class__.__name__, method_label="arun")
    async def arun(self, x: int):
        await asyncio.sleep(0)
        return x + 1


class DummySync(_DummyBase):
    def __init__(self):
        super().__init__("DummySync")

    @worker_logger(lambda self: self.__class__.__name__, method_label="run")
    def run(self, x: int):
        return x * 2


class DummyError(_DummyBase):
    def __init__(self):
        super().__init__("DummyError")

    @worker_logger(lambda self: self.__class__.__name__, method_label="run")
    def run(self):
        raise ValueError("boom")


class DummyNested(_DummyBase):
    def __init__(self):
        super().__init__("DummyNested")

    @worker_logger(lambda self: self.__class__.__name__, method_label="run")
    def outer(self, x: int):
        return self.inner(x)

    @worker_logger(lambda self: self.__class__.__name__, method_label="run")
    def inner(self, x: int):
        return x + 3


def _attach_capture_logger(component_logger: logging.Logger):
    stream = logging.StringIO() if hasattr(logging, "StringIO") else None
    if stream is None:
        from io import StringIO
        stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    component_logger.addHandler(handler)
    component_logger.setLevel(logging.DEBUG)
    return stream, handler


@pytest.mark.asyncio
async def test_async_decorator_logs_start_end():
    dummy = DummyAsync()
    logger = dummy._logger._logger  # underlying logging.Logger
    stream, handler = _attach_capture_logger(logger)
    try:
        result = await dummy.arun(10)
        assert result == 11
        text = stream.getvalue()
        # Expect WorkerStart and WorkerEnd events serialized as JSON strings
        assert "WorkerStart" in text
        assert "WorkerEnd" in text
        assert "\"worker_name\": \"DummyAsync\"" in text
    finally:
        logger.removeHandler(handler)


def test_sync_decorator_logs_end():
    dummy = DummySync()
    logger = dummy._logger._logger
    stream, handler = _attach_capture_logger(logger)
    try:
        result = dummy.run(7)
        assert result == 14
        text = stream.getvalue()
        assert "WorkerStart" in text
        assert "WorkerEnd" in text
        assert "\"worker_name\": \"DummySync\"" in text
    finally:
        logger.removeHandler(handler)


def test_sync_decorator_logs_error():
    dummy = DummyError()
    logger = dummy._logger._logger
    stream, handler = _attach_capture_logger(logger)
    try:
        with pytest.raises(ValueError):
            dummy.run()
        text = stream.getvalue()
        assert "WorkerStart" in text
        assert "WorkerError" in text
        assert "\"error_type\": \"ValueError\"" in text
    finally:
        logger.removeHandler(handler)


def test_nested_calls_guard_avoids_double_logging():
    dummy = DummyNested()
    logger = dummy._logger._logger
    stream, handler = _attach_capture_logger(logger)
    try:
        result = dummy.outer(5)
        assert result == 8  # inner adds 3
        text = stream.getvalue()
        # Should only have one pair of start/end from the outer call
        start_count = text.count("WorkerStart")
        end_count = text.count("WorkerEnd")
        assert start_count == 1
        assert end_count == 1
    finally:
        logger.removeHandler(handler)


