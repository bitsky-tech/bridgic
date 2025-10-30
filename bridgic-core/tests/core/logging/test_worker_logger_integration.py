import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from bridgic.core.automa._automa import Automa
from bridgic.core.automa.worker import Worker


class DummySyncWorker(Worker):
    def run(self, x: int, y: int) -> int:
        # Simple CPU-bound work to ensure thread pool path is taken
        return x + y


class DummyAutoma(Automa):
    # Minimal implementations for abstract methods; not used in this test
    def _locate_interacting_worker(self):
        return None

    def _get_worker_key(self, worker: Worker):
        for k, v in getattr(self, "_workers", {}).items():
            if v is worker:
                return k
        return None

    def _get_worker_instance(self, worker_key: str) -> Worker:
        return self._workers[worker_key]


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
async def test_worker_arun_thread_pool_logs_start_end_once():
    # Set up thread pool and automa
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        automa = DummyAutoma(name="TestAuto", thread_pool=executor)
        worker = DummySyncWorker()
        # Wire worker into automa context so _get_execution_context can resolve worker_key
        automa._workers = {"sum": worker}
        worker.parent = automa

        # Capture worker logger output
        logger = worker._logger._logger
        stream, handler = _attach_capture_logger(logger)
        try:
            # Call arun -> should offload run to thread pool and log start/end
            result = await worker.arun(3, 4)
            assert result == 7
            text = stream.getvalue()
            # Ensure single start/end pair
            assert text.count("WorkerStart") == 1
            assert text.count("WorkerEnd") == 1
            # Error should not appear
            assert "WorkerError" not in text
        finally:
            logger.removeHandler(handler)
    finally:
        executor.shutdown(wait=True)


