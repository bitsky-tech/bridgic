import pytest
import asyncio
from typing import Iterator, AsyncIterator
from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
from bridgic.core.utils.console import printer

class AsyncStreamGraph(GraphAutoma):
    @worker(is_start=True)
    async def stream_worker(self, text: str) -> AsyncIterator[str]:
        for char in text:
            await asyncio.sleep(0.05)
            yield char

    @worker(dependencies=["stream_worker"])
    async def consume_stream_worker(self, text_stream: AsyncIterator[str]) -> str:
        text = ""
        async for char in text_stream:
            await asyncio.sleep(0.05)
            printer.print(char, end="")
            text += char
        return text

    @worker(dependencies=["consume_stream_worker"])
    def result_worker(self, text: str) -> str:
        return text

@pytest.mark.asyncio
async def test_stream_in_mixed_concurrency_workers():
    graph = AsyncStreamGraph(name="async_stream_graph", output_worker_key="result_worker")
    result = await graph.arun(text="Hello, world!")
    assert result == "Hello, world!"