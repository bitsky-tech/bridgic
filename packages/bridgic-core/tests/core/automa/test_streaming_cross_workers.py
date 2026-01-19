import pytest

from typing import AsyncGenerator
from bridgic.core.automa.worker import Worker
from bridgic.core.automa import GraphAutoma, worker

@pytest.mark.asyncio
async def test_streaming_automa_with_class_worker():
    class StreamProducerWorker(Worker):
        async def arun(self, cnt: int) -> AsyncGenerator[int, None]:
            for i in range(cnt):
                yield i

    class StreamConsumerWorker(Worker):
        async def arun(self, stream: AsyncGenerator[int, None]) -> AsyncGenerator[int, None]:
            async for i in stream:
                yield i * 10

    automa_obj = GraphAutoma()
    automa_obj.add_worker("stream_producer", StreamProducerWorker(), is_start=True)
    automa_obj.add_worker("stream_consumer", StreamConsumerWorker(), dependencies=["stream_producer"], is_output=True)

    stream = await automa_obj.arun(cnt=10)
    print("")
    async for ele in stream:
        print(ele)

@pytest.mark.asyncio
async def test_streaming_automa_with_decorated_worker():
    class StreamingAutoma(GraphAutoma):
        @worker(is_start=True)
        async def stream_producer(self, cnt: int) -> AsyncGenerator[int, None]:
            for i in range(cnt):
                yield i

        @worker(dependencies=["stream_producer"], is_output=True)
        async def stream_consumer(self, stream: AsyncGenerator[int, None]) -> AsyncGenerator[int, None]:
            async for i in stream:
                yield i * 10

    automa_obj = StreamingAutoma()
    stream = await automa_obj.arun(cnt=10)
    print("")
    async for ele in stream:
        print(ele)
