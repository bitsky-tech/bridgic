import pytest
from bridgic.automa import Automa
from bridgic.automa.worker import Worker

# This test script demonstrates how to implement a simple flow using the "code-first" orchestration pattern.
# Input: x
# Output: 3x+5

class MultiplyWorker(Worker):
    async def process_async(self, x):
        result = x * 3
        return result

class AddWorker(Worker):
    async def process_async(self, x):
        result = x + 5
        return result

class SimpleFlow(Automa):
    def __init__(self):
        super().__init__()
        self.multiply_worker = MultiplyWorker()
        self.addition_worker = AddWorker()

    async def process_async(self, x):
        result = await self.multiply_worker.process_async(x)
        result = await self.addition_worker.process_async(result)
        return result

@pytest.fixture
def simple_flow():
    yield SimpleFlow()
    # teardown code may be here

@pytest.mark.asyncio
async def test_simple_flow(simple_flow):
    x = 7
    result = await simple_flow.process_async(x=x)
    assert result == 3 * x + 5
