import pytest
from bridgic.automa import Automa
from bridgic.automa.worker import CallableWorker

# This test scirpt demonstrates the "code-first orchestration" pattern, on how to wrap a function or method into a Worker.
# Input: x
# Output: 3x+5

def multiply_3(a: int) -> int:
    return a * 3

class MyAdder:
    def add_5(self, a: int) -> int:
        return a + 5

class SimpleFlow(Automa):
    def __init__(self):
        super().__init__()
        self.multiply_worker = CallableWorker(multiply_3)
        my_adder = MyAdder()
        self.add_worker = CallableWorker(my_adder.add_5)

    async def process_async(self, x):
        result = await self.multiply_worker.process_async(x)
        result = await self.add_worker.process_async(result)
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
