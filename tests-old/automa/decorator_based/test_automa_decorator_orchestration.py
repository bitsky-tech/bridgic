import pytest
from bridgic.automa import AutoMa
from bridgic.automa.bridge.decorator import worker

# This test script demonstrates how to implement a simple flow using the "decorator-based" orchestration pattern.
# Input: x
# Output: 3x+5

class SimpleFlow(AutoMa):

    def __init__(self):
        super().__init__()

    @worker(is_start=True)
    def multiply_3(self, x: int) -> int:
        return x * 3

    @worker(is_end=True, listen=multiply_3)
    def add_5(self, x: int) -> int:
        return x + 5

@pytest.fixture
def simple_flow():
    yield SimpleFlow()
    # teardown code may be here

@pytest.mark.asyncio
async def test_simple_flow(simple_flow):
    x = 7
    result = await simple_flow.process_async(x=x)
    assert result == 3 * x + 5