"""
Test cases for rerunning an Automa instance.
"""

from bridgic.core.automa import worker, GraphAutoma
from bridgic.core.automa.graph_automa import RuntimeContext
import pytest

#### Test case: rerun an Automa instance.

class ArithmeticAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return 3 * x

    @worker(dependencies=["start"])
    async def end(self, x: int):
        return x + 5

@pytest.fixture
def arithmetic():
    graph = ArithmeticAutoma(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_single_automa_rerun(arithmetic: ArithmeticAutoma):
    # First run.
    result = await arithmetic.arun(x=2)
    assert result == 11
    # Second run.
    result = await arithmetic.arun(x=5)
    assert result == 20
    # Third run.
    result = await arithmetic.arun(x=10)
    assert result == 35

#### Test case: rerun a nested Automa instance by ferry-to. The states (counter) of the nested Automa should be maintained after rerun.

class TopAutoma(GraphAutoma):
    # The start worker is a nested Automa which will be added by add_worker()

    @worker(dependencies=["start"])
    async def end(self, my_list: list[str]):
        if len(my_list) < 5:
            self.ferry_to("start")
        else:
            return my_list

class NestedAutoma(GraphAutoma):
    def should_reset_local_space(self) -> bool:
        return False
    
    @worker(is_start=True)
    async def counter(self):
        local_space = self.get_local_space(runtime_context=RuntimeContext(worker_key="counter"))
        local_space["count"] = local_space.get("count", 0) + 1
        return local_space["count"]

    @worker(dependencies=["counter"])
    async def end(self, count: int):
        return ['bridgic'] * count

@pytest.fixture
def nested_automa():
    graph = NestedAutoma(output_worker_key="end")
    return graph

@pytest.fixture
def topAutoma(nested_automa):
    graph = TopAutoma(output_worker_key="end")
    graph.add_worker("start", nested_automa, is_start=True)
    return graph

@pytest.mark.asyncio
async def test_nested_automa_rerun(topAutoma):
    # First run.
    result = await topAutoma.arun()
    assert result == ['bridgic'] * 5
