"""
Test cases for rerunning an Automa instance.
"""

from bridgic.automa import worker, GraphAutoma
import pytest
from typing import Dict, Any

#### Test case: rerun an Automa instance.

class ArithmeticAutoma(GraphAutoma):
    @worker(is_start=True)
    def start(self, x: int):
        return 3 * x

    @worker(dependencies=["start"])
    def end(self, x: int):
        return x + 5

@pytest.fixture
def arithmetic():
    graph = ArithmeticAutoma(output_worker_key="end")
    return graph

@pytest.mark.asyncio
async def test_single_automa_rerun(arithmetic: ArithmeticAutoma):
    # First run.
    result = await arithmetic.process_async(x=2)
    assert result == 11
    # Second run.
    result = await arithmetic.process_async(x=5)
    assert result == 20
    # Third run.
    result = await arithmetic.process_async(x=10)
    assert result == 35

#### Test case: rerun a nested Automa instance by ferry-to. The states (counter) of the nested Automa should be maintained after rerun.

class TopAutoma(GraphAutoma):
    # The start worker is a nested Automa which will be added by add_worker()

    @worker(dependencies=["start"])
    def end(self, my_list: list[str]):
        if len(my_list) < 5:
            self.ferry_to("start")
        else:
            return my_list

class NestedAutoma(GraphAutoma):
    @worker(is_start=True)
    def counter(self):
        local_space: Dict[str, Any] = self.counter.local_space
        local_space["count"] = local_space.get("count", 0) + 1
        return local_space["count"]

    @worker(dependencies=["counter"])
    def end(self, count: int):
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
    result = await topAutoma.process_async()
    assert result == ['bridgic'] * 5
