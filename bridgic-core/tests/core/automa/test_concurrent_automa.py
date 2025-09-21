import pytest
from typing import List

from bridgic.core.automa import ConcurrentAutoma, GraphAutoma, worker
from bridgic.core.automa.worker import Worker


###################### Test case 1: single concurrent worker #############################

async def func_1_async(automa, x: int) -> int:
    return x + 1

def func_2(automa, x: int) -> int:
    return x + 2

class Func3AsyncWorker(Worker):
    async def arun(self, x: int) -> int:
        return x + 3

class Func4SyncWorker(Worker):
    def run(self, x: int) -> int:
        return x + 4

@pytest.fixture
def concurrent_autom_1() -> ConcurrentAutoma[int]:
    concurrent = ConcurrentAutoma[int]()
    concurrent.add_func_as_worker(
        key="func_1",
        func=func_1_async,
    )
    concurrent.add_func_as_worker(
        key="func_2",
        func=func_2,
    )
    concurrent.add_worker(
        key="func_3",
        worker=Func3AsyncWorker(),
    )
    concurrent.add_worker(
        key="func_4",
        worker=Func4SyncWorker(),
    )
    return concurrent

@pytest.mark.asyncio
async def test_concurrent_autom_1(concurrent_autom_1: ConcurrentAutoma[int]):
    result: List[int] = await concurrent_autom_1.arun(x=100)
    assert result == [101, 102, 103, 104]

###################### Test case 2: netsted dynamic concurrent worker #########################

class MyGraph2(GraphAutoma):
    _concurrent: ConcurrentAutoma[int]

    def __init__(self):
        super().__init__()

        #TODO: how to serialize the self._concurrent field properly?
        self._concurrent = ConcurrentAutoma[int]()
        self.add_worker(
            key="concurrent",
            worker=self._concurrent,
            dependencies=["start"],
        )

    @worker(key="start", is_start=True)
    async def start_dynamic(self, x: int, count: int) -> int:
        for key in self._concurrent.all_workers():
            self._concurrent.remove_worker(key)
        for i in range(1, count+1):
            self._concurrent.add_func_as_worker(
                key=f"func_{i}",
                func=func_1_async,
            )
        return x

@pytest.fixture
def graph_2() -> GraphAutoma:
    graph = MyGraph2()
    graph.output_worker_key = "concurrent"
    return graph

@pytest.mark.asyncio
async def test_concurrent_autom_2(graph_2: GraphAutoma):
    #First run
    result = await graph_2.arun(x=100, count=3)
    assert result == [101, 101, 101]

    #Second run
    result = await graph_2.arun(x=100, count=5)
    assert result == [100 + 1] * 5