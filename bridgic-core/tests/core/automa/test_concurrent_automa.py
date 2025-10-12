import pytest
from typing import List
import re

from bridgic.core.automa import ConcurrentAutoma, GraphAutoma, worker
from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import WorkerSignatureError, AutomaRuntimeError

###################### Test case 1: single concurrent automa #############################

async def func_1_async(x: int) -> int:
    return x + 1

def func_2(x: int) -> int:
    return x + 2

class Func3AsyncWorker(Worker):
    async def arun(self, x: int) -> int:
        return x + 3

class Func4SyncWorker(Worker):
    def run(self, x: int) -> int:
        return x + 4

@pytest.fixture
def concurrent_autom_1() -> ConcurrentAutoma:
    concurrent = ConcurrentAutoma()
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
    @concurrent.worker(key="func_5")
    async def func_5_async(x: int) -> int:
        return x + 5
    return concurrent

@pytest.mark.asyncio
async def test_concurrent_autom_1(concurrent_autom_1: ConcurrentAutoma):
    result: List[int] = await concurrent_autom_1.arun(x=100)
    assert result == [101, 102, 103, 104, 105]

    # Test all_workers()
    assert concurrent_autom_1.all_workers() == ["func_1", "func_2", "func_3", "func_4", "func_5"]

###################### Test case 2: netsted dynamic concurrent automa #########################

class MyGraph2(GraphAutoma):
    _concurrent: ConcurrentAutoma

    def __init__(self):
        super().__init__()

        #TODO: how to serialize the self._concurrent field properly?
        self._concurrent = ConcurrentAutoma()
        self.add_worker(
            key="concurrent",
            worker=self._concurrent,
            dependencies=["start"],
            is_output=True,
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
    return MyGraph2()

@pytest.mark.asyncio
async def test_concurrent_autom_2(graph_2: GraphAutoma):
    #First run
    result = await graph_2.arun(x=100, count=3)
    assert result == [101, 101, 101]

    #Second run
    result = await graph_2.arun(x=100, count=5)
    assert result == [100 + 1] * 5

###################### Test case 3: worker decorator in concurrent automa #########################

class MyConcurrentAutoma3(ConcurrentAutoma):
    @worker(key="func_1")
    async def func_1(self, x: int) -> int:
        return x + 1

    @worker(key="func_2")
    async def func_2(self, x: int) -> int:
        return x + 2

@pytest.fixture
def concurrent_autom_3() -> ConcurrentAutoma:
    return MyConcurrentAutoma3()

@pytest.mark.asyncio
async def test_concurrent_autom_3(concurrent_autom_3: ConcurrentAutoma):
    result: List = await concurrent_autom_3.arun(x=100)
    assert result == [101, 102]
    # Test case for only one worker!
    concurrent_autom_3.remove_worker("func_1")
    result = await concurrent_autom_3.arun(x=100)
    assert result == [102]

###################### Test cases for Errors / Exceptions in concurrent automa #########################

def test_worker_signature_errors():
    with pytest.raises(WorkerSignatureError, match="Unexpected arguments:"):
        class MyConcurrentAutoma_WithWrongDependencies(ConcurrentAutoma):
            @worker(key="func_1")
            async def func_1(self, x: int) -> int:
                return x + 1

            @worker(key="func_2", dependencies=["func_1"])
            async def func_2(self, x: int) -> int:
                return x + 2

def test_topology_change_errors():
    concurrent = MyConcurrentAutoma3()

    with pytest.raises(AutomaRuntimeError, match="the reserved key `__merger__` is not allowed to be used"):
        concurrent.add_func_as_worker(
            key="__merger__",
            func=func_1_async,
        )
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        concurrent.add_func_as_worker(
            key="func_1",
            func=func_1_async,
        )
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__merger__` is not allowed to be used"):
        concurrent.add_worker(
            key="__merger__",
            worker=Func3AsyncWorker(),
        )
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        concurrent.add_worker(
            key="func_1",
            worker=Func3AsyncWorker(),
        )
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__merger__` is not allowed to be used"):
        @concurrent.worker(key="__merger__")
        async def func_5_async(automa, x: int) -> int:
            return x + 5
    with pytest.raises(AutomaRuntimeError, match="the merge worker is not allowed to be removed from the concurrent automa"):
        concurrent.remove_worker("__merger__")
    with pytest.raises(AutomaRuntimeError, match=re.escape("add_dependency() is not allowed to be called on a concurrent automa")):
        concurrent.add_dependency("__merger__", "func_1")

#############

class MyConcurrentAutoma_TryFerryTo(ConcurrentAutoma):
    @worker(key="func_1")
    async def func_1(self, x: int) -> int:
        return x + 1

    @worker(key="func_2")
    async def func_2(self, x: int) -> int:
        self.ferry_to("func_1") # This should raise an error
        return x + 2

@pytest.mark.asyncio
async def test_ferry_error():
    concurrent = MyConcurrentAutoma_TryFerryTo()
    with pytest.raises(AutomaRuntimeError, match=re.escape("ferry_to() is not allowed to be called on a concurrent automa")):
        result = await concurrent.arun(x=100)
