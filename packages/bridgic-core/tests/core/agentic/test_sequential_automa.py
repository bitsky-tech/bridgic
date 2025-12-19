import pytest
import re

from bridgic.core.automa import worker
from bridgic.core.agentic import SequentialAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.types._error import WorkerSignatureError, AutomaRuntimeError


################################################################################
# Test case 1: sequential automa with 0/1/multiple workers
################################################################################

async def func_1_async(x: int) -> int:
    return x + 1


class Func2SyncWorker(Worker):
    def run(self, x: int) -> int:
        return x + 2


@pytest.mark.asyncio
async def test_flow_1():
    """Test SequentialAutoma with 0, 1, and multiple workers."""
    flow = SequentialAutoma()
    
    # Test case for 0 workers.
    result = await flow.arun(100)
    assert result == None

    # Test case for 1 worker.
    flow.add_func_as_worker("func_1", func_1_async)
    result = await flow.arun(100)
    assert result == 100 + 1

    # Test case for 2 workers.
    flow.add_worker("func_2", Func2SyncWorker())
    result = await flow.arun(100)
    assert result == 100 + 1 + 2

    # Test case for 3 workers (using decorator).
    @flow.worker(key="func_3")
    def func_3(x: int) -> int:
        return x + 3
    result = await flow.arun(100)
    assert result == 100 + 1 + 2 + 3


################################################################################
# Test case 2: worker decorator in sequential automa
################################################################################

class MySequentialFlow(SequentialAutoma):
    @worker(key="func_1")
    async def func_1(self, x: int) -> int:
        return x + 1

    @worker(key="func_2")
    async def func_2(self, x: int) -> int:
        return x + 2


@pytest.fixture
def flow_2() -> MySequentialFlow:
    return MySequentialFlow()


@pytest.mark.asyncio
async def test_flow_2(flow_2: MySequentialFlow):
    """Test SequentialAutoma with worker decorators on class methods."""
    result = await flow_2.arun(x=100)
    assert result == 100 + 1 + 2


################################################################################
# Test cases for Errors / Exceptions in SequentialAutoma
################################################################################

def test_worker_signature_errors():
    """Test that dependencies are not allowed in SequentialAutoma workers."""
    with pytest.raises(WorkerSignatureError, match="Unexpected arguments:"):
        class MySequentialAutoma_WithWrongDependencies(SequentialAutoma):
            @worker(key="func_1")
            async def func_1(self, x: int) -> int:
                return x + 1

            @worker(key="func_2", dependencies=["func_1"])
            async def func_2(self, x: int) -> int:
                return x + 2


def test_topology_change_errors():
    """Test various topology change errors in SequentialAutoma."""
    flow = MySequentialFlow()

    # Test reserved key errors
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        flow.add_func_as_worker(key="__tail__", func=func_1_async)
    
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        flow.add_worker(key="__tail__", worker=Func2SyncWorker())
    
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        @flow.worker(key="__tail__")
        async def func_5_async(automa, x: int) -> int:
            return x + 5

    # Test duplicate key errors
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        flow.add_func_as_worker(key="func_1", func=func_1_async)
    
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        flow.add_worker(key="func_1", worker=Func2SyncWorker())

    # Test disallowed operations
    with pytest.raises(AutomaRuntimeError, match=re.escape("remove_worker() is not allowed to be called on a sequential automa")):
        flow.remove_worker("func_2")
    
    with pytest.raises(AutomaRuntimeError, match=re.escape("add_dependency() is not allowed to be called on a sequential automa")):
        flow.add_dependency("__tail__", "func_1")


class MySequentialAutoma_TryFerryTo(SequentialAutoma):
    @worker(key="func_1")
    async def func_1(self, x: int) -> int:
        return x + 1

    @worker(key="func_2")
    async def func_2(self, x: int) -> int:
        self.ferry_to("func_1")  # This should raise an error
        return x + 2


@pytest.mark.asyncio
async def test_ferry_error():
    """Test that ferry_to is not allowed in SequentialAutoma."""
    flow = MySequentialAutoma_TryFerryTo()
    with pytest.raises(AutomaRuntimeError, match=re.escape("ferry_to() is not allowed to be called on a sequential automa")):
        await flow.arun(x=100)
