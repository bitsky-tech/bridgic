from typing import Tuple

import pytest

from bridgic.core.automa import GraphAutoma, From, worker, ArgsMappingRule
from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import WorkerArgsMappingError

########################################################
#### Test case: All kinds of workers with From
########################################################

class AutomaWithAsyncWorker(GraphAutoma):
    """
    test from in automa with async worker
    """
    @worker(is_start=True)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1

    @worker(dependencies=["worker_0"])
    async def worker_1(self, x: int, z: int = 1) -> int:
        return x + 1

    @worker(dependencies=["worker_1"])
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

@pytest.fixture
def automa_with_async_worker():
    return AutomaWithAsyncWorker(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_async_worker(automa_with_async_worker: AutomaWithAsyncWorker):
    result = await automa_with_async_worker.arun(user_input=1)
    assert result == 5


class AutomaWithSyncWorker(GraphAutoma):
    """
    test from in automa with sync worker
    """
    @worker(is_start=True)
    def worker_0(self, user_input: int) -> int:
        return user_input + 1

    @worker(dependencies=["worker_0"])
    def worker_1(self, x: int, z: int = 1) -> int:
        return x + 1
    
    @worker(dependencies=["worker_1"])
    def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

@pytest.fixture
def automa_with_sync_worker():
    return AutomaWithSyncWorker(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_sync_worker(automa_with_sync_worker: AutomaWithSyncWorker):
    result = await automa_with_sync_worker.arun(user_input=1)
    assert result == 5


class AutomaWithSyncAndAsyncWorker(GraphAutoma):
    """
    test from in automa with sync and async worker
    """
    @worker(is_start=True)
    def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"])
    async def worker_1(self, x: int, z: int = 1) -> int:
        return x + z
    
    @worker(dependencies=["worker_1"])
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y
    
@pytest.fixture
def automa_with_sync_and_async_worker():
    return AutomaWithSyncAndAsyncWorker(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_sync_and_async_worker(automa_with_sync_and_async_worker: AutomaWithSyncAndAsyncWorker):
    result = await automa_with_sync_and_async_worker.arun(user_input=1)
    assert result == 5



def worker_0(automa: GraphAutoma, user_input: int) -> int:
    return user_input + 1

def worker_1(automa: GraphAutoma, x: int, z: int = 1) -> int:
    return x + z

def worker_2(automa: GraphAutoma, x: int, y: int = From("worker_0")) -> int:
    return x + y

class AutomaWithFuncAsWorker(GraphAutoma): 
    """
    test from in automa with func as worker
    """
    ...

@pytest.fixture
def automa_with_func_as_worker():
    automa = AutomaWithFuncAsWorker(output_worker_key="worker_2")
    automa.add_func_as_worker(
        key="worker_0", 
        func=worker_0,
        args_mapping_rule=ArgsMappingRule.AS_IS,
        is_start=True,
    )
    automa.add_func_as_worker(
        key="worker_1",
        func=worker_1,
        dependencies=["worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa.add_func_as_worker(
        key="worker_2",
        func=worker_2,
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_func_as_worker(automa_with_func_as_worker: AutomaWithFuncAsWorker):
    result = await automa_with_func_as_worker.arun(user_input=1)
    assert result == 5


def worker_0(automa: GraphAutoma, user_input: int) -> int:
    return user_input + 1

def worker_1(automa: GraphAutoma, x: int, z: int = 1) -> int:
    return x + z

class Worker2_Arun(Worker):
    async def arun(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

class AutomaWithClassWorkerArun(GraphAutoma): 
    """
    test from in automa with class worker
    """
    ...

@pytest.fixture
def automa_with_class_worker_arun():
    automa = AutomaWithClassWorkerArun(output_worker_key="Worker2_Arun")
    automa.add_func_as_worker(
        key="worker_0",
        func=worker_0,
        is_start=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa.add_func_as_worker(
        key="worker_1",
        func=worker_1,
        dependencies=["worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa.add_worker(
        key="Worker2_Arun",
        worker_obj=Worker2_Arun(),
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_class_worker_arun(automa_with_class_worker_arun: AutomaWithClassWorkerArun):
    result = await automa_with_class_worker_arun.arun(user_input=1)
    assert result == 5


def worker_0(automa: GraphAutoma, user_input: int) -> int:
    return user_input + 1

def worker_1(automa: GraphAutoma, x: int, z: int = 1) -> int:
    return x + z

class Worker2_Run(Worker):
    def run(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

class AutomaWithClassWorkerRun(GraphAutoma): 
    """
    test from in automa with class worker
    """
    ...

@pytest.fixture
def automa_with_class_worker_run():
    automa = AutomaWithClassWorkerRun(output_worker_key="Worker2_Run")
    automa.add_func_as_worker(
        key="worker_0",
        func=worker_0,
        is_start=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa.add_func_as_worker(
        key="worker_1",
        func=worker_1,
        dependencies=["worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa.add_worker(
        key="Worker2_Run",
        worker_obj=Worker2_Run(),
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_class_worker_run(automa_with_class_worker_run: AutomaWithClassWorkerRun):
    result = await automa_with_class_worker_run.arun(user_input=1)
    assert result == 5


########################################################
#### Test case: All kinds of ArgsMappingRule
########################################################


class AutomaArgsMappingASISandFrom(GraphAutoma):
    """
    test args mapping rule ASIS and From
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1, x

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x[0] + x[1] + y

@pytest.fixture
def automa_with_args_mapping_asis_and_from():
    return AutomaArgsMappingASISandFrom(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_asis_and_from(automa_with_args_mapping_asis_and_from: AutomaArgsMappingASISandFrom):
    result = await automa_with_args_mapping_asis_and_from.arun(user_input=1)
    assert result == 7


class AutomaArgsMappingUNPACKandFrom(GraphAutoma):
    """
    test args mapping rule UNPACK and From
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1, x

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def worker_2(self, x: int, y: int, z: int = From("worker_0")) -> int:
        return x + y + z

@pytest.fixture
def automa_with_args_mapping_unpack_and_from():
    return AutomaArgsMappingUNPACKandFrom(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_unpack_and_from(automa_with_args_mapping_unpack_and_from: AutomaArgsMappingUNPACKandFrom):
    result = await automa_with_args_mapping_unpack_and_from.arun(user_input=1)
    assert result == 7


class AutomaArgsMappingMERGEandFrom(GraphAutoma):
    """
    test args mapping rule MERGE and From
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1, x

    @worker(dependencies=["worker_0", "worker_1"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x[0] + x[1][0] + x[1][1] + y

@pytest.fixture
def automa_with_args_mapping_merge_and_from():
    return AutomaArgsMappingMERGEandFrom(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_merge_and_from(automa_with_args_mapping_merge_and_from: AutomaArgsMappingMERGEandFrom):
    result = await automa_with_args_mapping_merge_and_from.arun(user_input=1)
    assert result == 9


class AutomaArgsMappingSUPPRESSEDandFrom(GraphAutoma):
    """
    test args mapping rule SUPPRESSED and From
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1, x

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def worker_2(self, x: int = From("worker_0")) -> int:
        return x

@pytest.fixture
def automa_with_args_mapping_suppressed_and_from():
    return AutomaArgsMappingSUPPRESSEDandFrom(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_suppressed_and_from(automa_with_args_mapping_suppressed_and_from: AutomaArgsMappingMERGEandFrom):
    result = await automa_with_args_mapping_suppressed_and_from.arun(user_input=1)
    assert result == 2


class AutomaArgsMappingPositionLessThanParamsError(GraphAutoma):
    """
    test from cover param
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1, x

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_2(self, x: int = From("worker_0")) -> int:
        print(x)
        return x

@pytest.fixture
def automa_with_args_mapping_position_less_than_params_error():
    return AutomaArgsMappingPositionLessThanParamsError(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_position_less_than_params_error(
    automa_with_args_mapping_position_less_than_params_error: AutomaArgsMappingPositionLessThanParamsError,
):
    with pytest.raises(
        WorkerArgsMappingError, 
        match="The number of parameters is less than or equal to the number of positional arguments, but got 1 parameters and 1 positional arguments"
    ):
        await automa_with_args_mapping_position_less_than_params_error.arun(user_input=1)


class AutomaArgsMappingUNPACKandFromCover(GraphAutoma):
    """
    test args mapping rule UNPACK and From
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return {
            "x": x + 1,
            "y": x,
            "z": x + 2,
        }

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def worker_2(self, x: int, y: int, z: int = From("worker_0")) -> int:
        return x + y + z

@pytest.fixture
def automa_with_args_mapping_unpack_and_from_cover():
    return AutomaArgsMappingUNPACKandFromCover(output_worker_key="worker_2")

@pytest.mark.asyncio
async def test_automa_with_args_mapping_unpack_and_from_cover(automa_with_args_mapping_unpack_and_from_cover: AutomaArgsMappingUNPACKandFromCover):
    result = await automa_with_args_mapping_unpack_and_from_cover.arun(user_input=1)
    assert result == 7  
