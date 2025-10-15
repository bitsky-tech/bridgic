from typing import Tuple

import pytest

from bridgic.core.automa import GraphAutoma, From, worker, ArgsMappingRule, System, AutomaRuntimeError
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException
from bridgic.core.automa.worker import Worker
from bridgic.core.types._serialization import Snapshot
from bridgic.core.types._error import WorkerArgsInjectionError

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

    @worker(dependencies=["worker_1"], is_output=True)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

@pytest.fixture
def automa_with_async_worker():
    return AutomaWithAsyncWorker()

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
    
    @worker(dependencies=["worker_1"], is_output=True)
    def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y

@pytest.fixture
def automa_with_sync_worker():
    return AutomaWithSyncWorker()

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
    
    @worker(dependencies=["worker_1"], is_output=True)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x + y
    
@pytest.fixture
def automa_with_sync_and_async_worker():
    return AutomaWithSyncAndAsyncWorker()

@pytest.mark.asyncio
async def test_automa_with_sync_and_async_worker(automa_with_sync_and_async_worker: AutomaWithSyncAndAsyncWorker):
    result = await automa_with_sync_and_async_worker.arun(user_input=1)
    assert result == 5



def worker_0(user_input: int) -> int:
    return user_input + 1

def worker_1(x: int, z: int = 1) -> int:
    return x + z

def worker_2(x: int, y: int = From("worker_0")) -> int:
    return x + y

class AutomaWithFuncAsWorker(GraphAutoma): 
    """
    test from in automa with func as worker
    """
    ...

@pytest.fixture
def automa_with_func_as_worker():
    automa = AutomaWithFuncAsWorker()
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
        is_output=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_func_as_worker(automa_with_func_as_worker: AutomaWithFuncAsWorker):
    result = await automa_with_func_as_worker.arun(user_input=1)
    assert result == 5


def worker_0(user_input: int) -> int:
    return user_input + 1

def worker_1(x: int, z: int = 1) -> int:
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
    automa = AutomaWithClassWorkerArun()
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
        worker=Worker2_Arun(),
        is_output=True,
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_class_worker_arun(automa_with_class_worker_arun: AutomaWithClassWorkerArun):
    result = await automa_with_class_worker_arun.arun(user_input=1)
    assert result == 5


def worker_0(user_input: int) -> int:
    return user_input + 1

def worker_1(x: int, z: int = 1) -> int:
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
    automa = AutomaWithClassWorkerRun()
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
        worker=Worker2_Run(),
        is_output=True,
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa

@pytest.mark.asyncio
async def test_automa_with_class_worker_run(automa_with_class_worker_run: AutomaWithClassWorkerRun):
    result = await automa_with_class_worker_run.arun(user_input=1)
    assert result == 5


########################################################
#### Test case: System("automa") to access the current automa
########################################################


def automa_system_worker_0(user_input: int) -> int:
    return user_input + 1

def automa_system_worker_1(x: int, a: GraphAutoma = System("automa")) -> int:
    a.add_func_as_worker(
        key="automa_system_worker_2",
        func=automa_system_worker_2,
        dependencies=["automa_system_worker_1"],
        is_output=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return x + 1

def automa_system_worker_2(x: int) -> int:
    return x + 1

class AutomaWithSystem(GraphAutoma): ...

@pytest.fixture
def automa_with_system():
    automa_with_system = AutomaWithSystem()
    automa_with_system.add_func_as_worker(
        key="automa_system_worker_0",
        func=automa_system_worker_0,
        is_start=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa_with_system.add_func_as_worker(
        key="automa_system_worker_1",
        func=automa_system_worker_1,
        dependencies=["automa_system_worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa_with_system

@pytest.mark.asyncio
async def test_automa_with_system_automa(automa_with_system: AutomaWithSystem):
    result = await automa_with_system.arun(user_input=1)
    assert result == 4

def worker_0(user_input: int) -> int:
    return user_input + 1

def worker_1(x: int) -> int:
    return x + 1

class NameSpaceWorker:
    @classmethod
    def worker_2(cls, user_input: int) -> int:
        return user_input + 1
    
    def worker_3(self, x: int) -> int:
        return x + 1

class AutomaWithNameSpaceWorker(GraphAutoma): ...

@pytest.fixture
def automa_with_name_space_worker_class_method():
    automa_with_name_space_worker_class_method = AutomaWithNameSpaceWorker()
    automa_with_name_space_worker_class_method.add_func_as_worker(
        key="worker_0",
        func=worker_0,
        is_start=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa_with_name_space_worker_class_method.add_func_as_worker(
        key="worker_1",
        func=worker_1,
        dependencies=["worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa_with_name_space_worker_class_method

@pytest.mark.asyncio
async def test_automa_with_name_space_worker_class_method(automa_with_name_space_worker_class_method: AutomaWithNameSpaceWorker):
    with pytest.raises(
        AutomaRuntimeError, 
        match="the bounded instance of `func` must be the same as the instance of the GraphAutoma"
    ):
        automa_with_name_space_worker_class_method.add_func_as_worker(
            key="worker_2",
            func=NameSpaceWorker.worker_2,
            dependencies=["worker_1"],
            is_output=True,
            args_mapping_rule=ArgsMappingRule.AS_IS,
        )

@pytest.fixture
def automa_with_name_space_worker_instance_method():
    automa_with_name_space_worker_instance_method = AutomaWithNameSpaceWorker()
    automa_with_name_space_worker_instance_method.add_func_as_worker(
        key="worker_0",
        func=worker_0,
        is_start=True,
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa_with_name_space_worker_instance_method.add_func_as_worker(
        key="worker_1",
        func=worker_1,
        dependencies=["worker_0"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return automa_with_name_space_worker_instance_method

@pytest.mark.asyncio
async def test_automa_with_name_space_worker_instance_method(automa_with_name_space_worker_instance_method: AutomaWithNameSpaceWorker):
    name_space_worker = NameSpaceWorker()
    with pytest.raises(
        AutomaRuntimeError, 
        match="the bounded instance of `func` must be the same as the instance of the GraphAutoma"
    ):
        automa_with_name_space_worker_instance_method.add_func_as_worker(
            key="worker_3",
            func=name_space_worker.worker_3,
            dependencies=["worker_1"],
            is_output=True,
            args_mapping_rule=ArgsMappingRule.AS_IS,
        )


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

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x[0] + x[1] + y

@pytest.fixture
def automa_with_args_mapping_asis_and_from():
    return AutomaArgsMappingASISandFrom()

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

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.UNPACK, is_output=True)
    async def worker_2(self, x: int, y: int, z: int = From("worker_0")) -> int:
        return x + y + z

@pytest.fixture
def automa_with_args_mapping_unpack_and_from():
    return AutomaArgsMappingUNPACKandFrom()

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

    @worker(dependencies=["worker_0", "worker_1"], args_mapping_rule=ArgsMappingRule.MERGE, is_output=True)
    async def worker_2(self, x: int, y: int = From("worker_0")) -> int:
        return x[0] + x[1][0] + x[1][1] + y

@pytest.fixture
def automa_with_args_mapping_merge_and_from():
    return AutomaArgsMappingMERGEandFrom()

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

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.SUPPRESSED, is_output=True)
    async def worker_2(self, x: int = From("worker_0")) -> int:
        return x

@pytest.fixture
def automa_with_args_mapping_suppressed_and_from():
    return AutomaArgsMappingSUPPRESSEDandFrom()

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

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
    async def worker_2(self, x: int = From("worker_0")) -> int:
        print(x)
        return x

@pytest.fixture
def automa_with_args_mapping_position_less_than_params_error():
    return AutomaArgsMappingPositionLessThanParamsError()

@pytest.mark.asyncio
async def test_automa_with_args_mapping_position_less_than_params_error(
    automa_with_args_mapping_position_less_than_params_error: AutomaArgsMappingPositionLessThanParamsError,
):
    with pytest.raises(
        WorkerArgsInjectionError, 
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

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.UNPACK, is_output=True)
    async def worker_2(self, x: int, y: int, z: int = From("worker_0")) -> int:
        return x + y + z

@pytest.fixture
def automa_with_args_mapping_unpack_and_from_cover():
    return AutomaArgsMappingUNPACKandFromCover()

@pytest.mark.asyncio
async def test_automa_with_args_mapping_unpack_and_from_cover(automa_with_args_mapping_unpack_and_from_cover: AutomaArgsMappingUNPACKandFromCover):
    result = await automa_with_args_mapping_unpack_and_from_cover.arun(user_input=1)
    assert result == 7


class AutomaFromWithDefaultValue(GraphAutoma):
    """
    test from with default value
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
    async def worker_2(self, x: int, y: int = From("no_exist_worker", 1)) -> int:
        return x + y

@pytest.fixture
def automa_with_from_with_default_value():
    return AutomaFromWithDefaultValue()

@pytest.mark.asyncio
async def test_automa_with_from_with_default_value(automa_with_from_with_default_value: AutomaFromWithDefaultValue):
    result = await automa_with_from_with_default_value.arun(user_input=1)
    assert result == 4


class AutomaFromWithDefaultValueError(GraphAutoma):
    """
    test from with default value
    """
    @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1
    
    @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def worker_1(self, x: int) -> Tuple[int, int]:
        return x + 1

    @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
    async def worker_2(self, x: int, y: int = From("no_exist_worker")) -> int:
        return x + y

@pytest.fixture
def automa_with_from_with_default_value_error():
    return AutomaFromWithDefaultValueError()

@pytest.mark.asyncio
async def test_automa_with_from_with_default_value_error(automa_with_from_with_default_value_error: AutomaFromWithDefaultValueError):
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=f"the worker: `no_exist_worker` is not found in the worker dictionary. "
        "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
    ):
        await automa_with_from_with_default_value_error.arun(user_input=1)


class AutomaWithSystem_1(GraphAutoma):
    """
    test system descriptor
    """
    @worker(is_start=True)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1  # 2

    @worker(dependencies=["worker_0"])
    async def worker_1(self, x: int) -> int:
        return x + 1  # 3

def self_add(x: int):
    return x + 1

def worker_with_system_2(x: int, y: int = From("worker_0", 1), z: GraphAutoma = System("automa:worker_3")) -> int:
    z.add_func_as_worker(
        key="self_add",
        func=self_add,
        is_output=True,
        dependencies=['worker_0'],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    return x + y  # 5

class AutomaWithSystem_2(GraphAutoma):
    """
    inner worker in AutomaWithSystem_1
    """
    @worker(is_start=True)
    async def worker_0(self, x: int) -> int:
        return x + 1  # 6

async def end(x: int):
    return x  # 7


@pytest.fixture
def automa_with_system_1():
    return AutomaWithSystem_1()

@pytest.mark.asyncio
async def test_automa_with_system(automa_with_system_1: AutomaWithSystem_1):
    automa_with_system_1.add_func_as_worker(
        key="worker_2",
        func=worker_with_system_2,
        dependencies=["worker_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa_with_system_1.add_worker(
        key="worker_3",
        worker=AutomaWithSystem_2(),
        dependencies=["worker_2"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    automa_with_system_1.add_func_as_worker(
        key="end",
        func=end,
        is_output=True,
        dependencies=["worker_3"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    res = await automa_with_system_1.arun(user_input=1)
    assert res == 7


########################################################
#### Test case: serilization
########################################################


# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

##### Test cases for a simple Automa && a basic human interaction process #####

class AutomaSerializationWithFrom(GraphAutoma):
    @worker(is_start=True)
    async def worker_1(self, x: int):
        return x + 1

    @worker(dependencies=["worker_1"])
    async def worker_2(self, x: int):
        return x + 1

    @worker(dependencies=["worker_2"])
    async def worker_3(self, x: int, y: int = From("worker_1")):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is x: {x}, y: {y}, do you want to add another 200 to them (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            res = x + 200 + y + 200
        return res

    @worker(dependencies=["worker_3"], is_output=True)
    async def worker_4(self, x: int):
        return x

@pytest.fixture
def automa_serialization_with_from():
    return AutomaSerializationWithFrom()

@pytest.mark.asyncio
async def test_adder_automa_1_interact_serialized(automa_serialization_with_from: AutomaSerializationWithFrom, request, db_base_path):
    try:
        result = await automa_serialization_with_from.arun(x=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is x: 3, y: 2, do you want to add another 200 to them (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "automa_serialization_with_from.bytes"
        version_file = db_base_path / "automa_serialization_with_from.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_automa_serialization_with_from(db_base_path):
        bytes_file = db_base_path / "automa_serialization_with_from.bytes"
        version_file = db_base_path / "automa_serialization_with_from.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = AutomaSerializationWithFrom.load_from_snapshot(snapshot)
        assert type(deserialized_automa) is AutomaSerializationWithFrom
        return deserialized_automa

@pytest.fixture
def interaction_feedback_1_yes(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback

@pytest.mark.asyncio
async def test_adder_automa_1_interact_with_yes_feedback(interaction_feedback_1_yes, deserialized_automa_serialization_with_from):
    # Note: no need to pass the original argument x=100, because the deserialized_adder_automa1 is restored from the snapshot.
    result = await deserialized_automa_serialization_with_from.arun(
        interaction_feedback=interaction_feedback_1_yes
    )
    assert result == 405

