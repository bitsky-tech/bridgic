import pytest
import re

from bridgic.core.automa import worker
from bridgic.core.agentic import SequentialAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.types._error import WorkerSignatureError, AutomaRuntimeError
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot

########## Test case 1: sequential automa with 0/1/multiple workers ############

async def func_1_async(x: int) -> int:
    return x + 1

class Func2SyncWorker(Worker):
    def run(self, x: int) -> int:
        return x + 2

@pytest.mark.asyncio
async def test_flow_1():
    flow = SequentialAutoma()
    # Test case for 0 workers.
    result = await flow.arun(100)
    assert result == None

    # Test case for 1 worker.
    flow.add_func_as_worker(
        "func_1",
        func_1_async,
    )
    result = await flow.arun(100)
    assert result == 100 + 1

    # Test case for 2 worker.
    flow.add_worker(
        "func_2",
        Func2SyncWorker(),
    )
    result = await flow.arun(100)
    assert result == 100 + 1 + 2

    @flow.worker(key="func_3")
    def func_3(x: int) -> int:
        return x + 3
    result = await flow.arun(100)
    assert result == 100 + 1 + 2 + 3

########## Test case 2: worker decorator in sequential automa ############

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
    result = await flow_2.arun(x=100)
    assert result == 100 + 1 + 2

########## Test case 3: human interaction in sequential automa ############

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

class AdderAutoma(SequentialAutoma):
    @worker()
    async def func_1(self, x: int):
        return x + 1

    @worker()
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 200
        return x + 2

@pytest.fixture
def adder_automa():
    return AdderAutoma()

@pytest.mark.asyncio
async def test_adder_automa_to_run(adder_automa: AdderAutoma, request, db_base_path):
    try:
        result = await adder_automa.arun(x=5)
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 6" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "adder_automa.bytes"
        version_file = db_base_path / "adder_automa.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

# Utility function to deserialize a TopGraph from a bytes file and a version file.
def deserialize_adder_automa(bytes_file, version_file):
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == AdderAutoma.SERIALIZATION_VERSION
        deserialized_graph = AdderAutoma.load_from_snapshot(snapshot)
        assert type(deserialized_graph) is AdderAutoma
        return deserialized_graph

@pytest.fixture
def adder_automa_deserialized(db_base_path):
        bytes_file = db_base_path / "adder_automa.bytes"
        version_file = db_base_path / "adder_automa.version"
        return deserialize_adder_automa(bytes_file, version_file)

@pytest.fixture
def feedback_yes(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="yes"
    )
    return feedback_yes

@pytest.mark.asyncio
async def test_adder_automa_deserialized_resume(feedback_yes, adder_automa_deserialized, request, db_base_path):
    result = await adder_automa_deserialized.arun(
        feedback_data=feedback_yes
    )
    assert result == 6 + 200 + 2

########## Test cases for Errors / Exceptions in concurrent automa ############

def test_worker_signature_errors():
    with pytest.raises(WorkerSignatureError, match="Unexpected arguments:"):
        class MySequentialAutoma_WithWrongDependencies(SequentialAutoma):
            @worker(key="func_1")
            async def func_1(self, x: int) -> int:
                return x + 1

            @worker(key="func_2", dependencies=["func_1"])
            async def func_2(self, x: int) -> int:
                return x + 2

def test_topology_change_errors():
    flow = MySequentialFlow()

    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        flow.add_func_as_worker(
            key="__tail__",
            func=func_1_async,
        )
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        flow.add_func_as_worker(
            key="func_1",
            func=func_1_async,
        )
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        flow.add_worker(
            key="__tail__",
            worker=Func2SyncWorker(),
        )
    with pytest.raises(AutomaRuntimeError, match="duplicate workers with the same key"):
        flow.add_worker(
            key="func_1",
            worker=Func2SyncWorker(),
        )
    with pytest.raises(AutomaRuntimeError, match="the reserved key `__tail__` is not allowed to be used"):
        @flow.worker(key="__tail__")
        async def func_5_async(automa, x: int) -> int:
            return x + 5

    with pytest.raises(AutomaRuntimeError, match=re.escape("remove_worker() is not allowed to be called on a sequential automa")):
        flow.remove_worker("func_2")
    with pytest.raises(AutomaRuntimeError, match=re.escape("add_dependency() is not allowed to be called on a sequential automa")):
        flow.add_dependency("__tail__", "func_1")

#############

class MySequentialAutoma_TryFerryTo(SequentialAutoma):
    @worker(key="func_1")
    async def func_1(self, x: int) -> int:
        return x + 1

    @worker(key="func_2")
    async def func_2(self, x: int) -> int:
        self.ferry_to("func_1") # This should raise an error
        return x + 2

@pytest.mark.asyncio
async def test_ferry_error():
    flow = MySequentialAutoma_TryFerryTo()
    with pytest.raises(AutomaRuntimeError, match=re.escape("ferry_to() is not allowed to be called on a sequential automa")):
        result = await flow.arun(x=100)
