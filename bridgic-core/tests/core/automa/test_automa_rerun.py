"""
Test cases for rerunning an Automa instance.
"""

import pytest
from bridgic.core.automa import worker, GraphAutoma
from bridgic.core.automa import RuntimeContext
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.types._serialization import Snapshot

#### Test case: rerun an Automa instance.

class ArithmeticAutoma(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int):
        return 3 * x

    @worker(dependencies=["start"], is_output=True)
    async def end(self, x: int):
        return x + 5

@pytest.fixture
def arithmetic():
    graph = ArithmeticAutoma()
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

#### Test case: rerun a nested Automa instance by ferry-to. The states (counter) of the nested Automa can be maintained after rerun.

class TopAutoma(GraphAutoma):
    # The start worker is a nested Automa which will be added by add_worker()

    @worker(dependencies=["start"], is_output=True)
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

    @worker(dependencies=["counter"], is_output=True)
    async def end(self, count: int):
        return ['bridgic'] * count

@pytest.fixture
def nested_automa():
    graph = NestedAutoma()
    return graph

@pytest.fixture
def topAutoma(nested_automa):
    graph = TopAutoma()
    graph.add_worker("start", nested_automa, is_start=True)
    return graph

@pytest.mark.asyncio
async def test_nested_automa_rerun(topAutoma):
    # First run.
    result = await topAutoma.arun()
    assert result == ['bridgic'] * 5

#### Test case: rerun an Automa that contains a worker with a human interaction.

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

class AdderAutoma(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"], is_output=True)
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
def adder_automa_deserialized_first(db_base_path):
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
async def test_adder_automa_deserialized_rerun_to_interaction(feedback_yes, adder_automa_deserialized_first, request, db_base_path):
    result = await adder_automa_deserialized_first.arun(
        interaction_feedback=feedback_yes
    )
    assert result == 6 + 200 + 2

    #### Test case continue: rerun the deserialized Automa for the second time.
    try:
        result = await adder_automa_deserialized_first.arun(x=15)
    except InteractionException as e:
        assert len(e.interactions) == 1
        for interaction in e.interactions:
            assert interaction.event.event_type == "if_add"
            assert "Current value is 16" in interaction.event.data["prompt_to_user"]
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "adder_automa_rerun.bytes"
        version_file = db_base_path / "adder_automa_rerun.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def adder_automa_deserialized_second(db_base_path):
        bytes_file = db_base_path / "adder_automa_rerun.bytes"
        version_file = db_base_path / "adder_automa_rerun.version"
        return deserialize_adder_automa(bytes_file, version_file)

@pytest.fixture
def feedback_no(request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feedback_no = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="no"
    )
    return feedback_no

@pytest.mark.asyncio
async def test_adder_automa_deserialized_rerun_to_end(feedback_no, adder_automa_deserialized_second):
    result = await adder_automa_deserialized_second.arun(
        interaction_feedback=feedback_no
    )
    assert result == 16 + 2

