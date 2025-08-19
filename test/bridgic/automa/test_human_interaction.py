import pytest

from bridgic.automa import GraphAutoma
from bridgic.automa import worker
from bridgic.automa.interaction import Event
from bridgic.automa.interaction import InteractionFeedback, InteractionException
from bridgic.automa.serialization import Snapshot

################## Test cases for a Simple Automa ####################

class AdderAutoma(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"])
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            return x + 200
        return x

@pytest.fixture
def adder_automa():
    return AdderAutoma(output_worker_key="func_2")

@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

@pytest.mark.asyncio
async def test_adder_automa_interact(adder_automa: AdderAutoma, request, db_base_path):
    try:
        result = await adder_automa.process_async(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 101, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation with file storage.
        # Use the interaction_id as the search index.
        bytes_file = db_base_path / f"{interaction_id}.bytes"
        version_file = db_base_path / f"{interaction_id}.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def deserialized_adder_automa(adder_automa: AdderAutoma, request, db_base_path):
        interaction_id = request.config.cache.get("interaction_id", None)
        bytes_file = db_base_path / f"{interaction_id}.bytes"
        version_file = db_base_path / f"{interaction_id}.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = AdderAutoma.load_from_snapshot(snapshot)
        assert type(deserialized_automa) is AdderAutoma
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
async def test_adder_automa_interact_with_yes_feedback(interaction_feedback_1_yes, deserialized_adder_automa):
    result = await deserialized_adder_automa.process_async(
        x=100,
        interaction_feedback=interaction_feedback_1_yes
    )
    assert result == 301

@pytest.fixture
def interaction_feedback_1_no(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="no"
    )
    return feedback

@pytest.mark.asyncio
async def test_adder_automa_interact_with_no_feedback(interaction_feedback_1_no, deserialized_adder_automa):
    result = await deserialized_adder_automa.process_async(
        x=100,
        interaction_feedback=interaction_feedback_1_no
    )
    assert result == 101
