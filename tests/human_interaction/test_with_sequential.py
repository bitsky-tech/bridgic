"""
Integration tests for human interaction in SequentialAutoma.

These tests verify human-in-the-loop interaction mechanism
with SequentialAutoma instances.
"""
import pytest

from bridgic.core.automa import worker
from bridgic.core.agentic import SequentialAutoma
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot


########## Test case: human interaction in sequential automa ############

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
    """Test human interaction in SequentialAutoma."""
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
        assert snapshot.serialization_version == SequentialAutoma.SERIALIZATION_VERSION
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
    """Test resuming SequentialAutoma after human interaction."""
    result = await adder_automa_deserialized.arun(
        feedback_data=feedback_yes
    )
    assert result == 6 + 200 + 2








