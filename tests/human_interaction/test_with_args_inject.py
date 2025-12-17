import pytest

from bridgic.core.automa import GraphAutoma, worker, Snapshot
from bridgic.core.automa.args import From
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException


# Shared fixtures for all test cases
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


class AutomaSerializationWithFrom(GraphAutoma):
    """Test that From parameter injection is preserved after serialization."""
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
async def test_automa_serialization_with_from(automa_serialization_with_from: GraphAutoma, request, db_base_path):
    """Test that human interaction triggers serialization with From parameter."""
    try:
        result = await automa_serialization_with_from.arun(x=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is x: 3, y: 2, do you want to add another 200 to them (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("from_interaction_id", interaction_id)
        # Persist the snapshot to an external storage.
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
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_automa = AutomaSerializationWithFrom.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is AutomaSerializationWithFrom
    return deserialized_automa


@pytest.fixture
def from_interaction_feedback_yes(request):
    interaction_id = request.config.cache.get("from_interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback


@pytest.mark.asyncio
async def test_automa_deserialization_with_from(
    from_interaction_feedback_yes, 
    deserialized_automa_serialization_with_from: GraphAutoma
):
    """Test that From parameter injection works after deserialization."""
    result = await deserialized_automa_serialization_with_from.arun(
        feedback_data=from_interaction_feedback_yes
    )
    assert result == 405

