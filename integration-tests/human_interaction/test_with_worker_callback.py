import pytest
from typing import Dict, Any, Optional, TYPE_CHECKING

from bridgic.core.automa import GraphAutoma, worker, Snapshot
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException
from bridgic.core.automa.worker import WorkerCallback, WorkerCallbackBuilder

if TYPE_CHECKING:
    from bridgic.core.automa._automa import Automa


# Shared fixtures for all test cases
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


class InteractWithHumanCallback(WorkerCallback):
    """A callback that triggers human interaction."""
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        if parent:
            event = Event(
                event_type="interact_with_human",
                data="Return yes to continue, otherwise return no to stop."
            )
            feedback = parent.interact_with_human(event)
            if feedback.data == "yes":
                pass
            else:
                raise ValueError("User returned no to stop.")


class GraphWithInteractCallback(GraphAutoma):
    """Graph that uses InteractWithHumanCallback."""
    @worker(is_start=True)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1

    @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(InteractWithHumanCallback)], is_output=True)
    async def worker_1(self, x: int) -> int:
        return x + 1


@pytest.fixture
def graph_interact_with_human_callback():
    return GraphWithInteractCallback()


@pytest.mark.asyncio
async def test_callback_interact_with_human_save_serialized(graph_interact_with_human_callback: GraphWithInteractCallback, request, db_base_path):
    """Test that interact_with_human in callback triggers serialization."""
    try:
        result = await graph_interact_with_human_callback.arun(user_input=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "interact_with_human"
        assert e.interactions[0].event.data == "Return yes to continue, otherwise return no to stop."
        assert type(e.snapshot.serialized_bytes) is bytes
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("callback_interaction_id", interaction_id)
        bytes_file = db_base_path / "graph_interact_callback.bytes"
        version_file = db_base_path / "graph_interact_callback.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_graph_interact_callback(db_base_path):
    bytes_file = db_base_path / "graph_interact_callback.bytes"
    version_file = db_base_path / "graph_interact_callback.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_graph = GraphWithInteractCallback.load_from_snapshot(snapshot)
    assert type(deserialized_graph) is GraphWithInteractCallback
    return deserialized_graph


@pytest.fixture
def callback_feedback_yes(request):
    interaction_id = request.config.cache.get("callback_interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback


@pytest.mark.asyncio
async def test_callback_interact_with_human_deserialized(callback_feedback_yes, deserialized_graph_interact_callback):
    """Test that callback interact_with_human works after deserialization."""
    result = await deserialized_graph_interact_callback.arun(
        feedback_data=callback_feedback_yes
    )
    assert result == 3

