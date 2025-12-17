import pytest

from bridgic.core.automa import GraphAutoma, worker, Snapshot
from bridgic.core.automa.args import System
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException


# Shared fixtures for all test cases
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


class AdderAutomaWithLocalSpace(GraphAutoma):
    """Test local space persistence with human interaction and ferry_to."""
    @worker(is_start=True)
    async def func_1(self, x: int, rtx = System("runtime_context")):
        local_space = self.get_local_space(rtx)
        x = local_space.get("x", x)
        print("func_1 before add 1:", x)
        local_space["x"] = x + 1
        print("func_1 after add 1:", local_space["x"])
        return local_space["x"]

    @worker(dependencies=["func_1"], is_output=True)
    async def func_2(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        print(feedback.data)
        if feedback.data == "yes":
            x += 200
        elif feedback.data == "no":
            self.ferry_to("func_1", x=x)
        return x + 2


@pytest.fixture
def adder_automa_with_local_space():
    return AdderAutomaWithLocalSpace()


@pytest.mark.asyncio
async def test_local_space_interact_serialized(adder_automa_with_local_space: AdderAutomaWithLocalSpace, request, db_base_path):
    """Test that human interaction triggers serialization with local space."""
    try:
        result = await adder_automa_with_local_space.arun(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 101, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("local_space_interaction_id", interaction_id)
        bytes_file = db_base_path / "adder_automa_local_space.bytes"
        version_file = db_base_path / "adder_automa_local_space.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_adder_automa_local_space(db_base_path):
    bytes_file = db_base_path / "adder_automa_local_space.bytes"
    version_file = db_base_path / "adder_automa_local_space.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_automa = AdderAutomaWithLocalSpace.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is AdderAutomaWithLocalSpace
    return deserialized_automa


@pytest.fixture
def local_space_feedback_yes(request):
    interaction_id = request.config.cache.get("local_space_interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback


@pytest.mark.asyncio
async def test_local_space_interact_with_yes_feedback(local_space_feedback_yes, deserialized_adder_automa_local_space):
    """Test that local space is preserved after deserialization with yes feedback."""
    result = await deserialized_adder_automa_local_space.arun(
        feedback_data=local_space_feedback_yes
    )
    assert result == 303


@pytest.fixture
def local_space_feedback_no(request):
    interaction_id = request.config.cache.get("local_space_interaction_id", None)
    print(f"no: interaction_id: {interaction_id}")
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="no"
    )
    return feedback


@pytest.mark.asyncio
async def test_local_space_interact_with_no_feedback(local_space_feedback_no, deserialized_adder_automa_local_space, request, db_base_path):
    """Test that ferry_to works with local space after human interaction (no feedback triggers loop)."""
    try:
        result = await deserialized_adder_automa_local_space.arun(
            feedback_data=local_space_feedback_no
        )
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 102, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("local_space_interaction_id_2", interaction_id)
        bytes_file = db_base_path / "adder_automa_local_space_no.bytes"
        version_file = db_base_path / "adder_automa_local_space_no.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_adder_automa_local_space_no(db_base_path):
    bytes_file = db_base_path / "adder_automa_local_space_no.bytes"
    version_file = db_base_path / "adder_automa_local_space_no.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_automa = AdderAutomaWithLocalSpace.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is AdderAutomaWithLocalSpace
    return deserialized_automa


@pytest.fixture
def local_space_feedback_2_yes(request):
    interaction_id = request.config.cache.get("local_space_interaction_id_2", None)
    print(f"yes: interaction_id: {interaction_id}")
    feedback_yes = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback_yes


@pytest.mark.asyncio
async def test_local_space_interact_with_ferry_to_then_yes(local_space_feedback_2_yes, deserialized_adder_automa_local_space_no):
    """Test that local space is updated correctly after ferry_to loop and yes feedback."""
    result = await deserialized_adder_automa_local_space_no.arun(
        feedback_data=local_space_feedback_2_yes
    )
    assert result == 304

