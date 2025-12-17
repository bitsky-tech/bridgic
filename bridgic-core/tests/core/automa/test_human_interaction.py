"""
Unit tests for basic human interaction in GraphAutoma.

NOTE: These tests verify the fundamental human-in-the-loop interaction mechanism. Complex nested 
Automa interaction tests have been moved to another directory: `integration-tests/human_interaction/`.
"""
import pytest

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa import worker
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot


# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


################################################################################
# Test Case 1: Basic human interaction in middle worker
################################################################################

class AdderAutoma1(GraphAutoma):
    """Simple Automa with human interaction in a middle worker."""
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
def adder_automa1():
    return AdderAutoma1()


@pytest.mark.asyncio
async def test_adder_automa_1_interact_serialized(adder_automa1: AdderAutoma1, request, db_base_path):
    """Test that human interaction triggers InteractionException with correct data."""
    try:
        result = await adder_automa1.arun(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 101, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Save interaction data for subsequent tests
        request.config.cache.set("interaction_id_1", e.interactions[0].interaction_id)
        bytes_file = db_base_path / "adder_automa1.bytes"
        version_file = db_base_path / "adder_automa1.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_adder_automa1(db_base_path):
    bytes_file = db_base_path / "adder_automa1.bytes"
    version_file = db_base_path / "adder_automa1.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_automa = AdderAutoma1.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is AdderAutoma1
    return deserialized_automa


@pytest.fixture
def interaction_feedback_1_yes(request):
    interaction_id = request.config.cache.get("interaction_id_1", None)
    return InteractionFeedback(interaction_id=interaction_id, data="yes")


@pytest.fixture
def interaction_feedback_1_no(request):
    interaction_id = request.config.cache.get("interaction_id_1", None)
    return InteractionFeedback(interaction_id=interaction_id, data="no")


@pytest.mark.asyncio
async def test_adder_automa_1_interact_with_yes_feedback(interaction_feedback_1_yes, deserialized_adder_automa1):
    """Test resuming Automa with 'yes' feedback adds 200 to the value."""
    result = await deserialized_adder_automa1.arun(feedback_data=interaction_feedback_1_yes)
    assert result == 303  # 100 + 1 + 200 + 2


@pytest.mark.asyncio
async def test_adder_automa_1_interact_with_no_feedback(interaction_feedback_1_no, deserialized_adder_automa1):
    """Test resuming Automa with 'no' feedback does not add 200."""
    result = await deserialized_adder_automa1.arun(feedback_data=interaction_feedback_1_no)
    assert result == 103  # 100 + 1 + 2


################################################################################
# Test Case 2: Basic human interaction in start worker
################################################################################

class AdderAutoma2(GraphAutoma):
    """Simple Automa with human interaction in the start worker."""
    @worker(is_start=True)
    async def func_1(self, x: int):
        event = Event(
            event_type="if_add",
            data={
                "prompt_to_user": f"Current value is {x}, do you want to add another 200 to it (yes/no) ?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        if feedback.data == "yes":
            x += 200
        return x + 1

    @worker(dependencies=["func_1"], is_output=True)
    async def func_2(self, x: int):
        return x + 2


@pytest.fixture
def adder_automa2():
    return AdderAutoma2()


@pytest.mark.asyncio
async def test_adder_automa_2_interact_serialized(adder_automa2: AdderAutoma2, request, db_base_path):
    """Test human interaction in start worker triggers InteractionException."""
    try:
        result = await adder_automa2.arun(x=100)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is 100, do you want to add another 200 to it (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        request.config.cache.set("interaction_id_2", e.interactions[0].interaction_id)
        bytes_file = db_base_path / "adder_automa2.bytes"
        version_file = db_base_path / "adder_automa2.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)


@pytest.fixture
def deserialized_adder_automa2(db_base_path):
    bytes_file = db_base_path / "adder_automa2.bytes"
    version_file = db_base_path / "adder_automa2.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_automa = AdderAutoma2.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is AdderAutoma2
    return deserialized_automa


@pytest.fixture
def interaction_feedback_2_yes(request):
    interaction_id = request.config.cache.get("interaction_id_2", None)
    return InteractionFeedback(interaction_id=interaction_id, data="yes")


@pytest.mark.asyncio
async def test_adder_automa_2_interact_with_yes_feedback(interaction_feedback_2_yes, deserialized_adder_automa2):
    """Test resuming Automa with 'yes' feedback in start worker."""
    result = await deserialized_adder_automa2.arun(feedback_data=interaction_feedback_2_yes)
    assert result == 303  # 100 + 200 + 1 + 2
