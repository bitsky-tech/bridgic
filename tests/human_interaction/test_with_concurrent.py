"""
Integration tests for human interaction in concurrent execution scenarios.

These tests verify human-in-the-loop interaction mechanism with concurrent
workers and thread pools.
"""
import pytest
import threading
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.automa.interaction import Event
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa import Snapshot


################################################################################
#### Test case: interact_with_human in run() vs. arun(); with def vs. async def.
################################################################################

class Flow5(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        event = Event(
            event_type="add_func_1",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        assert feedback.data == 1
        return input_x + feedback.data, context_for_test

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    def func_2(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        event = Event(
            event_type="add_func_2",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        assert feedback.data == 2
        return input_x + feedback.data, context_for_test

class Func3SyncWorkerV4(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        event = Event(
            event_type="add_func_3",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        assert feedback.data == 3
        return input_x + feedback.data, context_for_test

class Func4AsyncWorkerV4(Worker):
    async def arun(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        event = Event(
            event_type="add_func_4",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback: InteractionFeedback = self.interact_with_human(event)
        assert feedback.data == 4
        return input_x + feedback.data

@pytest.fixture(params=[True, False])
def flow_5_and_thread_name_prefix(request):
    is_thread_pool_privided = request.param
    if is_thread_pool_privided:
        thread_name_prefix = "flow_5_thread"
        thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=thread_name_prefix)
    else:
        thread_name_prefix = "bridgic-thread"
        thread_pool = None
    flow = Flow5(thread_pool=thread_pool)
    flow.add_worker(
        "func_3", 
        Func3SyncWorkerV4(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    flow.add_worker(
        "func_4", 
        Func4AsyncWorkerV4(),
        dependencies=["func_3"],
        is_output=True,
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

# Utility functions for test cases.
def assert_and_persist_one_interaction(e: InteractionException, expected_event_type: str, db_base_path, request):
    assert len(e.interactions) == 1
    assert e.interactions[0].event.event_type == expected_event_type
    assert type(e.snapshot.serialized_bytes) is bytes
    # Send e.interactions to human. Here is a simulation.
    interaction_ids = [interaction.interaction_id for interaction in e.interactions]
    request.config.cache.set("interaction_ids", interaction_ids)
    # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
    # Use the Automa instance / scenario name as the search index name.
    bytes_file = db_base_path / "flow_5.bytes"
    version_file = db_base_path / "flow_5.version"
    bytes_file.write_bytes(e.snapshot.serialized_bytes)
    version_file.write_text(e.snapshot.serialization_version)

def deserialize_flow_5(db_base_path):
    bytes_file = db_base_path / "flow_5.bytes"
    version_file = db_base_path / "flow_5.version"
    serialized_bytes = bytes_file.read_bytes()
    serialization_version = version_file.read_text()
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes, 
        serialization_version=serialization_version
    )
    # Snapshot is restored.
    assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
    deserialized_flow = Flow5.load_from_snapshot(snapshot)
    assert type(deserialized_flow) is Flow5
    return deserialized_flow

@pytest.mark.asyncio
async def test_flow_5_interact_1(flow_5_and_thread_name_prefix, db_base_path, request):
    """Test human interaction in concurrent flow - first interaction."""
    flow_5, thread_name_prefix = flow_5_and_thread_name_prefix

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5.arun(
            input_x=8, 
            context_for_test={
                "main_thread_id": threading.get_ident(),
                "thread_name_prefix_in_thread_pool": thread_name_prefix
            }
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_1", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_first(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_2(flow_5_deserialized_first, db_base_path, request):
    """Test human interaction in concurrent flow - second interaction."""
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 1
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_first.arun(
            feedback_data=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_2", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_second(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_3(flow_5_deserialized_second, db_base_path, request):
    """Test human interaction in concurrent flow - third interaction."""
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 2
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_second.arun(
            feedback_data=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_3", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_third(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_4(flow_5_deserialized_third, db_base_path, request):
    """Test human interaction in concurrent flow - fourth interaction."""
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 3
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_third.arun(
            feedback_data=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_4", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_fourth(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_5(flow_5_deserialized_fourth, request):
    """Test human interaction in concurrent flow - final completion."""
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 4
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    result = await flow_5_deserialized_fourth.arun(
        feedback_data=feedback
    )
    assert result == (8 + 1 + 2 + 3 + 4)








