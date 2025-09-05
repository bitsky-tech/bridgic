"""
Test cases for the Bridgic Concurrency Model of Worker. Refer to the docstring of worker.py for more details.
"""

import pytest
import asyncio
import time
import threading
from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
from typing import Dict, Any
from bridgic.core.automa.worker import Worker
from concurrent.futures import ThreadPoolExecutor
from bridgic.core.automa.interaction import Event, Feedback, FeedbackSender
from bridgic.core.automa.interaction import InteractionFeedback, InteractionException
from bridgic.core.automa.serialization import Snapshot

########################################################
#### Test case: run() vs. arun(); def vs. async def.
########################################################

class Flow1(GraphAutoma):
    @worker(is_start=True)
    async def start(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        # TODO: top-down args mapping may be needed here...
        return input_x, context_for_test

    @worker(dependencies=["start"], args_mapping_rule=ArgsMappingRule.UNPACK)
    def func_1(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        time.sleep(0.01)
        return input_x + 1, context_for_test

class Func2SyncWorker(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        time.sleep(0.01)
        return input_x + 2, context_for_test

class MergeAsyncWorker(Worker):
    async def arun(
        self,
        *args
    ) -> int:
        input_x1 = args[0][0]
        context1_for_test = args[0][1]
        input_x2 = args[1][0]
        context2_for_test = args[1][1]
        assert threading.get_ident() == context1_for_test["main_thread_id"]
        assert threading.get_ident() == context2_for_test["main_thread_id"]
        return input_x1 * input_x2

@pytest.fixture(params=[True, False])
def flow_1_and_thread_name_prefix(request):
    is_thread_pool_privided = request.param
    if is_thread_pool_privided:
        thread_name_prefix = "flow_1_thread"
        thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=thread_name_prefix)
    else:
        thread_name_prefix = "bridgic-thread"
        thread_pool = None
    flow = Flow1(thread_pool=thread_pool)
    flow.add_worker(
        "func_2", 
        Func2SyncWorker(),
        dependencies=["start"],
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    flow.add_worker(
        "merge", 
        MergeAsyncWorker(),
        dependencies=["func_1", "func_2"],
        args_mapping_rule=ArgsMappingRule.AS_IS
    )
    flow.set_output_worker("merge")
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

@pytest.mark.asyncio
async def test_flow_1(flow_1_and_thread_name_prefix):
    flow_1, thread_name_prefix = flow_1_and_thread_name_prefix
    result = await flow_1.arun(
        input_x=5, 
        context_for_test={
            "main_thread_id": threading.get_ident(),
            "thread_name_prefix_in_thread_pool": thread_name_prefix
        }
    )
    assert result == (5 + 1) * (5 + 2)

################################################################################
#### Test case: ferry_to in run() vs. arun(); ferry_to with def vs. async def.
################################################################################

class Flow2(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        output_x = input_x + 1
        if output_x > 10:
            return output_x
        self.ferry_to("func_2", output_x, context_for_test)

    @worker()
    def func_2(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        time.sleep(0.01)
        self.ferry_to("func_3", input_x + 2, context_for_test)

class Func3SyncWorker(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        time.sleep(0.01)
        self.ferry_to("func_4", input_x + 3, context_for_test)

class Func4AsyncWorker(Worker):
    async def arun(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        self.ferry_to("func_1", input_x + 4, context_for_test)

@pytest.fixture(params=[True, False])
def flow_2_and_thread_name_prefix(request):
    is_thread_pool_privided = request.param
    if is_thread_pool_privided:
        thread_name_prefix = "flow_2_thread"
        thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=thread_name_prefix)
    else:
        thread_name_prefix = "bridgic-thread"
        thread_pool = None
    flow = Flow2(thread_pool=thread_pool)
    flow.add_worker(
        "func_3", 
        Func3SyncWorker(),
    )
    flow.add_worker(
        "func_4", 
        Func4AsyncWorker(),
    )
    flow.set_output_worker("func_1")
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

@pytest.mark.asyncio
async def test_flow_2(flow_2_and_thread_name_prefix):
    flow_2, thread_name_prefix = flow_2_and_thread_name_prefix
    result = await flow_2.arun(
        input_x=8, 
        context_for_test={
            "main_thread_id": threading.get_ident(),
            "thread_name_prefix_in_thread_pool": thread_name_prefix
        }
    )
    assert result == (8 + 1 + 2 + 3 + 4) + 1

################################################################################
#### Test case: add_worker in run() vs. arun(); add_worker with def vs. async def.
################################################################################

class Flow3(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        self.add_func_as_worker(
            "func_2", 
            self.func_2, 
            dependencies=["func_1"],
            args_mapping_rule=ArgsMappingRule.UNPACK
        )
        return input_x + 1, context_for_test

    def func_2(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        self.add_worker(
            "func_3", 
            Func3SyncWorkerV2(),
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK
        )
        return input_x + 2, context_for_test

class Func3SyncWorkerV2(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        self.parent.add_worker(
            "func_4", 
            Func4AsyncWorkerV2(),
            dependencies=["func_3"],
            args_mapping_rule=ArgsMappingRule.UNPACK
        )
        return input_x + 3, context_for_test

class Func4AsyncWorkerV2(Worker):
    async def arun(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        self.parent.add_worker(
            "func_5", 
            Func5SyncWorkerV2(),
            dependencies=["func_4"],
            args_mapping_rule=ArgsMappingRule.UNPACK
        )
        return input_x + 4, context_for_test

class Func5SyncWorkerV2(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        self.parent.set_output_worker("func_5")
        return input_x + 5

@pytest.fixture(params=[True, False])
def flow_3_and_thread_name_prefix(request):
    is_thread_pool_privided = request.param
    if is_thread_pool_privided:
        thread_name_prefix = "flow_3_thread"
        thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=thread_name_prefix)
    else:
        thread_name_prefix = "bridgic-thread"
        thread_pool = None
    flow = Flow3(thread_pool=thread_pool)
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

@pytest.mark.asyncio
async def test_flow_3(flow_3_and_thread_name_prefix):
    flow_3, thread_name_prefix = flow_3_and_thread_name_prefix
    result = await flow_3.arun(
        input_x=8, 
        context_for_test={
            "main_thread_id": threading.get_ident(),
            "thread_name_prefix_in_thread_pool": thread_name_prefix
        }
    )
    assert result == (8 + 1 + 2 + 3 + 4 + 5)

################################################################################
#### Test case: 
#### request_feedback in run() and with def.
#### request_feedback_async in arun() and with async def.
################################################################################

class Flow4(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        event = Event(
            event_type="add_func_1",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback = await self.request_feedback_async(event)
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
        feedback = self.request_feedback(event)
        assert feedback.data == 2
        return input_x + feedback.data, context_for_test

class Func3SyncWorkerV3(Worker):
    def run(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() != context_for_test["main_thread_id"]
        assert threading.current_thread().name.startswith(context_for_test["thread_name_prefix_in_thread_pool"])
        event = Event(
            event_type="add_func_3",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback = self.request_feedback(event)
        assert feedback.data == 3
        return input_x + feedback.data, context_for_test

class Func4AsyncWorkerV3(Worker):
    async def arun(self, input_x: int, context_for_test: Dict[str, Any]) -> int:
        assert threading.get_ident() == context_for_test["main_thread_id"]
        event = Event(
            event_type="add_func_4",
            data={
                "prompt_to_user": f"What number do you want to add to {input_x}?"
            }
        )
        feedback = await self.request_feedback_async(event)
        assert feedback.data == 4
        return input_x + feedback.data

@pytest.fixture(params=[True, False])
def flow_4_and_thread_name_prefix(request):
    is_thread_pool_privided = request.param
    if is_thread_pool_privided:
        thread_name_prefix = "flow_4_thread"
        thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix=thread_name_prefix)
    else:
        thread_name_prefix = "bridgic-thread"
        thread_pool = None
    flow = Flow4(thread_pool=thread_pool)
    flow.add_worker(
        "func_3", 
        Func3SyncWorkerV3(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    flow.add_worker(
        "func_4", 
        Func4AsyncWorkerV3(),
        dependencies=["func_3"],
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    flow.set_output_worker("func_4")
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

@pytest.mark.asyncio
async def test_flow_4(flow_4_and_thread_name_prefix):
    flow_4, thread_name_prefix = flow_4_and_thread_name_prefix
    async def give_feedback(feedback_sender: FeedbackSender, value: int):
        await asyncio.sleep(0.01)
        feedback_sender.send(Feedback(data=value))

    def event_handler_with_feedback(event: Event, feedback_sender: FeedbackSender):
        # This simulates the application layer code. After the user gives feedback, typically in a different task within the same event loop, the FeedbackSender would be called. This code here provides a basic simulation of that process.
        if event.event_type == "add_func_1":
            asyncio.create_task(give_feedback(feedback_sender, 1))
        elif event.event_type == "add_func_2":
            asyncio.create_task(give_feedback(feedback_sender, 2))
        elif event.event_type == "add_func_3":
            asyncio.create_task(give_feedback(feedback_sender, 3))
        elif event.event_type == "add_func_4":
            asyncio.create_task(give_feedback(feedback_sender, 4))
        else:
            assert False, f"Unexpected event type: {event.event_type}"
    
    flow_4.register_event_handler("add_func_1", event_handler_with_feedback)
    flow_4.register_event_handler("add_func_2", event_handler_with_feedback)
    flow_4.register_event_handler("add_func_3", event_handler_with_feedback)
    flow_4.register_event_handler("add_func_4", event_handler_with_feedback)

    result = await flow_4.arun(
        input_x=8, 
        context_for_test={
            "main_thread_id": threading.get_ident(),
            "thread_name_prefix_in_thread_pool": thread_name_prefix
        }
    )
    assert result == (8 + 1 + 2 + 3 + 4)

################################################################################
#### Test case: 
#### interact_with_human in run() vs. arun(); with def vs. async def.
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
        args_mapping_rule=ArgsMappingRule.UNPACK
    )
    flow.set_output_worker("func_4")
    yield flow, thread_name_prefix
    # clear up the thread pool
    if thread_pool:
        thread_pool.shutdown()

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

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
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 1
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_first.arun(
            interaction_feedback=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_2", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_second(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_3(flow_5_deserialized_second, db_base_path, request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 2
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_second.arun(
            interaction_feedback=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_3", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_third(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_4(flow_5_deserialized_third, db_base_path, request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 3
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    with pytest.raises(InteractionException) as excinfo:
        result = await flow_5_deserialized_third.arun(
            interaction_feedback=feedback
        )
    e = excinfo.value
    assert_and_persist_one_interaction(e, "add_func_4", db_base_path, request)

@pytest.fixture
def flow_5_deserialized_fourth(db_base_path):
        return deserialize_flow_5(db_base_path)

@pytest.mark.asyncio
async def test_flow_5_interact_5(flow_5_deserialized_fourth, db_base_path, request):
    interaction_ids = request.config.cache.get("interaction_ids", None)
    feeback_data = 4
    # Mock a feedback for each interaction_id.
    feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data=feeback_data
    )

    result = await flow_5_deserialized_fourth.arun(
        interaction_feedback=feedback
    )
    assert result == 8 + 1 + 2 + 3 + 4

################################################################################
#### Test case: 
#### Raise exceptions in run() vs. arun(); with def vs. async def.
################################################################################

class Graph_Exception_1(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, input_x: int) -> int:
        raise Exception("exception in async def")

@pytest.fixture
def graph_exception_1():
    graph = Graph_Exception_1(output_worker_key="func_1")
    return graph

@pytest.mark.asyncio
async def test_graph_exception_1(graph_exception_1):
    with pytest.raises(Exception, match="exception in async def"):
        await graph_exception_1.arun(input_x=1)

class Graph_Exception_2(GraphAutoma):
    @worker(is_start=True)
    def func_2(self, input_x: int) -> int:
        raise Exception("exception in normal def")

@pytest.fixture
def graph_exception_2():
    graph = Graph_Exception_2(output_worker_key="func_2")
    return graph

@pytest.mark.asyncio
async def test_graph_exception_2(graph_exception_2):
    with pytest.raises(Exception, match="exception in normal def"):
        await graph_exception_2.arun(input_x=1)

class Graph_Exception_3(GraphAutoma):
    ...

class Func3ExceptionSyncWorker(Worker):
    def run(self, input_x: int) -> int:
        raise Exception("exception in sync worker")

@pytest.fixture
def graph_exception_3():
    graph = Graph_Exception_3(output_worker_key="func_3")
    graph.add_worker(
        "func_3", 
        Func3ExceptionSyncWorker(),
        is_start=True,
    )
    return graph

@pytest.mark.asyncio
async def test_graph_exception_3(graph_exception_3):
    with pytest.raises(Exception, match="exception in sync worker"):
        await graph_exception_3.arun(input_x=1)

class Graph_Exception_4(GraphAutoma):
    ...

class Func4ExceptionAsyncWorker(Worker):
    async def arun(self, input_x: int) -> int:
        raise Exception("exception in async worker")

@pytest.fixture
def graph_exception_4():
    graph = Graph_Exception_4(output_worker_key="func_4")
    graph.add_worker(
        "func_4", 
        Func4ExceptionAsyncWorker(),
        is_start=True,
    )
    return graph

@pytest.mark.asyncio
async def test_graph_exception_4(graph_exception_4):
    with pytest.raises(Exception, match="exception in async worker"):
        await graph_exception_4.arun(input_x=1)
