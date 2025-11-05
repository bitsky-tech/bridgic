import pytest

from typing import List, Tuple, Dict, Any
from bridgic.core.automa import GraphAutoma, worker, Snapshot, WorkerCallback
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.interaction import Event, FeedbackSender, Feedback, InteractionException, InteractionFeedback


class PostEventCallback(WorkerCallback):
    async def post_worker_execute(
        self, 
        key: str, 
        parent: GraphAutoma, 
        arguments: Dict[str, Any], 
        result: Any,
    ) -> None:
        event = Event(
            event_type="post_event",
            data="this is a post event callback"
        )
        parent.post_event(event)


class RequestFeedbackCallback(WorkerCallback):
    async def pre_worker_execute(
        self, 
        key: str, 
        parent: GraphAutoma, 
        arguments: Dict[str, Any],
    ) -> None:
        event = Event(
            event_type="request_feedback",
            data="Return yes to continue, otherwise return no to stop."
        )
        feedback = await parent.request_feedback_async(event)
        if feedback.data == "yes":
            pass
        else:
            raise ValueError("User returned no to stop.")


class InteractWithHumanCallback(WorkerCallback):
    async def pre_worker_execute(
        self, 
        key: str, 
        parent: GraphAutoma, 
        arguments: Dict[str, Any],
    ) -> None:
        event = Event(
            event_type="interact_with_human",
            data="Return yes to continue, otherwise return no to stop."
        )
        feedback = parent.interact_with_human(event)
        if feedback.data == "yes":
            pass
        else:
            raise ValueError("User returned no to stop.")


class RemoveWorkerCallback(WorkerCallback):
    async def post_worker_execute(
        self, 
        key: str, 
        parent: GraphAutoma, 
        arguments: Dict[str, Any], 
        result: Any,
    ) -> None:
        parent.remove_worker(key)


# - - - - - - - - - - - - - - - -
# test case: callback run post_event correctly
# - - - - - - - - - - - - - - - -
@pytest.fixture
def graph_post_event_with_callbacks():
    def handler_post_event(event: Event):
        data = event.data
        print(data)

    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True, callbacks=[PostEventCallback()])
        async def worker_1(self, x: int) -> int:
            return x + 1

    automa = MyGraph()
    automa.register_event_handler(event_type="post_event", event_handler=handler_post_event)
    return automa

@pytest.mark.asyncio
async def test_graph_post_event_with_callbacks(graph_post_event_with_callbacks: GraphAutoma, capsys):
    result = await graph_post_event_with_callbacks.arun(user_input=1)
    assert result == 3

    captured = capsys.readouterr()
    output = captured.out
    assert "this is a post event callback" in output


# - - - - - - - - - - - - - - - -
# test case: callback run request_feedback_async correctly
# - - - - - - - - - - - - - - - -
@pytest.fixture
def graph_request_feedback_async_with_callbacks():
    def handler_request_feedback_yes(event: Event, feedback_sender: FeedbackSender):
        data = event.data
        print(data)
        feedback_sender.send(Feedback(data="yes"))

    def handler_request_feedback_no(event: Event, feedback_sender: FeedbackSender):
        data = event.data
        print(data)
        feedback_sender.send(Feedback(data="no"))

    class MyGraph1(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], callbacks=[RequestFeedbackCallback()], is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    class MyGraph2(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], callbacks=[RequestFeedbackCallback()], is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    graph1 = MyGraph1()
    graph2 = MyGraph2()
    graph1.register_event_handler(event_type="request_feedback", event_handler=handler_request_feedback_yes)
    graph2.register_event_handler(event_type="request_feedback", event_handler=handler_request_feedback_no)
    return graph1, graph2

@pytest.mark.asyncio
async def test_graph_request_feedback_async_with_callbacks(graph_request_feedback_async_with_callbacks: Tuple[GraphAutoma, GraphAutoma], capsys):
    graph1, graph2 = graph_request_feedback_async_with_callbacks
    result = await graph1.arun(user_input=1)
    assert result == 3

    with pytest.raises(ValueError, match="User returned no to stop."):
        await graph2.arun(user_input=1)

    captured = capsys.readouterr()
    output = captured.out
    assert "Return yes to continue, otherwise return no to stop." in output


# - - - - - - - - - - - - - - - -
# test case: callback run interact_with_human correctly
# - - - - - - - - - - - - - - - -
@pytest.fixture(scope="session")  # Shared fixtures for all test cases.
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


class MyGraph(GraphAutoma):
    @worker(is_start=True)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1

    @worker(dependencies=["worker_0"], callbacks=[InteractWithHumanCallback()], is_output=True)
    async def worker_1(self, x: int) -> int:
        return x + 1


@pytest.fixture
def graph_interact_with_human_with_callbacks():
    return MyGraph()

@pytest.mark.asyncio
async def test_graph_interact_with_human_with_callbacks_save_serialized(graph_interact_with_human_with_callbacks: MyGraph, request, db_base_path):
    try:
        result = await graph_interact_with_human_with_callbacks.arun(user_input=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "interact_with_human"
        assert e.interactions[0].event.data == "Return yes to continue, otherwise return no to stop."
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "graph_interact_with_human_with_callbacks.bytes"
        version_file = db_base_path / "graph_interact_with_human_with_callbacks.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def deserialized_graph_interact_with_human_with_callbacks(graph_interact_with_human_with_callbacks, db_base_path):
        bytes_file = db_base_path / "graph_interact_with_human_with_callbacks.bytes"
        version_file = db_base_path / "graph_interact_with_human_with_callbacks.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_graph = MyGraph.load_from_snapshot(snapshot)
        assert type(deserialized_graph) is MyGraph
        return deserialized_graph

@pytest.fixture
def interaction_feedback_yes(request):
    interaction_id = request.config.cache.get("interaction_id", None)
    feedback = InteractionFeedback(
        interaction_id=interaction_id,
        data="yes"
    )
    return feedback


@pytest.mark.asyncio
async def test_graph_interact_with_human_with_callbacks_deserialized(interaction_feedback_yes, deserialized_graph_interact_with_human_with_callbacks):
    result = await deserialized_graph_interact_with_human_with_callbacks.arun(
        interaction_feedback=interaction_feedback_yes
    )
    assert result == 3


# - - - - - - - - - - - - - - - -
# test case: callback to remove worker correctly
# - - - - - - - - - - - - - - - -
@pytest.fixture
def graph_remove_worker_with_callbacks_1():
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], callbacks=[RemoveWorkerCallback()])
        async def worker_1(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_1", "worker_0"], is_output=True)
        async def worker_2(self, x: int) -> int:
            print(f"worker_2 is executing")
            return x + 1

    return MyGraph()

@pytest.mark.asyncio
async def test_graph_remove_worker_with_callbacks_1(graph_remove_worker_with_callbacks_1: GraphAutoma):
    result = await graph_remove_worker_with_callbacks_1.arun(user_input=1)
    assert result == None
    assert "worker_1" not in graph_remove_worker_with_callbacks_1.all_workers()


@pytest.fixture
def graph_remove_worker_with_callbacks_2():
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], callbacks=[RemoveWorkerCallback()], is_output=True)
        async def worker_1(self, user_input: int) -> int:
            return user_input + 1

    return MyGraph()

@pytest.mark.asyncio
async def test_graph_remove_worker_with_callbacks_2(graph_remove_worker_with_callbacks_2: GraphAutoma):
    result = await graph_remove_worker_with_callbacks_2.arun(user_input=1)
    assert result == None
    assert "worker_1" not in graph_remove_worker_with_callbacks_2.all_workers()