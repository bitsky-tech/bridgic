import pytest
import uuid
import asyncio
import re

from typing import List, Tuple, Dict, Any, Union, Optional
from contextvars import ContextVar
from bridgic.core.automa import GraphAutoma, worker, Snapshot, RunningOptions, Automa
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from bridgic.core.automa.interaction import Event, FeedbackSender, Feedback, InteractionException, InteractionFeedback
from bridgic.core.config import GlobalSetting
from bridgic.core.utils._console import printer, colored


class PostEventCallback(WorkerCallback):
    async def on_worker_end(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        if parent:
            event = Event(
                event_type="post_event",
                data="this is a post event callback"
            )
            parent.post_event(event)


class RequestFeedbackCallback(WorkerCallback):
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        if parent:
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
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
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


class RemoveWorkerCallback(WorkerCallback):
    async def on_worker_end(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        if parent:
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

        @worker(dependencies=["worker_0"], is_output=True, callback_builders=[WorkerCallbackBuilder(PostEventCallback)])
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

        @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(RequestFeedbackCallback)], is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    class MyGraph2(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(RequestFeedbackCallback)], is_output=True)
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

    @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(InteractWithHumanCallback)], is_output=True)
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
        feedback_data=interaction_feedback_yes
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

        @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(RemoveWorkerCallback)])
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

        @worker(dependencies=["worker_0"], callback_builders=[WorkerCallbackBuilder(RemoveWorkerCallback)], is_output=True)
        async def worker_1(self, user_input: int) -> int:
            return user_input + 1

    return MyGraph()

@pytest.mark.asyncio
async def test_graph_remove_worker_with_callbacks_2(graph_remove_worker_with_callbacks_2: GraphAutoma):
    result = await graph_remove_worker_with_callbacks_2.arun(user_input=1)
    assert result == None
    assert "worker_1" not in graph_remove_worker_with_callbacks_2.all_workers()


# - - - - - - - - - - - - - - - -
# test case: GlobalSetting callback_builders
# - - - - - - - - - - - - - - - -
class GlobalCallback(WorkerCallback):
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        print(f"global callback for {key}")


@pytest.fixture
def graph_with_global_setting():
    # Set global callback
    GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(GlobalCallback)])

    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    return MyGraph()

@pytest.mark.asyncio
async def test_global_setting_callback(graph_with_global_setting: GraphAutoma, capsys):
    result = await graph_with_global_setting.arun(user_input=1)
    assert result == 3

    captured = capsys.readouterr()
    output = captured.out

    assert "global callback for worker_0" in output
    assert "global callback for worker_1" in output

    # Clean up: reset global setting
    GlobalSetting.set(callback_builders=[])

# - - - - - - - - - - - - - - - -
# test case: RunningOptions callback_builders
# - - - - - - - - - - - - - - - -
class AutomaCallback(WorkerCallback):
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        print(f"automa callback for {key}")

class AutomaObserveValueErrorCallback(WorkerCallback):
    async def on_worker_error(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: ValueError = None,
    ) -> bool:
        print(f"automa callback for {key} to observe value error: {str(error)}")
        return False

class AutomaSuppressValueErrorCallback(WorkerCallback):
    async def on_worker_error(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: ValueError = None,
    ) -> bool:
        print(f"automa callback for {key} to suppress value error: {str(error)}")
        return True


@pytest.fixture
def graph_with_running_options():
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(AutomaCallback)])
    return MyGraph(running_options=running_options)

@pytest.mark.asyncio
async def test_running_options_callback(graph_with_running_options: GraphAutoma, capsys):
    result = await graph_with_running_options.arun(user_input=1)
    assert result == 3

    captured = capsys.readouterr()
    output = captured.out

    assert "automa callback for worker_0" in output
    assert "automa callback for worker_1" in output

@pytest.fixture
def graph_with_running_options_with_observed_value_error():
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True)
        async def worker_1(self, x: int) -> int:
            raise ValueError("Test ValueError")

    running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(AutomaObserveValueErrorCallback)])
    return MyGraph(running_options=running_options)

@pytest.mark.asyncio
async def test_running_options_callback_with_error_observed(graph_with_running_options_with_observed_value_error: GraphAutoma, capsys):
    with pytest.raises(ValueError, match="Test ValueError"):
        await graph_with_running_options_with_observed_value_error.arun(user_input=1)

    captured = capsys.readouterr()
    output = captured.out
    assert "automa callback for worker_1 to observe value error: Test ValueError" in output

@pytest.fixture
def graph_with_running_options_with_value_error():
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True)
        async def worker_1(self, x: int) -> int:
            raise ValueError("Test ValueError")

    running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(AutomaSuppressValueErrorCallback)])
    return MyGraph(running_options=running_options)

@pytest.mark.asyncio
async def test_running_options_callback_with_error_suppressed(graph_with_running_options_with_value_error: GraphAutoma, capsys):
    result = await graph_with_running_options_with_value_error.arun(user_input=1)
    assert result == None

    captured = capsys.readouterr()
    output = captured.out
    assert "automa callback for worker_1 to suppress value error: Test ValueError" in output

@pytest.fixture
def graph_with_all_three_layers():
    GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(GlobalCallback)])

    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1

        @worker(
            dependencies=["worker_0"], 
            is_output=True,
            callback_builders=[WorkerCallbackBuilder(PostEventCallback)]
        )
        async def worker_1(self, x: int) -> int:
            return x + 1

    running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(AutomaCallback)])
    return MyGraph(running_options=running_options)

@pytest.mark.asyncio
async def test_all_callback_builders_merged(graph_with_all_three_layers: GraphAutoma, capsys):
    result = await graph_with_all_three_layers.arun(user_input=1)
    assert result == 3

    captured = capsys.readouterr()
    output = captured.out

    assert "global callback for worker_0" in output
    assert "automa callback for worker_0" in output
    assert "global callback for worker_1" in output
    assert "automa callback for worker_1" in output

    # Clean up: reset global setting
    GlobalSetting.set(callback_builders=[])


# - - - - - - - - - - - - - - - -
# test case: callback on_worker_error with specific exception type
# - - - - - - - - - - - - - - - -
class ValueErrorCallback(WorkerCallback):
    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: ValueError = None,  # Specific type annotation
    ) -> bool:
        print(f"ValueError handled for {key}: {str(error)}")
        return True  # Request to suppress the exception


class TypeErrorCallback(WorkerCallback):
    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: TypeError = None,  # Specific type annotation
    ) -> bool:
        print(f"TypeError handled for {key}: {str(error)}")
        return True  # Request to suppress the exception


class UnionExceptionCallback(WorkerCallback):
    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: Union[ValueError, TypeError] = None,  # Union type annotation
    ) -> bool:
        print(f"Union error handled for {key}: {str(error)}")
        return True  # Request to suppress the exception

class BaseExceptionCallback(WorkerCallback):
    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        error: Exception = None,  # Base class annotation
    ) -> bool:
        print(f"Exception handled for {key}: {str(error)}")
        return True  # Request to suppress the exception


@pytest.fixture
def graph_with_inheritance_error():
    class MyGraph(GraphAutoma):
        @worker(
            is_start=True,
            callback_builders=[
                WorkerCallbackBuilder(BaseExceptionCallback),
            ]
        )
        async def worker_0(self, user_input: int) -> int:
            raise ValueError("Test ValueError - should match Exception")

    return MyGraph()


@pytest.mark.asyncio
async def test_on_worker_error_inheritance_matching(graph_with_inheritance_error: GraphAutoma, capsys):
    """Test that exception matching works with inheritance relationship"""
    result = await graph_with_inheritance_error.arun(user_input=1)
    assert result is None

    captured = capsys.readouterr()
    output = captured.out
    assert "Exception handled for worker_0" in output
    assert "Test ValueError - should match Exception" in output

@pytest.fixture
def graph_with_multiple_callbacks_value_error():
    class MyGraph(GraphAutoma):
        @worker(
            is_start=True,
            callback_builders=[
                WorkerCallbackBuilder(ValueErrorCallback),
                WorkerCallbackBuilder(TypeErrorCallback),
                WorkerCallbackBuilder(UnionExceptionCallback),
            ]
        )
        async def worker_0(self, user_input: int) -> int:
            raise ValueError("Test ValueError with multiple callbacks")

    return MyGraph()


@pytest.fixture
def graph_with_multiple_callbacks_runtime_error():
    class MyGraph(GraphAutoma):
        @worker(
            is_start=True,
            callback_builders=[
                WorkerCallbackBuilder(ValueErrorCallback),
                WorkerCallbackBuilder(TypeErrorCallback),
                WorkerCallbackBuilder(UnionExceptionCallback),
            ]
        )
        async def worker_0(self, user_input: int) -> int:
            raise RuntimeError("Test RuntimeError - not in Union")

    return MyGraph()

@pytest.mark.asyncio
async def test_on_worker_error_with_matched_callbacks(graph_with_multiple_callbacks_value_error: GraphAutoma, capsys):
    result = await graph_with_multiple_callbacks_value_error.arun(user_input=1)
    assert result is None

    captured = capsys.readouterr()
    output = captured.out
    assert "ValueError handled for worker_0" in output
    assert "TypeError handled for worker_0" not in output
    assert "Union error handled for worker_0" in output

@pytest.mark.asyncio
async def test_on_worker_error_no_mathed_callbacks(graph_with_multiple_callbacks_runtime_error: GraphAutoma, capsys):
    """Test that Union callback doesn't handle exceptions not in the Union"""
    # Should raise exception because RuntimeError is not in Union[ValueError, TypeError]
    with pytest.raises(RuntimeError, match="Test RuntimeError - not in Union"):
        await graph_with_multiple_callbacks_runtime_error.arun(user_input=1)


# - - - - - - - - - - - - - - - -
# Test case: nested automa callback propagation with delayed worker addition
# - - - - - - - - - - - - - - - -
class TopLevelCallback(WorkerCallback):
    """Callback for top-level automa"""
    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        if is_top_level:
            self.trace_id = ContextVar("trace_id")
            self.trace_id.set(f"root-{uuid.uuid4().hex[:8]}")
            self.span_id = ContextVar("span_id")
            self.span_id.set(f"span-{key}")
            parent_span_id = None
        else:
            parent_span_id = self.span_id.get()
            self.span_id.set(f"span-{key}")

        printer.print(
            f"[TopLevel] on_worker_start: {key}, "
            f"trace_id={colored(self.trace_id.get(), 'red')}, "
            f"parent_span_id={colored(parent_span_id, 'blue')}, "
            f"span_id={colored(self.span_id.get(), 'yellow')}"
        )

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        printer.print(f"[TopLevel] on_worker_end: {key}, result={result}")

class MiddleLevelCallback(WorkerCallback):
    """Callback for middle-level automa"""
    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        printer.print(f"[MiddleLevel] on_worker_start: {key}")

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        printer.print(f"[MiddleLevel] on_worker_end: {key}, result={result}")


class InnerLevelCallback(WorkerCallback):
    """Callback for inner-level automa"""
    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        printer.print(f"[InnerLevel] on_worker_start: {key}")

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[Automa] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        printer.print(f"[InnerLevel] on_worker_end: {key}, result={result}")


@pytest.fixture
def nested_automa_with_delayed_addition():
    """Test nested automa with delayed worker addition"""
    
    class InnerAutoma(GraphAutoma):
        @worker(is_start=True)
        async def inner_start(self, x: int) -> int:
            # Dynamically add a worker during execution
            self.add_func_as_worker(
                key="inner_dynamic",
                func=self.inner_dynamic_func,
                dependencies=["inner_start"],
                is_output=True,
            )
            return x + 1

        async def inner_dynamic_func(self, x: int) -> int:
            return x + 2

    class MiddleAutoma(GraphAutoma):
        @worker(is_start=True)
        async def middle_start(self, x: int) -> int:
            return x + 10

    class TopAutoma(GraphAutoma):
        @worker(is_start=True)
        async def top_start(self, x: int) -> int:
            return x + 100

    inner_automa = InnerAutoma(
        running_options=RunningOptions(
            callback_builders=[WorkerCallbackBuilder(InnerLevelCallback)]
        )
    )

    middle_automa = MiddleAutoma(
        running_options=RunningOptions(
            callback_builders=[WorkerCallbackBuilder(MiddleLevelCallback)]
        )
    )

    top_automa = TopAutoma(
        name="top_automa_instance",
        running_options=RunningOptions(
            callback_builders=[WorkerCallbackBuilder(TopLevelCallback)]
        )
    )
    middle_automa.add_worker(
        key="inner_automa",
        worker=inner_automa,
        dependencies=["middle_start"],
        is_output=True,
    )
    top_automa.add_worker(
        key="middle_automa",
        worker=middle_automa,
        dependencies=["top_start"],
        is_output=True,
    )

    return top_automa


@pytest.mark.asyncio
async def test_nested_automa_delayed_worker_addition(nested_automa_with_delayed_addition: GraphAutoma, capsys):
    """Test that callbacks are propagated to dynamically added workers in nested automa"""
    result = await nested_automa_with_delayed_addition.arun(x=1)
    assert result == 114  # 1 + 100 + 10 + 1 + 2 = 114

    captured = capsys.readouterr()
    output = captured.out
    # printer.print(f"\n\n{output}")

    assert "[TopLevel] on_worker_start: top_automa_instance" in output
    assert "[TopLevel] on_worker_end: top_automa_instance" in output

    assert "[TopLevel] on_worker_start: top_start" in output
    assert "[TopLevel] on_worker_end: top_start" in output

    assert "[TopLevel] on_worker_start: middle_automa" in output
    assert "[TopLevel] on_worker_end: middle_automa" in output
    assert "[MiddleLevel] on_worker_start: middle_automa" in output
    assert "[MiddleLevel] on_worker_end: middle_automa" in output

    assert "[TopLevel] on_worker_start: middle_start" in output
    assert "[TopLevel] on_worker_end: middle_start" in output
    assert "[MiddleLevel] on_worker_start: middle_start" in output
    assert "[MiddleLevel] on_worker_end: middle_start" in output

    assert "[TopLevel] on_worker_start: inner_automa" in output
    assert "[TopLevel] on_worker_end: inner_automa" in output
    assert "[MiddleLevel] on_worker_start: inner_automa" in output
    assert "[MiddleLevel] on_worker_end: inner_automa" in output
    assert "[InnerLevel] on_worker_start: inner_automa" in output
    assert "[InnerLevel] on_worker_end: inner_automa" in output

    assert "[TopLevel] on_worker_start: inner_start" in output
    assert "[TopLevel] on_worker_end: inner_start" in output
    assert "[MiddleLevel] on_worker_start: inner_start" in output
    assert "[MiddleLevel] on_worker_end: inner_start" in output
    assert "[InnerLevel] on_worker_start: inner_start" in output
    assert "[InnerLevel] on_worker_end: inner_start" in output

    assert "[TopLevel] on_worker_start: inner_dynamic" in output
    assert "[TopLevel] on_worker_end: inner_dynamic" in output
    assert "[MiddleLevel] on_worker_start: inner_dynamic" in output
    assert "[MiddleLevel] on_worker_end: inner_dynamic" in output
    assert "[InnerLevel] on_worker_start: inner_dynamic" in output
    assert "[InnerLevel] on_worker_end: inner_dynamic" in output

    # Verify callback execution order
    lines = list(filter(lambda line: len(line.strip()) > 0, output.split('\n')))
    assert len(lines) == 30


# - - - - - - - - - - - - - - - -
# Test case: context isolation when using callbacks
# - - - - - - - - - - - - - - - -
@pytest.fixture
def simple_automa_1_with_context_callback():
    """Create a simple automa with TopLevelCallback that uses ContextVar"""
    class SimpleAutoma(GraphAutoma):
        @worker(is_start=True, is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1
    
    automa = SimpleAutoma(
        name="simple_automa_1",
        running_options=RunningOptions(
            callback_builders=[WorkerCallbackBuilder(TopLevelCallback)]
        )
    )
    
    return automa

@pytest.fixture
def simple_automa_2_with_context_callback():
    """Create a simple automa with TopLevelCallback that uses ContextVar"""
    class SimpleAutoma(GraphAutoma):
        @worker(is_start=True, is_output=True)
        async def worker_1(self, x: int) -> int:
            return x + 1

    automa = SimpleAutoma(
        name="simple_automa_2",
        running_options=RunningOptions(
            callback_builders=[WorkerCallbackBuilder(TopLevelCallback)]
        )
    )

    return automa


@pytest.mark.asyncio
async def test_context_isolation(
    simple_automa_1_with_context_callback: GraphAutoma,
    simple_automa_2_with_context_callback: GraphAutoma,
    capsys,
):
    """
    Test that ContextVar values are isolated between concurrent arun() calls.
    
    This test verifies that when multiple automa instances (or the same instance) 
    are run concurrently, the ContextVar values (like trace_id) are properly isolated, 
    ensuring that each arun() call gets its own context copy even when executed in parallel.
    """
    automa_1 = simple_automa_1_with_context_callback
    automa_2 = simple_automa_2_with_context_callback
    
    # Collect all trace_ids from multiple iterations
    all_trace_ids = []
    
    # Run 3 iterations, each with two concurrent arun() calls
    # This tests Context isolation across multiple concurrent executions
    for iteration in range(3):
        # Run two arun() calls concurrently on different automa instances
        results = await asyncio.gather(
            automa_1.arun(x=1),
            automa_2.arun(x=1),
        )
        
        # Verify all results are correct
        assert results == [2, 2]
        
        # Capture output for this iteration
        captured = capsys.readouterr()
        output = captured.out
        
        # Extract trace_ids from top-level automa callbacks only
        automa_1_pattern = r'on_worker_start: simple_automa_1.*?root-([a-f0-9]{8})'
        automa_2_pattern = r'on_worker_start: simple_automa_2.*?root-([a-f0-9]{8})'
        
        automa_1_match = re.search(automa_1_pattern, output)
        automa_2_match = re.search(automa_2_pattern, output)
        
        assert automa_1_match is not None, f"Could not find trace_id for automa_1 in iteration {iteration + 1}. Output: {output}"
        assert automa_2_match is not None, f"Could not find trace_id for automa_2 in iteration {iteration + 1}. Output: {output}"
        
        trace_id_1 = automa_1_match.group(1)
        trace_id_2 = automa_2_match.group(1)
        
        # Verify the two trace_ids are different within the same iteration
        assert trace_id_1 != trace_id_2, (
            f"Trace IDs should be different for concurrent execution in iteration {iteration + 1}. "
            f"Got trace_id_1={trace_id_1}, trace_id_2={trace_id_2}"
        )
        
        # Collect trace_ids
        all_trace_ids.append(trace_id_1)
        all_trace_ids.append(trace_id_2)
    
    # Verify all trace_ids are unique (Context isolation across all executions)
    unique_trace_ids = set(all_trace_ids)
    assert len(unique_trace_ids) == 6, (
        f"All trace_ids should be unique across all iterations. "
        f"Got {len(unique_trace_ids)} unique values out of {len(all_trace_ids)} total. "
        f"Trace IDs: {all_trace_ids}"
    )
