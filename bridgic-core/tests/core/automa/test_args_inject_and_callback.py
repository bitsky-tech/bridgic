import pytest
from typing import Tuple, Dict, Any, Optional

from bridgic.core.automa import GraphAutoma, Snapshot, worker, AutomaRuntimeError
from bridgic.core.automa.args import From, ArgsMappingRule, System, Distribute, ResultDispatchRule
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from bridgic.core.types._error import WorkerArgsInjectionError

########################################################
#### All callbacks used in the test cases
########################################################

# - - - - - - - - - - - - - - - -
# Test case: All kinds of workers running correctly
# - - - - - - - - - - - - - - - -
@pytest.fixture
def args_inject_and_callback_graph():
    """
    test args inject and callback with all kinds of workers
    """
    # Define a callback to print the worker key and the result.
    class PrintCallback(WorkerCallback):
        async def on_worker_start(
            self, 
            key: str,
            is_top_level: bool = False,
            parent: Optional[Automa] = None,
            arguments: Dict[str, Any] = None,
        ) -> None:
            print(f"Pre callback for worker {key}")
        
        async def on_worker_end(
            self, 
            key: str,
            is_top_level: bool = False,
            parent: Optional[Automa] = None,
            arguments: Dict[str, Any] = None,
            result: Any = None,
        ) -> None:
            print(f"Post callback for worker {key}")

    global my_graph_1
    my_graph_1 = GraphAutoma()
    
    # Define a graph with async & sync workers, add_worker() & add_func_as_worker() with callbacks.
    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1  # 2 

        @worker(is_start=True)
        async def worker_01(self, user_input: int) -> int:
            res = user_input + 1
            assert res == 3
            return res

        @worker(dependencies=["worker_0"])
        async def worker_1(self, x: int, z: int = 1) -> int:
            return x + 1  # 3

        @worker(dependencies=["worker_1"], callback_builders=[WorkerCallbackBuilder(PrintCallback)])
        async def worker_2(
            self, x: int, 
            y: int = From("worker_0"), 
            automa: GraphAutoma = System("automa"),
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            return x + y  # 5

        @worker(dependencies=["worker_2"], callback_builders=[WorkerCallbackBuilder(PrintCallback)])
        def worker_3(
            self, 
            x: int, 
            y: int = From("worker_0"), 
            automa: GraphAutoma = System("automa"),
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            return x + y  # 7

    global my_graph

    def worker_4(
        x: int, 
        y: int = From("worker_0"), 
        automa: GraphAutoma = System("automa"),
        sub_automa: GraphAutoma = System("automa:my_graph_1"),
    ) -> int:
        assert automa is my_graph
        assert sub_automa is my_graph_1
        return x + y  # 9

    class MyWorker(Worker):
        def run(
            self, x: int, 
            y: int = From("worker_0"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is my_graph
            assert sub_automa is my_graph_1
            return x + y  # 11

    class MyWorker2(Worker):
        async def arun(
            self, x: int, 
            y: int = From("worker_0"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is my_graph
            assert sub_automa is my_graph_1
            return x + y  # 13

    my_graph = MyGraph()
    my_graph.add_worker(
        key="my_graph_1",
        worker=my_graph_1,
    )
    my_graph.add_func_as_worker(
        key="worker_4",
        func=worker_4,
        dependencies=["worker_3"],
        callback_builders=[WorkerCallbackBuilder(PrintCallback)],
    )
    my_graph.add_worker(
        key="worker_5",
        worker=MyWorker(),
        dependencies=["worker_4"],
        callback_builders=[WorkerCallbackBuilder(PrintCallback)],
    )
    my_graph.add_worker(
        key="worker_6",
        worker=MyWorker2(),
        dependencies=["worker_5"],
        is_output=True,
        callback_builders=[WorkerCallbackBuilder(PrintCallback)],
    )
    return my_graph

@pytest.mark.asyncio
async def test_args_inject_and_callback(args_inject_and_callback_graph: GraphAutoma, capsys):
    result = await args_inject_and_callback_graph.arun(user_input=Distribute([1, 2]))
    assert result == 13

    outputs = capsys.readouterr()
    assert "Pre callback for worker worker_2" in outputs.out
    assert "Post callback for worker worker_2" in outputs.out
    assert "Pre callback for worker worker_3" in outputs.out
    assert "Post callback for worker worker_3" in outputs.out
    assert "Pre callback for worker worker_4" in outputs.out
    assert "Post callback for worker worker_4" in outputs.out
    assert "Pre callback for worker worker_5" in outputs.out
    assert "Post callback for worker worker_5" in outputs.out
    assert "Pre callback for worker worker_6" in outputs.out
    assert "Post callback for worker worker_6" in outputs.out


# - - - - - - - - - - - - - - - -
# Test case: All kinds of ArgsMappingRule and From & System
# - - - - - - - - - - - - - - - -
@pytest.fixture
def automa_with_args_mapping_and_from():

    global my_graph_1
    my_graph_1 = GraphAutoma()

    class MyGraph(GraphAutoma):
        @worker(is_start=True)
        async def start(self, user_input: int) -> int:
            return user_input

        @worker(is_start=True)
        async def start_1(self, user_input: int) -> int:
            assert user_input == 2
            return user_input + 1

        @worker(dependencies=["start"])
        async def worker_00(self, x: int) -> int:
            return x + 1  # 2
        
        @worker(dependencies=["start"])
        async def worker_01(self, y: int) -> int:
            return y + 2  # 3

        @worker(dependencies=["worker_00", "worker_01"], args_mapping_rule=ArgsMappingRule.AS_IS)
        async def worker_10(
            self, x: int, 
            y: int, 
            z: int = From("start"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            return x + y + z  # 6

        @worker(dependencies=["worker_00", "worker_01"], args_mapping_rule=ArgsMappingRule.MERGE)
        async def worker_11(
            self, x: int, 
            z: int = From("start"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            return x[0] + x[1] + z  # 6

        @worker(dependencies=["worker_10", "worker_11"])
        async def worker_21(self, x: int, y: int) -> int:
            return {
                "x": x,
                "y": y,
                "z": 0,  # test from cover the default value of z
            }

        @worker(dependencies=["worker_21"], args_mapping_rule=ArgsMappingRule.UNPACK)
        async def worker_31(
            self, x: int, y: int, z: int = From("start"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            res = x + y + z  # 13
            if res == 13:
                return res
            raise ValueError("res is not 13")

        @worker(dependencies=["worker_31"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
        async def worker_41(
            self, z: int = From("start"), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            assert automa is self
            assert sub_automa is my_graph_1
            return z + 1, z + 2  # 2

        @worker(dependencies=["worker_41"], result_dispatch_rule=ResultDispatchRule.DISTRIBUTE)
        async def worker_51(
            self, x: Tuple[int, int], z: int = From("no_exist_worker", 1), 
            automa: GraphAutoma = System("automa"), 
            sub_automa: GraphAutoma = System("automa:my_graph_1"),
        ) -> int:
            x1, x2 = x
            assert x2 == 3
            assert automa is self
            assert sub_automa is my_graph_1
            return x1 + z, x1  # (3, 2)

        @worker(dependencies=["worker_51"])
        async def worker_61(self, x: int) -> int:
            return x  # 3

        @worker(dependencies=["worker_51"])
        async def worker_62(self, x: int) -> int:
            return x  # 2

        @worker(dependencies=["worker_61", "worker_62"], is_output=True)
        async def worker_71(self, x: int, y: int) -> int:
            return x + y  # 5

    my_graph = MyGraph()
    my_graph.add_worker(
        key="my_graph_1",
        worker=my_graph_1,
    )
    return my_graph

@pytest.mark.asyncio
async def test_automa_with_args_mapping_and_from(automa_with_args_mapping_and_from: GraphAutoma):
    result = await automa_with_args_mapping_and_from.arun(user_input=Distribute([1, 2]))
    assert result == 5


########################################################
#### Test case: serilization
########################################################

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

# - - - - - - - - - - - - - -
# Automa Serialization with From
# - - - - - - - - - - - - - -

class AutomaSerializationWithFrom(GraphAutoma):
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
    try:
        result = await automa_serialization_with_from.arun(x=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is x: 3, y: 2, do you want to add another 200 to them (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
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
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = AutomaSerializationWithFrom.load_from_snapshot(snapshot)
        assert type(deserialized_automa) is AutomaSerializationWithFrom
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
async def test_automa_deserialization_with_from(
    interaction_feedback_1_yes, 
    deserialized_automa_serialization_with_from: GraphAutoma
):
    result = await deserialized_automa_serialization_with_from.arun(
        feedback_data=interaction_feedback_1_yes
    )
    assert result == 405


########################################################
#### Test case: error 
########################################################

# - - - - - - - - - - - - - -
# From Error
# - - - - - - - - - - - - - -
@pytest.fixture
def automa_with_from_error_1():
    class AutomaFromError1(GraphAutoma):
        """
        test from with default value
        """
        @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
        async def worker_0(self, user_input: int) -> int:
            return user_input + 1
        
        @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
        async def worker_1(self, x: int) -> Tuple[int, int]:
            return x + 1

        @worker(dependencies=["worker_1"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
        async def worker_2(self, x: int, y: int = From("no_exist_worker")) -> int:
            return x + y

    return AutomaFromError1()

@pytest.mark.asyncio
async def test_automa_with_from_error_1(automa_with_from_error_1: GraphAutoma):
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=(
            f"the worker: `no_exist_worker` is not found in the automa or `no_exist_worker` is already removed. "
            "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
        )
    ):
        await automa_with_from_error_1.arun(user_input=1)


@pytest.fixture
def automa_with_from_error_2():
    """
    test From to get a non-exist worker's output.
    """
    class AutomaFromError2(GraphAutoma):
        @worker(is_start=True)
        async def worker_0(self, user_input) -> int:
            return user_input + 1

        @worker(is_start=True)
        async def worker_01(self, user_input: int) -> int:
            self.remove_worker("worker_01")
            return user_input + 1

        @worker(dependencies=["worker_0"], is_output=True)
        async def worker_02(self, x: int, y: int = From("worker_01")) -> int:
            return x + y  # 4
        
    return AutomaFromError2()

@pytest.mark.asyncio
async def test_automa_with_from_error_2(automa_with_from_error_2: GraphAutoma):
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=(
            f"the worker: `worker_01` is not found in the automa or `worker_01` is already removed. "
            "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
        )
    ):
        await automa_with_from_error_2.arun(user_input=1)


# - - - - - - - - - - - - - -
# System Error
# - - - - - - - - - - - - - -
@pytest.mark.asyncio
async def test_automa_with_system_error_1():
    """
    test system with default value, the keys are supported is limited.
    """
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=(
            f"Key 'automa-no_exist_automa' is not supported. Supported keys: \n"
            f"- `runtime_context`: a context for data persistence of the current worker.\n"
            f"- `automa:<worker_key>`: a sub-automa in current automa.\n"
            f"- `automa`: the current automa instance.\n"
        )
    ):
        class AutomaWithSystemError1(GraphAutoma):
            @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
            async def worker_0(self, user_input: int, automa: GraphAutoma = System("automa-no_exist_automa")) -> int:
                return user_input + 1

        await AutomaWithSystemError1().arun(user_input=1)


@pytest.fixture
def automa_with_system_error_2():
    """
    test system with sub-automa key, the key must be a valid worker key.
    """
    class AutomaWithSystemError2(GraphAutoma):
        @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
        async def worker_0(self, user_input: int, sub_automa: GraphAutoma = System("automa:no_exist_automa")) -> int:
            return user_input + 1

    return AutomaWithSystemError2()

@pytest.mark.asyncio
async def test_automa_with_system_error_2(automa_with_system_error_2: GraphAutoma):
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=(
            f"the sub-atoma: `automa:no_exist_automa` is not found in current automa. "
        )
    ):
        await automa_with_system_error_2.arun(user_input=1)


@pytest.fixture
def automa_with_system_error_3():
    """
    test system with sub-automa key, the key must be an Automa instance.
    """
    class AutomaWithSystemError3(GraphAutoma):
        @worker(is_start=True, args_mapping_rule=ArgsMappingRule.AS_IS)
        async def worker_0(self, user_input: int, sub_automa: GraphAutoma = System("automa:worker_0")) -> int:
            return user_input + 1

    return AutomaWithSystemError3()

@pytest.mark.asyncio
async def test_automa_with_system_error_3(automa_with_system_error_3: GraphAutoma):
    with pytest.raises(
        WorkerArgsInjectionError, 
        match=(
            f"the `automa:worker_0` instance is not an Automa. "
        )
    ):
        await automa_with_system_error_3.arun(user_input=1)