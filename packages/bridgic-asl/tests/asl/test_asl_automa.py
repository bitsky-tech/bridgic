import pytest
from typing import List

from bridgic.core.automa import GraphAutoma, Snapshot, worker
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System, ArgsMappingRule, From, ResultDispatchingRule
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException
from bridgic.asl._error import ASLCompilationError
from bridgic.asl import graph, concurrent, ASLAutoma, Settings, Data, ASLField


####################################################################################################################################################
# shared fixtures
####################################################################################################################################################
# - - - - - - - - - - - - - - -
# worker functions
# - - - - - - - - - - - - - - -
def worker1(user_input: int = None, automa: GraphAutoma = System("automa")):
    res = user_input + 1
    return res

def worker11(user_input: int):
    res = user_input + 2
    return res

def worker12(user_input: int):
    res = user_input + 1
    return res

async def worker2(x: int):
    res =  x + 2
    return res

class Worker3(Worker):
    def __init__(self, y: int):
        super().__init__()
        self.y = y

    async def arun(self, x: int):
        res = self.y + x
        return res

class Worker31:
    async def worker31_func(self, x: int):
        res = x + 1
        return res

async def worker4(x: int, y: int):
    res = x + y
    return res

async def worker5(x: int, y: int, z: int):
    res = x + y + z
    return res

async def worker6(x: int, y: int, z: int):
    res = x + y + z + 1
    return res

async def merge(x: int, y: int):
    return x, y

async def ferry_to_worker(user_input: int, automa: GraphAutoma = System("automa")):
    if user_input == 11:
        automa.ferry_to("worker2", user_input)
    else:
        automa.ferry_to("worker3", user_input)

async def merge_tasks(tasks_res: List[int]):
    return tasks_res

async def produce_tasks(user_input: int) -> list:
        return [
            user_input + 1,
            user_input + 2,
            user_input + 3,
            user_input + 4,
            user_input + 5,
            user_input + 6,
        ]

async def tasks_done(task_input: int) -> int:
    return task_input + 3

async def tasks_done_2(task_input: int) -> int:
    return task_input


async def interact_with_user(res: List, automa: GraphAutoma = System("automa")) -> int:
    event = Event(
        event_type="if_add",
        data={
            "prompt_to_user": f"Current value is {res}, do you want to add another 200 to them (yes/no) ?"
        }
    )
    feedback: InteractionFeedback = automa.interact_with_human(event)
    if feedback.data == "yes":
        res = [item + 200 for item in res if isinstance(item, int)]
    return res


class MyGraphAutoma(GraphAutoma):
    @worker(is_start=True)
    async def worker_0(self, user_input: int) -> int:
        return user_input + 1

    @worker(dependencies=["worker_0"])
    async def worker_1(self, x: int) -> int:
        return x + 1

    @worker(dependencies=["worker_1"], is_output=True)
    async def worker_2(self, x: int) -> int:
        return x + 1


# - - - - - - - - - - - - - - -
# asl graphs
# - - - - - - - - - - - - - - -
class SubGraph(ASLAutoma):
    with graph as g:  # input: 5
        a = worker1  # 6
        b = worker2  # 8
        c = Worker3(y=1)  # 9
        d = produce_tasks  # [10, 11, 12, 13, 14, 15]
        
        +a >> b >> c >> ~d

class SubGraph2(ASLAutoma):
    with graph as g:
        a = worker1

        ~(+a)

class SubGraph3(ASLAutoma):
    with concurrent(subtasks = ASLField(list, dispatching_rule=ResultDispatchingRule.IN_ORDER)) as sub_concurrent:
        dynamic_logic = lambda subtasks: (
            tasks_done_2 *Settings(key=f"tasks_done_{i}")
            for i, subtask in enumerate(subtasks)
        )

class MyGraph(ASLAutoma):
    with graph as g:  # input: 1
        a = SubGraph2()   # 2
        b = worker2   # 4
        c = Worker3(y=1) # 5
        d = SubGraph() *Settings(result_dispatching_rule=ResultDispatchingRule.IN_ORDER)  # [10, 11, 12, 13, 14]

        arrangement_1 = +a >> b >> c  # 5
        arrangement_2 = arrangement_1 >> d  # [10, 11, 12, 13, 14]

        with graph as sub_graph_1:  # input: 10 -> res: (34, 35)
            a = worker1 # 11
            b = worker11 # 12
            c = worker12 # 11
            d = worker5 # 34
            e = worker6 # 35
            merge = merge # (34, 35)

            fragment_1 = +(a & b & c)
            fragment_2 = (d & e)
            fragment_1 >> fragment_2 >> ~merge  # (34, 35)

        with graph as sub_graph_2:  # input: 11 -> res: 13
            ferry_to_worker = ferry_to_worker
            worker2 = worker2
            worker3 = Worker3(y=1)

            +ferry_to_worker, ~worker2, ~worker3  # 13

        with graph as sub_graph_3:  # input: 12 -> res: 16
            a = worker1  # 13
            b = worker2 *Settings(args_mapping_rule=ArgsMappingRule.AS_IS)  # 15
            c = Worker3(y=1)  # 16

            +a >> b >> ~c  # 16

        with graph as sub_graph_4:  # input: 13 -> res: (16, 14)
            a = worker1  # 14
            b = worker2  # 16
            c = merge *Data(y=From("a"))

            +a >> b >> ~c

        with graph as sub_graph_5:  # input: 14 -> res: [18, 19, 20, 21, 22]
            a = produce_tasks  # [15, 16, 17, 18, 19, 20]
            with concurrent(subtasks = ASLField(list, dispatching_rule=ResultDispatchingRule.IN_ORDER)) as sub_concurrent:
                dynamic_logic = lambda subtasks: (
                    tasks_done *Settings(key=f"tasks_done_{i}")
                    for i, subtask in enumerate(subtasks)
                )
            
            +a >> ~sub_concurrent  # [18, 19, 20, 21, 22, 23]
        
        sub_graph_6 = MyGraphAutoma(name="sub_graph_6")  # input: 15 -> res: 18
        merger = merge_tasks *Settings(args_mapping_rule=ArgsMappingRule.MERGE)
        concurrent_merge = SubGraph3()

        arrangement_2 >> (sub_graph_1 & sub_graph_2 & sub_graph_3 & sub_graph_4 & sub_graph_5 & sub_graph_6) >> ~merger


class MyGraphInteract(MyGraph):
    with graph as g:
        flow = MyGraph()
        interact = interact_with_user
        +flow >> ~interact


####################################################################################################################################################
# test ASL-python code work correctly
####################################################################################################################################################

# - - - - - - - - - - - - - - -
# All features of ASL-python can run correctly
#   1. Kinds of worker can run correctly written by python-asl
#   2. The combination of arrangement logic
#   3. Component-based reuse
#   4. Group workers can run correctly
#   5. Nested graphs can run correctly
#   6. Ferry_to graph with no dependency
#   7. Settings can be set correctly
#   8. Args Injection can run correctly
#   9. Dynamic lambda worker can run correctly
# - - - - - - - - - - - - - - -
@pytest.mark.asyncio
async def test_asl_run_correctly():
    graph_1 = MyGraph()
    result_1 = await graph_1.arun(user_input=1)
    assert result_1 == [(34, 35), 13, 16, (16, 14), [18, 19, 20, 21, 22, 23], 18]

    graph_2 = MyGraph()
    result_2 = await graph_2.arun(user_input=1)
    assert result_2 == [(34, 35), 13, 16, (16, 14), [18, 19, 20, 21, 22, 23], 18]


# - - - - - - - - - - - - - - -
# test interact with user correctly
#   1. serialization correctly
#   2. deserialization and running correctly
# - - - - - - - - - - - - - - -

# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def db_base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")

@pytest.mark.asyncio
async def test_asl_interact_with_user_correctly_serialization(request, db_base_path):
    try:
        graph = MyGraphInteract()
        result = await graph.arun(user_input=1)
    except InteractionException as e:
        assert e.interactions[0].event.event_type == "if_add"
        assert e.interactions[0].event.data["prompt_to_user"] == "Current value is [(34, 35), 13, 16, (16, 14), [18, 19, 20, 21, 22, 23], 18], do you want to add another 200 to them (yes/no) ?"
        assert type(e.snapshot.serialized_bytes) is bytes
        # Send e.interactions to human. Here is a simulation.
        interaction_id = e.interactions[0].interaction_id
        request.config.cache.set("interaction_id", interaction_id)
        # Persist the snapshot to an external storage. Here is a simulation implementation using file storage.
        # Use the Automa instance / scenario name as the search index name.
        bytes_file = db_base_path / "asl_interact_with_user_correctly.bytes"
        version_file = db_base_path / "asl_interact_with_user_correctly.version"
        bytes_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)

@pytest.fixture
def deserialized_asl_interact_with_user_correctly(db_base_path):
        bytes_file = db_base_path / "asl_interact_with_user_correctly.bytes"
        version_file = db_base_path / "asl_interact_with_user_correctly.version"
        serialized_bytes = bytes_file.read_bytes()
        serialization_version = version_file.read_text()
        snapshot = Snapshot(
            serialized_bytes=serialized_bytes, 
            serialization_version=serialization_version
        )
        # Snapshot is restored.
        assert snapshot.serialization_version == GraphAutoma.SERIALIZATION_VERSION
        deserialized_automa = MyGraphInteract.load_from_snapshot(snapshot)
        # assert type(deserialized_automa) is MyGraphInteract
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
async def test_asl_interact_with_user_correctly_deserialization(
    interaction_feedback_1_yes, 
    deserialized_asl_interact_with_user_correctly: GraphAutoma
):
    result = await deserialized_asl_interact_with_user_correctly.arun(
        feedback_data=interaction_feedback_1_yes
    )
    assert result == [213, 216, 218]
    


####################################################################################################################################################
# test ASL-python code raise error correctly
####################################################################################################################################################

# - - - - - - - - - - - - - - -
# test worker declaration must be under graph
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_asl_worker_declaration_must_under_graph():
    with pytest.raises(ASLCompilationError, match=(
        f"All workers must be written under one graph."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                b = worker2
            c = Worker3(y=1)


# - - - - - - - - - - - - - - -
# test worker declaration must have a unique key
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_asl_worker_declaration_must_have_a_unique_key():
    with pytest.raises(ASLCompilationError, match=(
        f"Duplicate key: a under graph: g of a fragment or a registered worker."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                a = worker2

    with pytest.raises(ASLCompilationError, match=(
        f"Duplicate key: a under graph: g of a fragment or a registered worker."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                b = worker2
                a = +a >> ~b

    with pytest.raises(ASLCompilationError, match=(
        f"Duplicate key: a under graph: g of a fragment or a registered worker."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                with graph as a:
                    b = worker2


# - - - - - - - - - - - - - - -
# test can not use canvas itself as a worker
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_asl_worker_declaration_must_under_graph():
    with pytest.raises(ASLCompilationError, match=(
        f"Invalid worker key: g, cannot use the canvas itself as a worker."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                g = worker2


    with pytest.raises(ASLCompilationError, match=(
        f"Invalid worker key: g, cannot use the canvas itself as a worker."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                +a >> ~g

# - - - - - - - - - - - - - - -
# test parent worker must have a key
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_asl_parent_worker_must_have_a_key():
    with pytest.raises(ASLCompilationError, match=(
        f"The parent of worker a has no name! Please declare the parent key with `with graph as <name>:`."
    )):
        class MyGraph(ASLAutoma):
            with graph:
                a = worker1
                b = worker2
                c = Worker3(y=1)
                +a >> ~b >> ~c


# - - - - - - - - - - - - - - -
# test can not have multiple root graph
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_asl_can_not_have_multiple_root_graph():
    with pytest.raises(ASLCompilationError, match=(
        f"Multiple root graph are not allowed."
    )):
        class MyGraph2(ASLAutoma):
            with graph as g:
                a = worker1
                b = worker2
                c = Worker3(y=1)

            with graph as g1:
                a = worker1
                b = worker2
                c = Worker3(y=1)


# - - - - - - - - - - - - - - -
# test lambda dynamic logic must be written under a concurrent or sequential graph
# - - - - - - - - - - - - - - -  
@pytest.mark.asyncio
async def test_asl_lambda_dynamic_logic_declaration_correctly():
    with pytest.raises(ASLCompilationError, match=(
        f"Lambda dynamic logic must be written under a `concurrent`, `sequential`, not `graph`."
    )):
        class MyGraph(ASLAutoma):
            with graph as g:
                a = worker1
                b = worker2
                c = Worker3(y=1)
                dynamic_logic = lambda: worker1
                +a >> b >> c >> ~dynamic_logic
    

# - - - - - - - - - - - - - - -
# test graph params declaration must be ASLField
# - - - - - - - - - - - - - - -  
@pytest.mark.asyncio
async def test_asl_graph_params_declaration_must_be_ASLField():
    with pytest.raises(ASLCompilationError, match=(
        f"Invalid field type: <class 'int'>."
    )):
        class MyGraph(ASLAutoma):
            with graph(x=1) as g:
                a = worker1
                b = worker2 *Settings(args_mapping_rule=ArgsMappingRule.AS_IS)
                c = Worker3(y=1)
                +a >> b >> c

    