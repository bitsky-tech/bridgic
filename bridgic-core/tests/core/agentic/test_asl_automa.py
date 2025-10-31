from bridgic.core.automa._graph_automa import GraphAutoma
import pytest
from typing import List
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System, ArgsMappingRule, From
from bridgic.core.agentic.asl import graph, ASLAutoma, Settings, Data


def worker1(user_input: int = None):
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
    if user_input == 1:
        automa.ferry_to("worker2", user_input)
    else:
        automa.ferry_to("worker3", user_input)

async def produce_tasks(x: int):
    res = [i for i in range(x)]
    return res


async def tasks_done(x = None, y = None, z = None):
    if x:
        return x + 1
    elif y:
        return y + 1
    elif z:
        return z + 1


async def merge_tasks(tasks_res: List[int]):
    return tasks_res


####################################################################################################################################################
# test ASL-python code work correctly
####################################################################################################################################################

# - - - - - - - - - - - - - - -
# test kinds of worker can run correctly written by python-asl
# - - - - - - - - - - - - - - -
@pytest.fixture
def kinds_of_worker_graph():
    class MyGraph(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = worker2
            c = Worker3(y=1)

            +a >> b >> ~c

    return MyGraph

@pytest.mark.asyncio
async def test_kinds_of_worker_can_run_correctly(kinds_of_worker_graph):
    graph = kinds_of_worker_graph()
    result = await graph.arun(user_input=1)
    assert result == 5


# - - - - - - - - - - - - - - -
# test the combination of arrangement logic
# - - - - - - - - - - - - - - -
@pytest.fixture
def combine_of_arrangement_logic_graph():
    class MyGraph(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = worker2
            c = Worker3(y=1)

            arrangement_1 = +a >> b
            arrangement_2 = ~c

            arrangement_1 >> arrangement_2

    return MyGraph

@pytest.mark.asyncio
async def test_combine_of_arrangement_logic_can_run_correctly(combine_of_arrangement_logic_graph):
    graph = combine_of_arrangement_logic_graph()
    result = await graph.arun(user_input=1)
    assert result == 5


# - - - - - - - - - - - - - - -
# test Component-based reuse
# - - - - - - - - - - - - - - -
@pytest.fixture
def component_based_reuse_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = worker2
            c = Worker3(y=1)

            +a >> b >> ~c

    class MyGraph2(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = MyGraph1()
            c = Worker3(y=1)

            +a >> b >> ~c

    return MyGraph2

@pytest.mark.asyncio
async def test_component_based_reuse(component_based_reuse_graph):
    graph = component_based_reuse_graph()
    result = await graph.arun(user_input=1)
    assert result == 7


# - - - - - - - - - - - - - - -
# test group workers can run correctly
# - - - - - - - - - - - - - - -
@pytest.fixture
def group_workers_can_run_correctly_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = worker2
            c = Worker3(y=1)
            d = worker4

            +a >> (b & c) >> ~d

    return MyGraph1

@pytest.mark.asyncio
async def test_group_workers_can_run_correctly(group_workers_can_run_correctly_graph):
    graph = group_workers_can_run_correctly_graph()
    result = await graph.arun(user_input=1)
    assert result == 7


@pytest.fixture
def groups_workers_can_run_correctly_graph():
    class MyGraph1(ASLAutoma):
        x: int = None

        with graph() as g:
            a = worker1
            b = worker11
            c = worker12
            d = worker5
            e = worker6
            merge = merge

            +(a & b & c) >> (d & e) >> ~merge

    return MyGraph1

@pytest.mark.asyncio
async def test_groups_workers_can_run_correctly(groups_workers_can_run_correctly_graph):
    graph = groups_workers_can_run_correctly_graph()
    result = await graph.arun(user_input=1)
    assert result == (7, 8)


# - - - - - - - - - - - - - - -
# test nested graphs can run correctly
# - - - - - - - - - - - - - - -
@pytest.fixture
def nested_graphs_can_run_correctly_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g1:
            a = worker1 
            with graph() as g2:
                c = worker2
                d = Worker3(y=1)
                +c >> ~d
            b = worker2

            +a >> g2 >> ~b

    return MyGraph1

@pytest.mark.asyncio
async def test_nested_graphs_can_run_correctly(nested_graphs_can_run_correctly_graph):
    graph = nested_graphs_can_run_correctly_graph()
    result = await graph.arun(user_input=1)
    assert result == 7


# - - - - - - - - - - - - - - -
# test ferry_to graph with no dependency
# - - - - - - - - - - - - - - - 
@pytest.fixture
def ferry_to_with_no_dependency_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g1:
            ferry_to_worker = ferry_to_worker
            worker2 = worker2
            worker3 = Worker3(y=1)

            +ferry_to_worker, ~worker2, ~worker3

    return MyGraph1

@pytest.mark.asyncio
async def test_no_dependency_can_run_correctly(ferry_to_with_no_dependency_graph):
    graph = ferry_to_with_no_dependency_graph()
    result = await graph.arun(user_input=1)
    assert result == 3

    result = await graph.arun(user_input=2)
    assert result == 3


# - - - - - - - - - - - - - - -
# test Settings can be set correctly
# - - - - - - - - - - - - - - - 
@pytest.fixture
def settings_can_be_set_correctly_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1 * Settings(
                key="worker_1", 
                is_start=True, 
            )
            b = worker2 * Settings(
                key="worker_2", 
                dependencies=["worker_1"], 
                args_mapping_rule=ArgsMappingRule.AS_IS
            )
            c = Worker3(y=1) * Settings(
                key="worker_3", 
                is_output=True, 
                dependencies=["worker_2"], 
            )

    return MyGraph1

@pytest.mark.asyncio
async def test_settings_can_be_set_correctly(settings_can_be_set_correctly_graph):
    graph = settings_can_be_set_correctly_graph()
    print(graph)
    result = await graph.arun(user_input=1)
    assert result == 5


# - - - - - - - - - - - - - - -
# test Args Injection can run correctly
# - - - - - - - - - - - - - - - 
@pytest.fixture
def args_injection_can_run_correctly_graph():
    class MyGraph1(ASLAutoma):
        user_input: int = None

        with graph() as g:
            a = worker1
            b = worker2
            c = merge * Data(y=From("a"))

            +a >> b >> ~c

    return MyGraph1

@pytest.mark.asyncio
async def test_args_injection_can_run_correctly(args_injection_can_run_correctly_graph):
    graph = args_injection_can_run_correctly_graph()
    print(graph)
    result = await graph.arun(user_input=1)
    assert result == (4, 2)


# - - - - - - - - - - - - - - -
# test dynamic lambda worker can run correctly
# - - - - - - - - - - - - - - - 
@pytest.fixture
def dynamic_lambda_worker_can_run_correctly_graph():
    class MyGraph1(ASLAutoma):
        x: int = None

        with graph() as g:
            a = produce_tasks

            dynamic_logic = lambda subtasks, **kwargs: (
                tasks_done *Settings(
                    key=f"tasks_done_{i}"
                )
                for i, subtask in enumerate(subtasks)
            )
            
            # with concurrent() as g2:
                
            
            b = merge_tasks

            +a >> dynamic_logic >> ~b

    return MyGraph1

@pytest.mark.asyncio
async def test_dynamic_lambda_worker_can_run_correctly(dynamic_lambda_worker_can_run_correctly_graph):
    graph = dynamic_lambda_worker_can_run_correctly_graph()
    result = await graph.arun(x=3)
    assert result == [0, 1, 2]


####################################################################################################################################################
# test ASL-python code raise error correctly
####################################################################################################################################################

# - - - - - - - - - - - - - - -
# test raise duplicate dependency error correctly
# - - - - - - - - - - - - - - - 
@pytest.mark.asyncio
async def test_raise_duplicate_dependency_error_correctly():
    with pytest.raises(ValueError, match="Duplicate dependency"):
        class MyGraph1(ASLAutoma):
            user_input: int = None

            with graph() as g:
                a = worker1 * Settings(
                    key="worker_1", 
                    is_start=True, 
                )
                b = worker2 * Settings(
                    key="worker_2", 
                    dependencies=["worker_1"], 
                    args_mapping_rule=ArgsMappingRule.AS_IS
                )
                c = Worker3(y=1) * Settings(
                    key="worker_3", 
                    is_output=True, 
                    dependencies=["worker_2"], 
                )
                
                +a >> b >> ~c

    with pytest.raises(ValueError, match="Duplicate dependency"):
        class MyGraph2(ASLAutoma):
            user_input: int = None

            with graph() as g:
                a = worker1 * Settings(
                    dependencies=["worker_1", "worker_1"]
                )

    