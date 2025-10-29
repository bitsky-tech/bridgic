from typing import overload
from bridgic.core.automa._graph_automa import GraphAutoma
import pytest

from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System
from bridgic.asl.component import component, graph, concurrent


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


####################################################################################################################################################
# test ASL-python code work correctly
####################################################################################################################################################

# - - - - - - - - - - - - - - -
# test kinds of worker can run correctly written by python-asl
# - - - - - - - - - - - - - - -
@pytest.fixture
def kinds_of_worker_graph():
    @component
    class MyGraph:
        user_input: int = None

        with graph() as g:
            a = worker1 @ g
            b = worker2 @ g
            c = Worker3(y=1) @ g

            +a >> b >> ~c

    return MyGraph

@pytest.mark.asyncio
async def test_kinds_of_worker_can_run_correctly(kinds_of_worker_graph):
    graph = kinds_of_worker_graph()
    result = await graph.arun(user_input=1)
    assert result == 5


# - - - - - - - - - - - - - - -
# test Component-based reuse
# - - - - - - - - - - - - - - -
@pytest.fixture
def component_based_reuse_graph():
    @component
    class MyGraph1:
        user_input: int = None

        with graph() as g:
            a = worker1 @ g
            b = worker2 @ g
            c = Worker3(y=1) @ g

            +a >> b >> ~c

    @component
    class MyGraph2:
        user_input: int = None

        with graph() as g:
            a = worker1 @ g
            b = MyGraph1() @ g
            c = Worker3(y=1) @ g

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
    @component
    class MyGraph1:
        user_input: int = None

        with graph() as g:
            a = worker1 @ g
            b = worker2 @ g
            c = Worker3(y=1) @ g
            d = worker4 @ g

            +a >> (b & c) >> ~d

    return MyGraph1

@pytest.mark.asyncio
async def test_group_workers_can_run_correctly(group_workers_can_run_correctly_graph):
    graph = group_workers_can_run_correctly_graph()
    result = await graph.arun(user_input=1)
    assert result == 7


@pytest.fixture
def groups_workers_can_run_correctly_graph():
    @component
    class MyGraph1:
        x: int = None

        with graph() as g:
            a = worker1 @ g
            b = worker11 @ g
            c = worker12 @ g
            d = worker5 @ g
            e = worker6 @ g
            merge = merge @ g

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
    @component
    class MyGraph1:
        user_input: int = None

        with graph() as g1:
            a = worker1 @ g1
            with graph() @ g1 as g2:
                c = worker2 @ g2
                d = Worker3(y=1) @ g2
                +c >> ~d
            b = worker2 @ g1

            +a >> g2 >> ~b

    return MyGraph1

@pytest.mark.asyncio
async def test_nested_graphs_can_run_correctly(nested_graphs_can_run_correctly_graph):
    graph = nested_graphs_can_run_correctly_graph()
    print(graph)
    result = await graph.arun(user_input=1)
    assert result == 7


# - - - - - - - - - - - - - - -
# test ferry_to graph with no dependency
# - - - - - - - - - - - - - - - 
@pytest.fixture
def ferry_to_with_no_dependency_graph():
    @component
    class MyGraph1:
        user_input: int = None

        with graph() as g1:
            ferry_to_worker = ferry_to_worker @ g1
            worker2 = worker2 @ g1
            worker3 = Worker3(y=1) @ g1

            +ferry_to_worker, ~worker2, ~worker3

    return MyGraph1

@pytest.mark.asyncio
async def test_no_dependency_can_run_correctly(ferry_to_with_no_dependency_graph):
    graph = ferry_to_with_no_dependency_graph()
    print(graph)
    result = await graph.arun(user_input=1)
    assert result == 3

    print(graph)
    result = await graph.arun(user_input=2)
    assert result == 3
