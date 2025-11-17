from bridgic.core.automa._graph_automa import GraphAutoma
import pytest
from typing import List
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System, ArgsMappingRule, From
from bridgic.core.agentic.asl import graph, concurrent, ASLAutoma, Settings, Data, ASLField


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
    print(f"user_input: {user_input}")
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
        ]

async def tasks_done(task_input: int) -> int:
    return task_input + 3


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
@pytest.fixture
def asl_run_correctly_graph():
    class SubGraph(ASLAutoma):
        with graph as g:  # input: 5
            a = worker1  # 6
            b = worker2  # 8
            c = Worker3(y=1)  # 9
            d = produce_tasks  # [10, 11, 12, 13, 14]
            
            +a >> b >> c >> ~d

    class MyGraph(ASLAutoma):
        with graph as g:  # input: 1
            a = worker1   # 2
            b = worker2   # 4
            c = Worker3(y=1) # 5
            d = SubGraph() *Settings(args_mapping_rule=ArgsMappingRule.DISTRIBUTE)  # [10, 11, 12, 13, 14]

            arrangement_1 = +a >> b >> c  # 5
            arrangement_2 = arrangement_1 >> d  # [10, 11, 12, 13, 14]

            with graph as sub_graph_1:  # input: 10 -> res: (34, 35)
                a = worker1 # 11
                b = worker11 # 12
                c = worker12 # 11
                d = worker5 # 34
                e = worker6 # 35
                merge = merge # (34, 35)

                +(a & b & c) >> (d & e) >> ~merge # (34, 35)

            with graph as sub_graph_2:  # input: 11 -> res: 13
                ferry_to_worker = ferry_to_worker
                worker2 = worker2
                worker3 = Worker3(y=1)

                +ferry_to_worker, ~worker2, ~worker3  # 13

            with graph as sub_graph_3:  # input: 12 -> res: 16
                a = worker1 * Settings( 
                    is_start=True, 
                )  # 13
                b = worker2 * Settings(
                    dependencies=["a"], 
                    args_mapping_rule=ArgsMappingRule.AS_IS
                )  # 15
                c = Worker3(y=1) * Settings(
                    is_output=True, 
                    dependencies=["b"], 
                )  # 16
        
            with graph as sub_graph_4:  # input: 13 -> res: (16, 14)
                a = worker1  # 14
                b = worker2  # 16
                c = merge * Data(y=From("a"))

                +a >> b >> ~c

            with graph as sub_graph_5:  # input: 14 -> res: [18, 19, 20, 21, 22]
                a = produce_tasks  # [15, 16, 17, 18, 19]
                with concurrent(subtasks = ASLField(list, distribute=True)) as sub_concurrent:
                    dynamic_logic = lambda subtasks, **kwargs: (
                        tasks_done *Settings(
                            key=f"tasks_done_{i}"
                        )
                        for i, subtask in enumerate(subtasks)
                    )
                
                +a >> ~sub_concurrent  # [18, 19, 20, 21, 22]

            merger = merge_tasks *Settings(args_mapping_rule=ArgsMappingRule.MERGE)

            arrangement_2 >> (sub_graph_1 & sub_graph_2 & sub_graph_3 & sub_graph_4 & sub_graph_5) >> ~merger

    return MyGraph

@pytest.mark.asyncio
async def test_asl_run_correctly(asl_run_correctly_graph):
    graph = asl_run_correctly_graph()
    result = await graph.arun(user_input=1)
    assert result == [(34, 35), 13, 16, (16, 14), [18, 19, 20, 21, 22]]


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
            with graph as g:
                a = worker1 * Settings(
                    is_start=True, 
                )
                b = worker2 * Settings(
                    dependencies=["a"], 
                    args_mapping_rule=ArgsMappingRule.AS_IS
                )
                c = Worker3(y=1) * Settings(
                    is_output=True, 
                    dependencies=["b"], 
                )
                
                +a >> b >> ~c

    with pytest.raises(ValueError, match="Duplicate dependency"):
        class MyGraph2(ASLAutoma):
            with graph as g:
                a = worker1 * Settings(
                    dependencies=["worker_1", "worker_1"]
                )

    