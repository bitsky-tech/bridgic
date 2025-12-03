import asyncio
from typing import List

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.automa.worker import Worker
from bridgic.asl import ASLAutoma, graph, concurrent, Settings, Data, ASLField
from bridgic.core.automa.args import ResultDispatchingRule, System, ArgsMappingRule, From
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException


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


# - - - - - - - - - - - - - - -
# asl graphs
# - - - - - - - - - - - - - - -
class SubGraph(ASLAutoma):
    with graph as g:  # input: 5
        a = worker1  # 6
        b = worker2  # 8
        c = Worker3(y=1)  # 9
        +a >> b >> ~c

        # d = produce_tasks  # [10, 11, 12, 13, 14]
        # +a >> b >> c >> ~d


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
    with graph as main_graph:  # input: 1
        a = SubGraph2()   # 2
        b = worker2   # 4
        c = Worker3(y=1) # 5
        # d = SubGraph() *Settings(result_dispatching_rule=ResultDispatchingRule.IN_ORDER)  # [10, 11, 12, 13, 14]
        d = SubGraph()  # [10, 11, 12, 13, 14]

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
            a = produce_tasks  # [15, 16, 17, 18, 19]
            with concurrent(subtasks = ASLField(list, dispatching_rule=ResultDispatchingRule.IN_ORDER)) as sub_concurrent:
                dynamic_logic = lambda subtasks: (
                    tasks_done *Settings(key=f"tasks_done_{i}")
                    for i, subtask in enumerate(subtasks)
                )
            
            +a >> ~sub_concurrent  # [18, 19, 20, 21, 22]

        merger = merge_tasks *Settings(args_mapping_rule=ArgsMappingRule.MERGE)
        concurrent_merge = SubGraph3()

        arrangement_2 >> (sub_graph_1 & sub_graph_2 & sub_graph_3 & sub_graph_4 & sub_graph_5) >> ~merger
        # interact = interact_with_user
        # arrangement_2 >> sub_graph_1 >> ~interact


if __name__ == "__main__":
    graph_1 = MyGraph()
    result_1 = asyncio.run(graph_1.arun(user_input=1))
    print(result_1)


    graph_2 = MyGraph()
    result_2 = asyncio.run(graph_2.arun(user_input=1))
    print(result_2)

    # graph = ConcurrentAutoma()
    # graph.add_func_as_worker(
    #     key="worker1",
    #     func=worker1,
    #     is_start=True,
    #     is_output=True
    # )
    # graph.add_func_as_worker(
    #     key="worker2",
    #     func=interact_with_user
    # )
    # try:
    #     result = asyncio.run(graph.arun(res=1))
    # except InteractionException as e:
    #     __merger__ = graph._get_worker_instance("__merger__")
    #     print(__merger__.parent)
    #     print(e.interactions[0].event.event_type)
    #     print(e.interactions[0].event.data)
    