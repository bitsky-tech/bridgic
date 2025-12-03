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
class MyGraph1(ASLAutoma):
    with graph as g:
        a = worker1
        b = worker2


if __name__ == "__main__":
    graph_1 = MyGraph1()
    result_1 = asyncio.run(graph_1.arun(user_input=1))

    # try:
    #     graph_2 = MyGraph()
    #     result_2 = asyncio.run(graph_2.arun(user_input=1))
    #     print(result_2)
    # except Exception as e:
    #     print(e.interactions[0].event.event_type)

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
    