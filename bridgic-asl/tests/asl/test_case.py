import asyncio
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker
from bridgic.asl import ASLAutoma, graph, concurrent, Settings, Data, ASLField
from bridgic.core.automa.args import ResultDispatchingRule


def worker1(user_input: int = None):
    print(f"worker1: {user_input}")
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
        self.y = res
        return res

async def produce_tasks(user_input: int) -> list:
    return [
        user_input + 1,
        user_input + 2,
        user_input + 3,
        user_input + 4,
        user_input + 5,
    ]

async def add_one(x: list[int]) -> list[int]:
    return [item + 1 for item in x]


class SubGraph(ASLAutoma):
    with graph as g:  # input: 5
        a = worker1  # 6
        b = worker2  # 8
        c = Worker3(y=1)  # 9
        x = produce_tasks  # [10, 11, 12, 13, 14]
        
        +a >> b >> c >> ~x

class SubGraph2(ASLAutoma):
    with graph as g:
        a = worker1

        ~(+a)


class MyGraph(ASLAutoma):
    with graph as g:  # input: 1
        a = SubGraph2()   # 2
        b = worker2   # 4
        c = Worker3(y=1) # 5
        d = SubGraph()  # [10, 11, 12, 13, 14]

        with graph as sub_graph_1:
            a = add_one

            +(~a)

        arrangement_1 = +a >> b >> c
        arrangement_1 >> d >> ~sub_graph_1


if __name__ == "__main__":
    graph_1 = MyGraph()
    result_1 = asyncio.run(graph_1.arun(user_input=1))
    print(graph_1)
    print(result_1)


    graph_2 = MyGraph()
    result_2 = asyncio.run(graph_2.arun(user_input=1))
    print(graph_2)
    print(result_2)