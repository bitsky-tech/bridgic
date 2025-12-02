import asyncio
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker
from bridgic.asl import ASLAutoma, graph, concurrent, Settings, Data, ASLField


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


class SubGraph(ASLAutoma):
    with graph as g:
        a = worker1  # 5
        b = worker2  # 7
        c = Worker3(y=1)  # 8

        +a >> b >> ~c


class MyGraph(ASLAutoma):
    with graph as g:
        a = worker1  # 2
        b = worker2  # 4
        c = Worker3(y=1)
        d = SubGraph()  # 8

        # +a >> b >> c >> ~d
        arrangement_1 = +a >> b >> c
        # arrangement_2 = c >> ~d
        # arrangement_1 >> arrangement_2
        arrangement_1 >> ~d


if __name__ == "__main__":
    graph_1 = MyGraph()
    result_1 = asyncio.run(graph_1.arun(user_input=1))
    print(graph_1)
    print(result_1)


    graph_2 = MyGraph()
    result_2 = asyncio.run(graph_2.arun(user_input=1))
    print(graph_2)
    print(result_2)