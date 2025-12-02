import asyncio
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker
from bridgic.asl import ASLAutoma, graph, concurrent, Settings, Data, ASLField


def worker1(user_input: int = None):
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
        x = Worker3(y=1)  # 8

        +a >> b >> ~x


class MyGraph(ASLAutoma):
    with graph as g:
        a = worker1  # 2
        b = worker2  # 4
        # c = Worker3(y=1)
        c = SubGraph()  # 8

        # +a >> b >> c >> ~d
        +a >> b >> ~c


if __name__ == "__main__":
    graph_1 = MyGraph()
    result_1 = asyncio.run(graph_1.arun(user_input=1))
    print(result_1)


    graph_2 = MyGraph()
    result_2 = asyncio.run(graph_2.arun(user_input=1))
    print(result_2)

    # print(graph_1)
    # print('=======================')
    # print(graph_2)

    graph_3 = GraphAutoma()
    graph_3.add_func_as_worker(
        key="worker_1",
        func=worker1,
        is_start=True
    )
    graph_3.add_func_as_worker(
        key="worker_2",
        func=worker2,
        dependencies=["worker_1"]
    )
    graph_3.add_worker(
        key="worker_3",
        worker=Worker3(y=1),
        dependencies=["worker_2"],
        is_output=True
    )
    result_3 = asyncio.run(graph_3.arun(user_input=1))
    result_3_2 = asyncio.run(graph_3.arun(user_input=1))
    print(result_3)
    print(result_3_2)