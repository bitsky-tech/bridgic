import asyncio
from pydoc import describe
from pydantic import Field
from typing import Annotated

from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import From
from bridgic.core.agentic.asl import graph, concurrent, Data, Settings, ASLAutoma, ASLField


async def worker1(user_input: int) -> int:
    return user_input + 1

async def worker2(x: int) -> int:
    return x + 2

class Worker3(Worker):
    def __init__(self, y: int):
        super().__init__()
        self.y = y

    async def arun(self, x: int) -> int:
        return x + self.y


async def common_worker(task_input: int) -> int:
    return task_input + 3


class MyGraph(ASLAutoma):
    with graph as g:
        a = worker1
        b = worker2
        c = Worker3(y=1)

        +a >> b >> ~c


class MyGraph1(ASLAutoma):
    user_input: int = None

    with graph as g1:
        a = worker1 
        with graph as g2:
            c = worker2
            d = Worker3(y=1)
            +c >> ~d
        b = worker2

        +a >> g2 >> ~b


if __name__ == "__main__":
    my_graph = MyGraph()
    result = asyncio.run(my_graph.arun(user_input=1))
    print(result)

    my_graph = MyGraph1()
    result = asyncio.run(my_graph.arun(user_input=1))
    print(result)
    
    pass
