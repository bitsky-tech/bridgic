from pydantic import BaseModel
from dataclasses import dataclass
import asyncio
from bridgic.asl.component import component, graph, concurrent, Data
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System


def worker1(x: int):
    res = x + 1
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

async def worker4(x, y, z):
    res = x + y + z
    print(f'==> worker4: {res}')
    return res

async def worker5(x, y, z):
    res = x + y + z + 1
    print(f'==> worker5: {res}')
    return res

async def merge(x, y):
    print(f'result: {x}, {y}')
    return x, y


# 最常用的形式，不使用设置和参数，使用 bridigc 框架的默认行为
@component
class MyGraph:
    x: int = None

    with graph() as g:
        a = worker1 @ g

        with graph() @ g as g2:
            d = worker1 @ g2
            e = worker2 @ g2
            +a >> d >> ~e

        b = worker2 @ g
        c = Worker3(y=1) @ g

        +a >> g2 >> b >> ~c

async def get_res():
    graph = MyGraph()
    result = await graph.arun(x=1)
    print(result)


if __name__ == "__main__":
    asyncio.run(get_res())
