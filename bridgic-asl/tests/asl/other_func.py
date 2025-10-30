import asyncio

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.args import From


class MyGraph(GraphAutoma):
    @worker(is_start=True)
    async def worker1(self, x: int) -> int:
        return x + 1

    @worker(dependencies=["worker1"])
    async def worker2(self, y: int) -> int:
        return y + 2

    @worker(dependencies=["worker2"], is_output=True)
    async def worker3(self, z: int = From("worker1"), w: int = From("worker2")) -> int:
        return (z, w)


if __name__ == "__main__":
    res = asyncio.run(MyGraph().arun(x=1))
    print(res)
