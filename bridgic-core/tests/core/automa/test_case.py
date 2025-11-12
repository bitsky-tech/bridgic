import asyncio
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.automa.worker import Worker
from typing import Tuple, Any

class MyGraph(GraphAutoma):
    @worker(is_start=True)
    async def start_1(self, x):
        return x + 1

    @worker(dependencies=["start_1"], is_output=True)
    async def start_2(self, x):
        return x + 2


if __name__ == "__main__":
    automa = MyGraph()
    result = asyncio.run(automa.arun(x=1))
    print(result)