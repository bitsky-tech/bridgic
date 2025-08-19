from bridgic.automa import GraphAutoma
from bridgic.automa import worker

class AddAutoma(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        return x + 2