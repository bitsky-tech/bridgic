from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerWrong(Automa):
    @worker(is_start=True, dependencies=["worker_3"])
    def worker_0(self, *args, **kwargs) -> int:
        return 0

    @worker(dependencies=["worker_0"])
    def worker_1(self, *args, **kwargs) -> int:
        return 1

    @worker(dependencies=["worker_0"])
    def worker_2(self, *args, **kwargs) -> int:
        return 2

    @worker(dependencies=["worker_1", "worker_2"])
    async def worker_3(self, *args, **kwargs) -> int:
        return 3