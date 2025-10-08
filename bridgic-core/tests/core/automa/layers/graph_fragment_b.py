from bridgic.core.automa import GraphFragment, worker
from bridgic.core.utils.console import printer

class GraphFragmentB(GraphFragment):
    @worker(is_start=True)
    async def worker_4(self, *args, **kwargs):
        pass

    @worker(is_start=True)
    async def worker_5(self, *args, **kwargs):
        pass

    @worker(dependencies=["worker_4", "worker_5"])
    async def worker_6(self, *args, **kwargs):
        if "entry_point_worker_8" in self._workers.keys():
            self.ferry_to("entry_point_worker_8", *args, **kwargs)