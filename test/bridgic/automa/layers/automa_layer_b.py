from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerB(Automa):
    @worker(is_start=True)
    def worker_4(self, *args, **kwargs):
        pass

    @worker(is_start=True)
    def worker_5(self, *args, **kwargs):
        pass

    @worker(dependencies=["worker_4", "worker_5"])
    def worker_6(self, *args, **kwargs):
        if type(self) != AutomaLayerB:
            self.ferry_to("entry_point_worker_8", *args, **kwargs)