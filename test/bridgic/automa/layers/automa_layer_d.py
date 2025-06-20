from datetime import datetime

from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerD(Automa):
    @worker(is_start=True)
    def easy_start_worker(self, *args, **kwargs) -> str:
        self.ferry_to("continue_automa", *args, **kwargs)
