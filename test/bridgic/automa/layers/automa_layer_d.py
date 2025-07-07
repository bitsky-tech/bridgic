import random

from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerD(Automa):
    @worker(is_start=True)
    def easy_start_worker(self, *args, **kwargs) -> str:
        time_zone_list = ["Asia/Shanghai", "Asia/Hong_Kong", "Asia/Tokyo", "America/New_York", "Europe/London"]
        self.ferry_to("continue_automa", time_zone=random.choice(time_zone_list), *args, **kwargs)
