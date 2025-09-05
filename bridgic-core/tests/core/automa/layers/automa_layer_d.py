import random

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.utils.console import printer

class AutomaLayerD(GraphAutoma):
    @worker(is_start=True)
    async def inner_start_worker(self, *args, **kwargs) -> str:
        time_zone_list = ["Asia/Shanghai", "Asia/Hong_Kong", "Asia/Tokyo", "America/New_York", "Europe/London"]
        self.ferry_to("continue_automa", time_zone=random.choice(time_zone_list))
