from datetime import datetime
from zoneinfo import ZoneInfo

from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.utils.console import printer

class AutomaLayerF(GraphAutoma):
    @worker(is_start=True)
    async def datetime_worker(self, time_zone: str = "Asia/Shanghai") -> str:
        printer.print("  datetime_worker", f"time_zone: {time_zone}")
        current_time = datetime.now()
        target_time = current_time.astimezone(ZoneInfo(time_zone))
        return target_time.strftime("%Y-%m-%d %H:%M:%S")

    @worker(dependencies=["datetime_worker"])
    async def transfer_datetime_worker(self, target_time: str):
        printer.print("  transfer_datetime_worker", self._worker_output["datetime_worker"])
        assert target_time == self._worker_output["datetime_worker"]
        return self._worker_output["datetime_worker"]
