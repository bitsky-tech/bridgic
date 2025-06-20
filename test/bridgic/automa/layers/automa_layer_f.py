from datetime import datetime

from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerF(Automa):
    @worker(is_start=True)
    def datetime_worker(self, *args, **kwargs) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @worker(dependencies=["datetime_worker"])
    def transfer_datetime_worker(self, *args, **kwargs):
        printer.print("transfer_datetime_worker", self.datetime_worker.output_buffer)
        return self.datetime_worker.output_buffer
