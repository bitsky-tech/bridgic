from typing import Any, Dict

from bridgic.automa import Automa
from bridgic.automa.worker import Worker
from bridgic.utils.console import printer

from .automa_layer_a import AutomaLayerA
from .automa_layer_b import AutomaLayerB

class PrintWorker(Worker):
    async def process_async(self, *args, **kwargs) -> None:
        assert isinstance(self.parent_automa, Automa)
        printer.print(
            f"  {self.name}: parent_automa is Automa =>",
            isinstance(self.parent_automa, Automa),
        )

class EndWorker(Worker):
    async def process_async(self, *args, **kwargs) -> str:
        return "happy_ending"

class AutomaLayerC(AutomaLayerA, AutomaLayerB):

    def __init__(
        self,
        name: str = None,
        parallel_num: int = 2,
        output_worker_name: str = None,
        workers: Dict[str, Worker] = {},
    ):
        super().__init__(
            name=name,
            parallel_num=parallel_num,
            output_worker_name=output_worker_name,
            workers=workers,
        )
        self.add_func_as_worker(
            name="entry_point_worker_7",
            func=lambda *args, **kwargs: 7,
            dependencies=[],
        )
        self.add_worker(
            PrintWorker(name="entry_point_worker_8"),
            dependencies=[],
        )
