from typing import Any, Dict

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.utils.console import printer

from .automa_layer_a import AutomaLayerA
from .automa_layer_b import AutomaLayerB

class PrintWorker(Worker):
    async def arun(self, *args, **kwargs) -> None:
        assert isinstance(self.parent, GraphAutoma)
        printer.print(
            f"  parent_automa is GraphAutoma =>",
            isinstance(self.parent, GraphAutoma),
        )

class EndWorker(Worker):
    async def arun(self, time_str: str) -> str:
        assert isinstance(time_str, str)
        return "happy_ending"

class AutomaLayerC(AutomaLayerA, AutomaLayerB):

    def __init__(
        self,
        output_worker_key: str = None,
    ):
        super().__init__(
            output_worker_key=output_worker_key,
        )
        self.add_func_as_worker(
            key="entry_point_worker_7",
            func=lambda *args, **kwargs: 7,
            dependencies=[],
        )
        self.add_worker(
            key="entry_point_worker_8",
            worker=PrintWorker(),
            dependencies=[],
        )
