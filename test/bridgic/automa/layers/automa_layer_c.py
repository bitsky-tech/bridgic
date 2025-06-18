import asyncio

from typing import Any, Dict, Callable

from bridgic.automa import Automa, worker
from bridgic.worker import CallableLandableWorker
from bridgic.utils.console import printer

from .automa_layer_a import AutomaLayerA
from .automa_layer_b import AutomaLayerB

class AutomaLayerC(AutomaLayerA, AutomaLayerB):
    def __init__(self, name: str = None, parallel_num: int = 2, workers: Dict[str, Callable] = {}):
        super().__init__(name=name, parallel_num=parallel_num, workers=workers)
        self.add_worker(
            name="entry_point_worker_7",
            func=lambda atm, *args, **kwargs: printer.print("  entry_point_worker_7:", "atm is Automa =>", isinstance(atm, Automa)),
            dependencies=[],
        )
        self.worker_8 = CallableLandableWorker(
            name="entry_point_worker_8",
            func=lambda atm, *args, **kwargs: printer.print("  entry_point_worker_8:", "atm is Automa =>", isinstance(atm, Automa)),
            dependencies=[],
        )