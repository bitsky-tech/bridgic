import asyncio

from typing import Any, Dict

from bridgic.automa import Automa, worker
from bridgic.utils.console import printer

class AutomaLayerA(Automa):
    @worker(name="defined_start_worker_0", is_start=True, as_thread=True)
    def worker_0(self, *args, **kwargs) -> tuple[int, int]:
        return (1, 2)

    @worker(dependencies=["defined_start_worker_0"])
    def worker_1(self, *args, **kwargs) -> int:
        printer.print(f"  worker_1:", "self is Automa =>", isinstance(self, Automa))
        zero_output = self.defined_start_worker_0.output_buffer
        return zero_output[0]

    @worker(dependencies=["defined_start_worker_0"])
    def worker_2(self, *args, **kwargs) -> int:
        printer.print("  worker_2:", "self is Automa =>", isinstance(self, Automa))
        zero_output = self.defined_start_worker_0.output_buffer
        return zero_output[1]

    @worker(name="loop_worker_3", dependencies=["worker_1", "worker_2"])
    async def worker_3(self, *args, **kwargs) -> int:
        await asyncio.sleep(0.25)

        one_output: int = self.worker_1.output_buffer
        two_output: int = self.worker_2.output_buffer
        one_two_sum = one_output + two_output

        local_space: Dict[str, Any] = self.loop_worker_3.local_space
        local_space["cnt"] = local_space.get("cnt", 0) + 1
        printer.print("  loop_worker_3:", "local_space =>", local_space)

        if local_space["cnt"] < one_two_sum:
            self.ferry_to("defined_start_worker_0", *args, **kwargs)
        else:
            if type(self) != AutomaLayerA:
                self.ferry_to("entry_point_worker_7", *args, **kwargs)

        return 3