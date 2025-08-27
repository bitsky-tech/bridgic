import asyncio

from typing import Any, Dict

from bridgic.automa import GraphAutoma, worker, ArgsMappingRule
from bridgic.utils.console import printer

class AutomaLayerA(GraphAutoma):
    @worker(key="defined_start_worker_0", is_start=True)
    async def worker_0(self, greeting: str = "hi", loop_back: bool = False) -> tuple[int, int]:
        printer.print("  defined_start_worker_0:", "greeting =>", greeting, "loop_back =>", loop_back)
        assert (greeting != "hi" and loop_back) or (greeting == "hi" and not loop_back)
        return (1, 2)

    @worker(dependencies=["defined_start_worker_0"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def worker_1(self) -> int:
        printer.print(f"  worker_1:", "self is GraphAutoma =>", isinstance(self, GraphAutoma))
        zero_output = self.defined_start_worker_0.output_buffer
        return zero_output[0]

    @worker(dependencies=["defined_start_worker_0"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def worker_2(self, *args, **kwargs) -> int:
        printer.print("  worker_2:", "self is GraphAutoma =>", isinstance(self, GraphAutoma))
        zero_output = self.defined_start_worker_0.output_buffer
        return zero_output[1]

    @worker(key="loop_worker_3", dependencies=["worker_1", "worker_2"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def worker_3(self, *args, **kwargs) -> int:
        await asyncio.sleep(0.25)

        one_output: int = self.worker_1.output_buffer
        two_output: int = self.worker_2.output_buffer
        one_two_sum = one_output + two_output

        assert one_two_sum == 3

        local_space: Dict[str, Any] = self.loop_worker_3.local_space
        local_space["cnt"] = local_space.get("cnt", 0) + 1
        printer.print("  loop_worker_3:", "local_space =>", local_space)

        if local_space["cnt"] < one_two_sum:
            greetings = [None, "good morning", "good afternoon", "good evening"]
            self.ferry_to("defined_start_worker_0", greeting=greetings[local_space["cnt"]], loop_back=True, *args, **kwargs)
        else:
            if type(self) != AutomaLayerA:
                self.ferry_to("entry_point_worker_7", *args, **kwargs)

        return 3