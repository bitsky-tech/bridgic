import asyncio

from bridgic.core.automa import GraphFragment, worker
from bridgic.core.automa.arguments_descriptor import System
from bridgic.core.utils.console import printer

class GraphFragmentA(GraphFragment):
    @worker(key="defined_start_worker_0", is_start=True)
    async def worker_0(self, greeting: str = "hi") -> tuple[int, int]:
        printer.print("  defined_start_worker_0:", "greeting =>", greeting)
        return (1, 2)

    @worker(dependencies=["defined_start_worker_0"])
    async def worker_1(self, zero_output: tuple[int, int]) -> int:
        return zero_output[0]

    @worker(dependencies=["defined_start_worker_0"])
    async def worker_2(self, zero_output: tuple[int, int]) -> int:
        return zero_output[1]

    @worker(key="loop_worker_3", dependencies=["worker_1", "worker_2"])
    async def worker_3(self, one_output: int, two_output: int, rtx = System("runtime_context"), *args, **kwargs) -> int:
        await asyncio.sleep(0.25)
        one_two_sum = one_output + two_output

        assert one_two_sum == 3
        local_space = self.get_local_space(runtime_context=rtx)
        local_space["cnt"] = local_space.get("cnt", 0) + 1
        cnt = local_space["cnt"]
        printer.print("  loop_worker_3:", "cnt =>", cnt)

        if cnt <= one_two_sum:
            greetings = [None, "good morning", "good afternoon", "good evening"]
            self.ferry_to("defined_start_worker_0", greeting=greetings[cnt], *args, **kwargs)
        else:
            if "entry_point_worker_7" in self._workers.keys():
                self.ferry_to("entry_point_worker_7", *args, **kwargs)

        return 3