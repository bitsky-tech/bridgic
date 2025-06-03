from bridgic.automa import AutoMa
import asyncio
from bridgic.automa.bridge.decorator import processor

# 这个例子展示如何通过“decorated-based”的编排模式，实现最简单的流程。
# 还是实现：输入x，输出 3x+5，这样的功能

class SimpleFlow(AutoMa):

    def __init__(self):
        super().__init__()

    @processor(is_start=True)
    def multiply_3(self, x: int) -> int:
        return x * 3

    @processor(is_end=True, listen=multiply_3)
    def add_5(self, x: int) -> int:
        self.post_out_event(OutEvent(event_id="add_5", message="ee"))
        return x + 5

def simple_flow_hook(event: OutEvent, future: InEventFuture) -> None:
    event

async def main():
    flow = SimpleFlow()
    flow.register_conduit_hook(simple_flow_hook)
    result = await flow.process(x=7)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())