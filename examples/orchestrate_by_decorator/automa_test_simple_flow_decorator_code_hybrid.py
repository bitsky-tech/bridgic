from bridgic.automa import AutoMa
import asyncio
from bridgic.automa.bridge.decorator import processor, router
from bridgic.typing.event import OutEvent, InEvent
from typing import Future

# 这个例子展示如何通过“decorated-based”的编排模式，实现一个中间夹杂动态判断逻辑的流程。
# 最终实现的逻辑是：
# 输入x，先计算 3x+5的值；
# 然后，如果3x+5的值大于20，则最终输出3x+5的平方；否则，最终输出3x+5的立方。

class ExampleState(BaseModel):
    # Note: 'id' field is automatically added to all states
    counter: int = 0
    message: str = ""

def simple_flow_hook(out_event: OutEvent, in_event_emiter: InEventEmiter) -> None:
    pass


# TODO: 还没有调试通过，这个例子只是接口定义的展示。
class SimpleFlow(AutoMa[ExampleState]):

    def __init__(self):
        super().__init__()
        self.event_emiter = InEventEmiter()

    @processor(is_start=True)
    def multiply_3(self, x: int) -> int:
        return {"x": 4， “y”: "sss", "state": exampleState}

    @processor(listen=multiply_3)
    def add_5(self, x: int, ,y : str) -> int:
        y = context.set_variable("y", y)
        return x + 5
    
    @router(listen=and_(add_5, multiply_3))
    def dynamic_branch(self, x: int) -> None:
        input( "please input the value of x:")
        if x > 20:
            self.square(x)
        elif x < 10:
            self.add_10(x)
        else:
            self.cube(x)

    @processor(is_end=True,  context: XXXContext[ExampleState])
    def square(self, x: int) -> int:
        self.post_out_event(OutEvent(event_id="square", message="ee"))
        return x * x
    
    @processor(is_end=True)
    def cube(self, x: int) -> int:
        o = OutEvent(data_prompt="")
        event_emiter = InEventEmiter()
        self.post_out_event(o, event_emiter)
        event = await self.event_emiter.get()
        event.data

        return x * x * x


    def f(self):
        self.event_emiter.emit(InEvent(data_prompt=""))

async def main():
    flow = SimpleFlow()
    flow.register_conduit_hook(simple_flow_hook)
    result = await flow.process(x=7)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())