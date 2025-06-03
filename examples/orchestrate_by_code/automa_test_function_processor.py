import asyncio
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import DataRecord
from bridgic.automa import AutoMa
from bridgic.core.worker import FunctionProcessor
from bridgic.core.worker import MethodProcessor
# 这个例子测试“通过代码编排”的模式，演示如何将一个function或method包装成一个Processor。
# 输入x，输出 3x+5，最终用一个自动创建的乘法Processor和一个加法Processor来实现。

def multiply_3(a: int) -> int:
    return a * 3

class MyAdder:
    def add_5(self, a: int) -> int:
        return a + 5

class SimpleFlow(AutoMa):
    def __init__(self):
        super().__init__()
        self.multiply_processor = FunctionProcessor(multiply_3)
        my_adder = MyAdder()
        self.add_processor = MethodProcessor(my_adder.add_5)

    async def process(self, data: DataRecord) -> DataRecord:
        data = await self.multiply_processor.process(data)
        data = await self.add_processor.process(data)
        return data


async def main():
    x = 7
    flow = SimpleFlow()
    data = DataRecord.create_from_value(x)
    result = await flow.process(data)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())