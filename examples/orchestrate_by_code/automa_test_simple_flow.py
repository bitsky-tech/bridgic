import asyncio
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import DataRecord
from bridgic.automa import AutoMa

# 这个例子展示如何通过“code-first”的编排模式，实现最简单的流程。
# 输入x，输出 3x+5，用一个乘法Processor和一个加法Processor来实现。

class MultiplyProcessor(Worker):
    async def process_record(self, data: DataRecord) -> DataRecord:
        result = data.value * 3
        return DataRecord.create_from_value(result)

class AddProcessor(Worker):
    async def process_record(self, data: DataRecord) -> DataRecord:
        result = data.value + 5
        return DataRecord.create_from_value(result)

class SimpleFlow(AutoMa):
    def __init__(self):
        super().__init__()
        self.multiply_processor = MultiplyProcessor()
        self.add_processor = AddProcessor()

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