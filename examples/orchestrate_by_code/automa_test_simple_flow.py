import asyncio
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task, TaskResult
from bridgic.automa import AutoMa

# 这个例子展示如何通过“code-first”的编排模式，实现最简单的流程。
# 输入x，输出 3x+5，用一个乘法Worker和一个加法Worker来实现。

class MultiplyWorker(Worker):
    async def process_async(self, x):
        result = x * 3
        return result

class AddWorker(Worker):
    async def process_async(self, x):
        result = x + 5
        return result

class SimpleFlow(AutoMa):
    def __init__(self):
        super().__init__()
        self.multiply_worker = MultiplyWorker()
        self.add_worker = AddWorker()

    async def process_async(self, x) -> Task:
        result = await self.multiply_worker.process_async(x)
        result = await self.add_worker.process_async(result)
        return result


def main():
    flow = SimpleFlow()
    result = flow.process(x=7)
    print(result)

if __name__ == "__main__":
    main()