import asyncio
from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import Task
from bridgic.automa import AutoMa
from bridgic.core.worker import CallableWorker
# 这个例子测试“通过代码编排”的模式，演示如何将一个function或method包装成一个Processor。
# 输入x，输出 3x+5，最终用一个自动创建的乘法Worker和一个加法Worker来实现。

def multiply_3(a: int) -> int:
    return a * 3

class MyAdder:
    def add_5(self, a: int) -> int:
        return a + 5

class SimpleFlow(AutoMa):
    def __init__(self):
        super().__init__()
        self.multiply_worker = CallableWorker(multiply_3)
        my_adder = MyAdder()
        self.add_worker = CallableWorker(my_adder.add_5)

    async def process_async(self, x):
        result = await self.multiply_worker.process_async(x)
        result = await self.add_worker.process_async(result)
        return result

def main():
    flow = SimpleFlow()
    result = flow.process(x=7)
    print(result)

if __name__ == "__main__":
    main()
