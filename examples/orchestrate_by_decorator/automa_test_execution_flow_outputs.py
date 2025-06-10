from bridgic.automa import AutoMa
import asyncio
from bridgic.automa.bridge.decorator import worker
from pydantic import BaseModel
from bridgic.core.worker import WorkerLocalBuffer
# 这个例子展示了如何读取执行流的输出。

class ArithmeticResult(BaseModel):
    sum: int
    diff: int
    product: int
    quotient: float

class FakeFlow(AutoMa):
    @worker(is_start=True)
    def multiply_3(self, x: int) -> int:
        return x * 3

    @worker(listen=multiply_3)
    def square_and_cube(self, x: int) -> dict:
        return {"square_x": x * x, "cube_x": x * x * x}

    @worker(listen=square_and_cube)
    def arithmetic(self, square_x: int, cube_x: int) -> ArithmeticResult:
        local_buffer: WorkerLocalBuffer = self.arithmetic.worker_local_buffer
        local_buffer.state_a = "xxx"
        local_buffer.state_b = "yyy"
        self.arithmetic.worker_local_buffer.state_c = "zzz"
        print(f"self.arithmetic.worker_local_buffer: {self.arithmetic.worker_local_buffer}")

        sum = square_x + cube_x
        diff = square_x - cube_x
        product = square_x * cube_x
        quotient = square_x / cube_x
        return ArithmeticResult(sum=sum, diff=diff, product=product, quotient=quotient)

    @worker(listen=arithmetic)
    def average(self, arithmetic_result: ArithmeticResult) -> float:        
        return (arithmetic_result.sum + arithmetic_result.diff + arithmetic_result.product + arithmetic_result.quotient) / 4

    @worker(is_end=True, listen=average)
    def merge(self, avg: float) -> dict:
        square = self.execution_flow_output_buffer.square_and_cube['square_x']
        cube = self.execution_flow_output_buffer.square_and_cube['cube_x']
        sum = self.execution_flow_output_buffer.arithmetic.sum
        return {"avg": avg, "square": square, "cube": cube, "sum": sum}

def main():
    flow = FakeFlow()
    result = flow.process(x=7)
    print(result)
    
if __name__ == "__main__":
    main()