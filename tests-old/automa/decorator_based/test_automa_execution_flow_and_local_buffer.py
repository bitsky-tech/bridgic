import pytest
from bridgic.automa import AutoMa
from bridgic.automa.bridge.decorator import worker
from pydantic import BaseModel
from bridgic.core.worker import WorkerLocalBuffer

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

        sum = square_x + cube_x
        diff = square_x - cube_x
        product = square_x * cube_x
        quotient = square_x / cube_x
        print(f"sum: {sum}, diff: {diff}, product: {product}, quotient: {quotient}")
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

@pytest.fixture
def fake_flow():
    yield FakeFlow()
    # teardown code may be here

@pytest.mark.asyncio
async def test_fake_flow_result(fake_flow):
    x = 7
    result = await fake_flow.process_async(x=x)
    x_3 = x * 3
    square_x = x_3 * x_3
    cube_x = x_3 * x_3 * x_3
    sum = square_x + cube_x
    diff = square_x - cube_x
    product = square_x * cube_x
    quotient = square_x / cube_x
    avg = (sum + diff + product + quotient) / 4
    assert result["avg"] == avg
    assert result["square"] == square_x
    assert result["cube"] == cube_x
    assert result["sum"] == sum
