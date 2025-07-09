import pytest

from bridgic.automa import *

class SimpleFlow_1(GraphAutoma):
    @worker(is_start=True)
    def func_1(self, x: int, y) -> dict:
        return {"x": x, "y": y}

    @worker(dependencies=["func_1"], args_mapping_rule="as_dict")
    def func_2(self, x: int, y: int = 4):
        return (x, y)

@pytest.fixture
def simple_flow_1():
    yield SimpleFlow_1(output_worker_name="func_2")

@pytest.mark.asyncio
async def test_simple_flow_1(simple_flow_1):
    result = await simple_flow_1.process_async(x=1, y=3)
    assert result == (1, 3)


class SimpleFlow_2(GraphAutoma):
    @worker(is_start=True)
    def func_1(self, x: int, y) -> tuple:
        return x, y

    @worker(dependencies=["func_1"], args_mapping_rule="as_list")
    def func_2(self, x: int, y: int):
        return x * y

@pytest.fixture
def simple_flow_2():
    yield SimpleFlow_2(output_worker_name="func_2")

@pytest.mark.asyncio
async def test_simple_flow_2(simple_flow_2):
    result = await simple_flow_2.process_async(x=2, y=3)
    assert result == 6


class SimpleFlow_3(GraphAutoma):
    @worker(is_start=True)
    def func_1(self, x: int, y) -> dict:
        return {"sum": x+y}

    @worker(dependencies=["func_1"], args_mapping_rule="as_list")
    def func_2(self, obj: dict):
        return obj["sum"] * 2

@pytest.fixture
def simple_flow_3():
    yield SimpleFlow_3(output_worker_name="func_2")

@pytest.mark.asyncio
async def test_simple_flow_3(simple_flow_3):
    result = await simple_flow_3.process_async(x=1, y=3)
    assert result == 8

class SimpleFlow_4(GraphAutoma):
    def __init__(self):
        super().__init__(output_worker_name="merge_23")

    @worker(is_start=True)
    def start_1(self, x: int, y: int) -> dict:
        return {"key_a": x, "key_b": y}

    @worker(dependencies=["start_1"], args_mapping_rule="as_dict")
    def func_2(self, key_a: int):
        return key_a * 3

    @worker(dependencies=["start_1"], args_mapping_rule="as_dict")
    def func_3(self, key_b: int):
        return key_b * 5

    @worker(dependencies=["func_2", "func_3"])
    def merge_23(self):
        a_3 = self.func_2.output_buffer
        b_5 = self.func_3.output_buffer
        return a_3 + b_5

@pytest.fixture
def simple_flow_4():
    yield SimpleFlow_4()
    # teardown code may be here

@pytest.mark.asyncio
async def test_simple_flow_4(simple_flow_4):
    x = 1
    y = 3
    result = await simple_flow_4.process_async(x=x, y=y)
    assert result == x * 3 + y * 5
