import pytest

from bridgic.automa import GraphAutoma, worker, ArgsMappingRule

class DynamicFlow_1(GraphAutoma):
    @worker(is_start=True)
    def func_1(self, x: int, y: int):
        # Test case: Dynamically add the immediate successor worker.
        self.add_func_as_worker(
            key="func_2",
            func=self.func_2,
            dependencies=["func_1"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        self.set_output_worker("func_2")
        return {"x": x+1, "y": y+1}

    def func_2(self, x: int, y: int):
        return {"x": x, "y": y}

@pytest.fixture
def dynamic_flow_1():
    flow = DynamicFlow_1()
    return flow

@pytest.mark.asyncio
async def test_dynamically_add_immediate_successor(dynamic_flow_1):
    coord = await dynamic_flow_1.process_async(x=2, y=3)
    assert coord == {"x": 3, "y": 4}
