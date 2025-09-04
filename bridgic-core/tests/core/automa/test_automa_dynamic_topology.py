import pytest

from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
from bridgic.core.automa import AutomaRuntimeError, AutomaCompilationError

class DynamicFlow_1(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    async def func_3(self, map: dict):
        return {"x": map["x"]+300, "y": map["y"]+300}

    async def func_4(self, x: int, y: int):
        return {"x": x+400, "y": y+400}

@pytest.fixture
def dynamic_flow_1():
    # Test case: call add_worker(), remove_worker() before running.
    flow = DynamicFlow_1()
    # Add a new worker.
    flow.add_func_as_worker(
        key="func_3",
        func=flow.func_3,
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    # Remove a declared worker.
    flow.remove_worker("func_2")
    # Remove a dynamically added worker.
    flow.remove_worker("func_3")
    # Re-add a removed worker with different dependencies.
    flow.add_func_as_worker(
        key="func_3",
        func=flow.func_3,
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.AS_IS,
    )
    # Add another new worker.
    flow.add_func_as_worker(
        key="func_4",
        func=flow.func_4,
        dependencies=["func_3"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    # Specify the output worker.
    flow.set_output_worker("func_4")
    # Final flow topology (expected): func_1 -> func_3 -> func_4.
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_1(dynamic_flow_1):
    coord = await dynamic_flow_1.arun(x=2, y=3)
    assert coord == {"x": 703, "y": 704}

#############################################################
class DynamicFlow_2(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        # Test case: call add_worker(), remove_worker() after running.
        # Add a new worker.
        self.add_func_as_worker(
            key="func_3",
            func=self.func_3,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.AS_IS,
        )
        # Remove a declared worker.
        self.remove_worker("func_2")
        # Remove a dynamically added worker.
        self.remove_worker("func_3")
        # Re-add a removed worker with different dependencies.
        self.add_func_as_worker(
            key="func_3",
            func=self.func_3,
            dependencies=["func_1"],
            args_mapping_rule=ArgsMappingRule.AS_IS,
        )
        # Add another new worker.
        self.add_func_as_worker(
            key="func_4",
            func=self.func_4,
            dependencies=["func_3"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        # Specify the output worker.
        self.set_output_worker("func_4")
        # Final flow topology (expected): func_1 -> func_3 -> func_4.
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    async def func_3(self, map: dict):
        return {"x": map["x"]+300, "y": map["y"]+300}

    async def func_4(self, x: int, y: int):
        return {"x": x+400, "y": y+400}

@pytest.fixture
def dynamic_flow_2():
    flow = DynamicFlow_2()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_2(dynamic_flow_2):
    coord = await dynamic_flow_2.arun(x=2, y=3)
    assert coord == {"x": 703, "y": 704}

#############################################################

class DynamicFlow_3(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

@pytest.fixture
def dynamic_flow_3():
    # Test case: call remove_worker() to get a empty graph.
    flow = DynamicFlow_3(output_worker_key="func_2")
    # The removal order for different workers is irrelevant.
    flow.remove_worker("func_1")
    flow.remove_worker("func_2")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_3(dynamic_flow_3):
    result = await dynamic_flow_3.arun(x=2, y=3)
    assert result == None

#############################################################

class DynamicFlow_4(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        # Test case: Dynamically add the immediate successor worker and the follow-up workers "in batch".
        self.add_func_as_worker(
            key="func_2",
            func=self.func_2,
            dependencies=["func_1"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        for i in range(3, 7):
            self.add_func_as_worker(
                key=f"func_{i}",
                func=self.func_resusable,
                dependencies=[f"func_{i-1}"],
            )
        self.set_output_worker("func_6")
        return {"x": x+1, "y": y+1}

    async def func_2(self, x: int, y: int):
        return {"x": x, "y": y}

    async def func_resusable(self, coord: dict):
        return {"x": coord["x"]+10, "y": coord["y"]+20}

@pytest.fixture
def dynamic_flow_4():
    flow = DynamicFlow_4()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_4(dynamic_flow_4):
    coord = await dynamic_flow_4.arun(x=2, y=3)
    assert coord == {"x": 43, "y": 84}

#############################################################

class DynamicFlow_5_AddWorkerStepByStep(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        # Test case: Dynamically add the immediate successor worker "step by step".
        self.add_func_as_worker(
            key="func_2",
            func=self.func_2,
            dependencies=["func_1"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        self.set_output_worker("func_2")
        return {"x": x+1, "y": y+1}

    async def func_2(self, x: int, y: int):
        # Dynamically add the immediate successor worker
        self.add_func_as_worker(
            key="func_3",
            func=self.func_resusable,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        count_run = 0
        return count_run, {"x": x, "y": y}

    async def func_resusable(self, count_run: int, coord: dict):
        my_key = f"func_{count_run+3}"
        if my_key != "func_6":
            # Dynamically add the next worker "step by step".
            self.add_func_as_worker(
                key=f"func_{count_run+4}",
                func=self.func_resusable,
                dependencies=[my_key],
                args_mapping_rule=ArgsMappingRule.UNPACK,
            )
        else:
            self.set_output_worker("func_6")
        return count_run + 1, {"x": coord["x"]+10, "y": coord["y"]+20}

@pytest.fixture
def dynamic_flow_5():
    flow = DynamicFlow_5_AddWorkerStepByStep()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_5(dynamic_flow_5):
    coord = await dynamic_flow_5.arun(x=2, y=3)
    assert coord == (4, {"x": 43, "y": 84})
    assert dynamic_flow_5.all_workers() == [f"func_{i}" for i in range(1, 7)]

#############################################################

class DynamicFlow_6_RemovePredecessor(GraphAutoma):
    """
    Test case: Dynamically remove the predecessor worker.
    This is the typical use case of remove_worker().
    """
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"])
    async def func_2(self, x: int):
        # Remove the predecessor worker.
        self.remove_worker("func_1")
        return x + 2

    @worker(dependencies=["func_2"])
    async def func_3(self, x: int):
        # Remove the predecessor worker again.
        self.remove_worker("func_2")
        return x + 3

@pytest.fixture
def dynamic_flow_6():
    flow = DynamicFlow_6_RemovePredecessor(output_worker_key="func_3")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_6(dynamic_flow_6):
    result = await dynamic_flow_6.arun(x=90)
    assert result == 96

#############################################################

class DynamicFlow_7_RemoveSelf(GraphAutoma):
    """
    Test case: Dynamically remove the currently running worker.
    Note: Removeing myself and the next worker are uncommon use cases.
    """
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"])
    async def func_2(self, x: int):
        # Remove myself!
        self.remove_worker("func_2")
        # Note: the next worker 'func_2' will not be executed, because 
        # all dependencies related to 'func_2' are also removed.
        return x + 2

    @worker(dependencies=["func_2"])
    async def func_3(self, x: int):
        return x + 3

@pytest.fixture
def dynamic_flow_7():
    flow = DynamicFlow_7_RemoveSelf(output_worker_key="func_2")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_7(dynamic_flow_7):
    result = await dynamic_flow_7.arun(x=90)
    # Note: the output worker 'func_2' has been removed after execution, so the result is None.
    assert result == None

#############################################################

class DynamicFlow_8_RemoveNext(GraphAutoma):
    """
    Test case: Dynamically remove the immediate successor worker.
    Note: Removeing myself and the next worker are uncommon use cases.
    """
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"])
    async def func_2(self, x: int):
        # Remove next worker
        self.remove_worker("func_3")
        return x + 2

    @worker(dependencies=["func_2"])
    async def func_3(self, x: int):
        return x + 3

@pytest.fixture
def dynamic_flow_8():
    flow = DynamicFlow_8_RemoveNext(output_worker_key="func_2")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_8(dynamic_flow_8):
    result = await dynamic_flow_8.arun(x=90)
    assert result == 93

#############################################################

class DynamicFlow_8_RemoveSelfAndFerryToNext(GraphAutoma):
    """
    Test case: Dynamically remove the currently running worker and ferry to the next worker.
    Note: Removeing myself and the next worker are uncommon use cases.
    """
    @worker(is_start=True)
    async def func_1(self, x: int):
        return x + 1

    @worker(dependencies=["func_1"])
    async def func_2(self, x: int):
        # Remove myself!
        self.remove_worker("func_2")
        self.ferry_to("func_3", x + 2)

    @worker(dependencies=["func_2"])
    async def func_3(self, x: int):
        return x + 3

@pytest.fixture
def dynamic_flow_8():
    flow = DynamicFlow_8_RemoveSelfAndFerryToNext(output_worker_key="func_3")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_8(dynamic_flow_8):
    result = await dynamic_flow_8.arun(x=90)
    assert result == 96

#############################################################

class DynamicFlow_9_RemoveBranch(GraphAutoma):
    """
    Test case: Dynamically remove a branch of the flow.
    Note: This is a rare use case. Just for testing.
    """
    @worker(is_start=True)
    async def start(self, x: int):
        return x + 10

    @worker(dependencies=["start"])
    async def func_1(self, x: int):
        # Remove myself!
        self.remove_worker("func_1")
        return x + 1

    @worker(dependencies=["start"])
    async def func_2(self, x: int):
        return x + 2

    @worker(dependencies=["start"])
    async def func_3(self, x: int):
        return x + 3

    @worker(dependencies=["func_1", "func_2", "func_3"])
    async def end(self, x_from_func_2: int, x_from_func_3: int):
        # Note: the dependency 'func_1' is removed, so only x_from_func_2 and x_from_func_3 are received.
        return x_from_func_2 + x_from_func_3

@pytest.fixture
def dynamic_flow_9():
    flow = DynamicFlow_9_RemoveBranch(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_9(dynamic_flow_9):
    result = await dynamic_flow_9.arun(x=2)
    assert result == 29

##############################################################
########## Following: Error && Exception Test Cases ##########
##############################################################

class DynamicFlow_A_DuplicateWorker(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        return {"x": x+2, "y": y+2}

    async def func_3(self, x: int, y: int):
        return {"x": x+3, "y": y+3}

@pytest.fixture
def dynamic_flow_a():
    flow = DynamicFlow_A_DuplicateWorker()
    flow.add_func_as_worker(
        key="func_3",
        func=DynamicFlow_A_DuplicateWorker.func_3,
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    with pytest.raises(AutomaRuntimeError, match="duplicate workers"):
        flow.add_func_as_worker(
            key="func_3",
            func=DynamicFlow_A_DuplicateWorker.func_3,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_a(dynamic_flow_a):
    await dynamic_flow_a.arun(x=1, y=2)

#############################################################

class DynamicFlow_B_DuplicateWorker(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        self.add_func_as_worker(
            key="func_3",
            func=DynamicFlow_A_DuplicateWorker.func_3,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        self.add_func_as_worker(
            key="func_3",
            func=DynamicFlow_A_DuplicateWorker.func_3,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        return {"x": x+2, "y": y+2}

    async def func_3(self, x: int, y: int):
        return {"x": x+3, "y": y+3}

@pytest.fixture
def dynamic_flow_b():
    flow = DynamicFlow_B_DuplicateWorker()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_b(dynamic_flow_b):
    with pytest.raises(AutomaRuntimeError, match="duplicate workers"):
        await dynamic_flow_b.arun(x=1, y=2)

#############################################################

class DynamicFlow_C_RemoveNotExist(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        self.remove_worker("func_3")
        return {"x": x+2, "y": y+2}

@pytest.fixture
def dynamic_flow_c():
    flow = DynamicFlow_C_RemoveNotExist()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_c(dynamic_flow_c):
    with pytest.raises(AutomaRuntimeError, match="not exist"):
        await dynamic_flow_c.arun(x=1, y=2)

#############################################################

class DynamicFlow_D_DependencyNotExist(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int):
        self.add_func_as_worker(
            key="func_3",
            func=DynamicFlow_D_DependencyNotExist.func_3,
            dependencies=["func_4"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        return {"x": x+2, "y": y+2}

    async def func_3(self, x: int, y: int):
        return {"x": x+3, "y": y+3}

@pytest.fixture
def dynamic_flow_d():
    flow = DynamicFlow_D_DependencyNotExist()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_d(dynamic_flow_d):
    with pytest.raises(AutomaCompilationError, match=r"the dependency .* not exist"):
        await dynamic_flow_d.arun(x=1, y=2)

#############################################################

class DynamicFlow_E_LoopDetect(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        self.add_func_as_worker(
            key="func_2",
            func=DynamicFlow_E_LoopDetect.func_2,
            dependencies=["func_3"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )
        self.add_func_as_worker(
            key="func_3",
            func=DynamicFlow_E_LoopDetect.func_3,
            dependencies=["func_2"],
            args_mapping_rule=ArgsMappingRule.UNPACK,
        )

        return {"x": x+1, "y": y+1}

    async def func_2(self, x: int, y: int):
        return {"x": x+2, "y": y+2}

    async def func_3(self, x: int, y: int):
        return {"x": x+3, "y": y+3}

@pytest.fixture
def dynamic_flow_e():
    flow = DynamicFlow_E_LoopDetect()
    return flow

@pytest.mark.asyncio
async def test_dynamic_flow_e(dynamic_flow_e):
    with pytest.raises(AutomaCompilationError, match=r".* not meet the DAG constraints, .* in cycle: .*"):
        await dynamic_flow_e.arun(x=1, y=2)
