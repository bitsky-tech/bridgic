"""
Test cases for the arguments mapping mechanism of Bridgic framework.

This file only covers test cases for custom workers, while test cases for decorated workers are in another file, i.e., test_args_mapping_decorated_workers.py.
"""
import pytest
import re

from bridgic.core.automa import GraphAutoma, WorkerArgsMappingError, ArgsMappingRule
from bridgic.core.automa.worker import Worker
from typing import List

class Coordinate:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

########################################################
######### Part One: Test ArgsMappingRule.AS_IS #########
########################################################

class Flow_1(GraphAutoma):
    ...

class Flow1Func1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        # Return a dict.
        return {"x": x, "y": y}

class Flow1Func2AsyncWorker(Worker):
    async def arun(self, map: dict):
        assert map == {"x": 1, "y": 3}
        # Return a list.
        return [map["x"], map["y"]]

class Flow1Func3AsyncWorker(Worker):
    async def arun(self, my_list):
        assert my_list == [1, 3]
        # Return a list.
        return my_list

class Flow1Func4AsyncWorker(Worker):
    async def arun(self, *args):
        # Use *args to receive a list as is.
        assert args[0] == [1, 3]
        # Return a list.
        return args[0]

class Flow1Func5AsyncWorker(Worker):
    async def arun(self, my_list: List[int], *args):
        # Test case that both my_list and *args are used.
        assert my_list == [1, 3]
        assert args == ()
        # Return a custom object.
        return Coordinate(*my_list)

class Flow1Func6AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 1
        assert coord.y == 3
        return coord

class Flow1Func7AsyncWorker(Worker):
    async def arun(self, coord: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        return coord

class Flow1Func8AsyncWorker(Worker):
    async def arun(self, coord, coord2: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        assert coord2.x == 11
        assert coord2.y == 33
        # return None test

class Flow1Func9AsyncWorker(Worker):
    async def arun(self, value):
        assert value is None
        # return None again
        return None

class Flow1Func10AsyncWorker(Worker):
    async def arun(self, *args):
        # Use args to receive None.
        assert len(args) == 1
        assert args[0] is None
        # return None again

class Flow1Func11AsyncWorker(Worker):
    async def arun(self):
        # Test the special case of returning None and no arguments are expected.
        assert self.parent.func_10.output_buffer is None

@pytest.fixture
def flow_1_arun():
    flow = Flow_1()
    flow.add_worker(
        "func_1",
        Flow1Func1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow1Func2AsyncWorker(),
        dependencies=["func_1"],
    )
    flow.add_worker(
        "func_3",
        Flow1Func3AsyncWorker(),
        dependencies=["func_2"],
    )
    flow.add_worker(
        "func_4",
        Flow1Func4AsyncWorker(),
        dependencies=["func_3"],
    )
    flow.add_worker(
        "func_5",
        Flow1Func5AsyncWorker(),
        dependencies=["func_4"],
    )
    flow.add_worker(
        "func_6",
        Flow1Func6AsyncWorker(),
        dependencies=["func_5"],
    )
    flow.add_worker(
        "func_7",
        Flow1Func7AsyncWorker(),
        dependencies=["func_6"],
    )
    flow.add_worker(
        "func_8",
        Flow1Func8AsyncWorker(),
        dependencies=["func_7"],
    )
    flow.add_worker(
        "func_9",
        Flow1Func9AsyncWorker(),
        dependencies=["func_8"],
    )
    flow.add_worker(
        "func_10",
        Flow1Func10AsyncWorker(),
        dependencies=["func_9"],
    )
    flow.add_worker(
        "func_11",
        Flow1Func11AsyncWorker(),
        dependencies=["func_10"],
    )
    flow.output_worker_key = "func_11"
    return flow

@pytest.mark.asyncio
async def test_flow_1_arun(flow_1_arun):
    # Test case for positional input arguments.
    result = await flow_1_arun.arun(1, 3)
    assert result is None
    # Test case for keyword input arguments.
    result = await flow_1_arun.arun(x=1, y=3)
    assert result is None

class Flow1Func1SyncWorker(Worker):
    def run(self, x: int, y: int):
        # Return a dict.
        return {"x": x, "y": y}

class Flow1Func2SyncWorker(Worker):
    def run(self, map: dict):
        assert map == {"x": 1, "y": 3}
        # Return a list.
        return [map["x"], map["y"]]

class Flow1Func3SyncWorker(Worker):
    def run(self, my_list):
        assert my_list == [1, 3]
        # Return a list.
        return my_list

class Flow1Func4SyncWorker(Worker):
    def run(self, *args):
        # Use *args to receive a list as is.
        assert args[0] == [1, 3]
        # Return a list.
        return args[0]

class Flow1Func5SyncWorker(Worker):
    def run(self, my_list: List[int], *args):
        # Test case that both my_list and *args are used.
        assert my_list == [1, 3]
        assert args == ()
        # Return a custom object.
        return Coordinate(*my_list)

class Flow1Func6SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 1
        assert coord.y == 3
        return coord

class Flow1Func7SyncWorker(Worker):
    def run(self, coord: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        return coord

class Flow1Func8SyncWorker(Worker):
    def run(self, coord, coord2: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        assert coord2.x == 11
        assert coord2.y == 33
        # return None test

class Flow1Func9SyncWorker(Worker):
    def run(self, value):
        assert value is None
        # return None again
        return None

class Flow1Func10SyncWorker(Worker):
    def run(self, *args):
        # Use args to receive None.
        assert len(args) == 1
        assert args[0] is None
        # return None again

class Flow1Func11SyncWorker(Worker):
    def run(self):
        # Test the special case of returning None and no arguments are expected.
        assert self.parent.func_10.output_buffer is None

@pytest.fixture
def flow_1_run():
    flow = Flow_1()
    flow.add_worker(
        "func_1",
        Flow1Func1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow1Func2SyncWorker(),
        dependencies=["func_1"],
    )
    flow.add_worker(
        "func_3",
        Flow1Func3SyncWorker(),
        dependencies=["func_2"],
    )
    flow.add_worker(
        "func_4",
        Flow1Func4SyncWorker(),
        dependencies=["func_3"],
    )
    flow.add_worker(
        "func_5",
        Flow1Func5SyncWorker(),
        dependencies=["func_4"],
    )
    flow.add_worker(
        "func_6",
        Flow1Func6SyncWorker(),
        dependencies=["func_5"],
    )
    flow.add_worker(
        "func_7",
        Flow1Func7SyncWorker(),
        dependencies=["func_6"],
    )
    flow.add_worker(
        "func_8",
        Flow1Func8SyncWorker(),
        dependencies=["func_7"],
    )
    flow.add_worker(
        "func_9",
        Flow1Func9SyncWorker(),
        dependencies=["func_8"],
    )
    flow.add_worker(
        "func_10",
        Flow1Func10SyncWorker(),
        dependencies=["func_9"],
    )
    flow.add_worker(
        "func_11",
        Flow1Func11SyncWorker(),
        dependencies=["func_10"],
    )
    flow.output_worker_key = "func_11"
    return flow

@pytest.mark.asyncio
async def test_flow_1_run(flow_1_run):
    # Test case for positional input arguments.
    result = await flow_1_run.arun(1, 3)
    assert result is None
    # Test case for keyword input arguments.
    result = await flow_1_run.arun(x=1, y=3)
    assert result is None

class Flow_2(GraphAutoma):
    """
    Test case for mutiple depenencies when args_mapping_rule is ArgsMappingRule.AS_IS.
    """
    ...

class Flow2StartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return Coordinate(x, y)

class Flow2Func1AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x + coord.y

class Flow2Func2AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test

class Flow2Func3AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x * coord.y

class Flow2EndAsyncWorker(Worker):
    async def arun(self, a, b, c):
        # !!! The values of a,b,c are consistent with the order of dependencies.
        assert a is None
        assert b == 5
        assert c == 6
        return (a, b, c)

@pytest.fixture
def flow_2_arun():
    flow = Flow_2()
    flow.add_worker(
        "start",
        Flow2StartAsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow2Func1AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        Flow2Func2AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_3",
        Flow2Func3AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end",
        Flow2EndAsyncWorker(),
        #Note: The order of dependencies is important for args mapping!!!
        dependencies=["func_2", "func_1", "func_3"],
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_2_arun(flow_2_arun):
    # Test case for positional input arguments.
    result = await flow_2_arun.arun(2, 3)
    assert result == (None, 5, 6)
    # Test case for keyword input arguments.
    result = await flow_2_arun.arun(x=2, y=3)
    assert result == (None, 5, 6)

class Flow2StartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return Coordinate(x, y)

class Flow2Func1SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x + coord.y

class Flow2Func2SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test

class Flow2Func3SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x * coord.y

class Flow2EndSyncWorker(Worker):
    def run(self, a, b, c):
        # !!! The values of a,b,c are consistent with the order of dependencies.
        assert a is None
        assert b == 5
        assert c == 6
        return (a, b, c)

@pytest.fixture
def flow_2_run():
    flow = Flow_2()
    flow.add_worker(
        "start",
        Flow2StartSyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow2Func1SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        Flow2Func2SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_3",
        Flow2Func3SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end",
        Flow2EndSyncWorker(),
        #Note: The order of dependencies is important for args mapping!!!
        dependencies=["func_2", "func_1", "func_3"],
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_2_run(flow_2_run):
    # Test case for positional input arguments.
    result = await flow_2_run.arun(2, 3)
    assert result == (None, 5, 6)
    # Test case for keyword input arguments.
    result = await flow_2_run.arun(x=2, y=3)
    assert result == (None, 5, 6)

class Flow_3(GraphAutoma):
    """
    Test case for dynamically adding mutiple depenency workers when args_mapping_rule is ArgsMappingRule.AS_IS.
    """
    ...

class Flow3FuncAsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

class Flow3MergeAsyncWorker(Worker):
    async def arun(self, *args):
        # For args_mapping_rule=ArgsMappingRule.AS_IS, you can use *args to receive the return values of dynamically added workers.
        return [*args]

class Flow3StartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.parent.add_worker(
                worker_key,
                Flow3FuncAsyncWorker(),
            )
            self.ferry_to(worker_key, coord=coord)
        self.parent.add_worker(
            "merge",
            Flow3MergeAsyncWorker(),
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            #args_mapping_rule is the default value, which is ArgsMappingRule.AS_IS,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.parent.output_worker_key = "merge"

@pytest.fixture
def flow_3_arun():
    flow = Flow_3()
    flow.add_worker(
        "start",
        Flow3StartAsyncWorker(),
        is_start=True,
    )
    return flow

@pytest.mark.asyncio
async def test_flow_3_arun_positional_inputs(flow_3_arun):
    # Test case for positional input arguments.
    coordinates = await flow_3_arun.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_3_arun_keyword_inputs(flow_3_arun):
    # Test case for keyword input arguments.
    coordinates = await flow_3_arun.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class Flow3FuncSyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

class Flow3MergeSyncWorker(Worker):
    def run(self, *args):
        # For args_mapping_rule=ArgsMappingRule.AS_IS, you can use *args to receive the return values of dynamically added workers.
        return [*args]

class Flow3StartSyncWorker(Worker):
    def run(self, x: int, y: int):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.parent.add_worker(
                worker_key,
                Flow3FuncSyncWorker(),
            )
            self.ferry_to(worker_key, coord=coord)
        self.parent.add_worker(
            "merge",
            Flow3MergeSyncWorker(),
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            #args_mapping_rule is the default value, which is ArgsMappingRule.AS_IS,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.parent.output_worker_key = "merge"

@pytest.fixture
def flow_3_run():
    flow = Flow_3()
    flow.add_worker(
        "start",
        Flow3StartSyncWorker(),
        is_start=True,
    )
    return flow

@pytest.mark.asyncio
async def test_flow_3_run_positional_inputs(flow_3_run):
    # Test case for positional input arguments.
    coordinates = await flow_3_run.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_3_run_keyword_inputs(flow_3_run):
    # Test case for keyword input arguments.
    coordinates = await flow_3_run.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class Flow_4_ErrorTest(GraphAutoma):
    ...

class Flow4Func1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return {"x": x, "y": y}

class Flow4Func2AsyncWorker(Worker):
    async def arun(self, x2: int, y2: int):
        # This will raise an error due to too much non-default parameters.
        return [x2, y2]
    
@pytest.fixture
def flow_4_arun():
    flow = Flow_4_ErrorTest()
    flow.add_worker(
        "func_1",
        Flow4Func1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow4Func2AsyncWorker(),
        dependencies=["func_1"],
    )
    flow.output_worker_key = "func_2"
    return flow

@pytest.mark.asyncio
async def test_flow_4_arun(flow_4_arun):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() missing 1 required positional argument: 'y2'")):
        await flow_4_arun.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() missing 1 required positional argument: 'y2'")):
        await flow_4_arun.arun(x=2, y=3)

class Flow4Func1SyncWorker(Worker):
    def run(self, x: int, y: int):
        return {"x": x, "y": y}

class Flow4Func2SyncWorker(Worker):
    def run(self, x2: int, y2: int):
        # This will raise an error due to too much non-default parameters.
        return [x2, y2]
    
@pytest.fixture
def flow_4_run():
    flow = Flow_4_ErrorTest()
    flow.add_worker(
        "func_1",
        Flow4Func1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow4Func2SyncWorker(),
        dependencies=["func_1"],
    )
    flow.output_worker_key = "func_2"
    return flow

@pytest.mark.asyncio
async def test_flow_4_run(flow_4_run):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("run() missing 1 required positional argument: 'y2'")):
        await flow_4_run.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("run() missing 1 required positional argument: 'y2'")):
        await flow_4_run.arun(x=2, y=3)

class Flow_5_ErrorTest(GraphAutoma):
    ...

class Flow5StartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return {"x": x, "y": y}

class Flow5Func1AsyncWorker(Worker):
    async def arun(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["x"]

class Flow5Func2AsyncWorker(Worker):
    async def arun(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["y"]

class Flow5EndAsyncWorker(Worker):
    async def arun(self):
        # This will raise an error due to no positional parameters are provided.
        return None

@pytest.fixture
def flow_5_arun():
    flow = Flow_5_ErrorTest()
    flow.add_worker(
        "start",
        Flow5StartAsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow5Func1AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        Flow5Func2AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end",
        Flow5EndAsyncWorker(),
        dependencies=["func_1", "func_2"],
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_5_arun(flow_5_arun):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() takes 1 positional argument but 3 were given")):
        await flow_5_arun.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() takes 1 positional argument but 3 were given")):
        await flow_5_arun.arun(x=2, y=3)

class Flow5StartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return {"x": x, "y": y}

class Flow5Func1SyncWorker(Worker):
    def run(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["x"]

class Flow5Func2SyncWorker(Worker):
    def run(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["y"]

class Flow5EndSyncWorker(Worker):
    def run(self):
        # This will raise an error due to no positional parameters are provided.
        return None

@pytest.fixture
def flow_5_run():
    flow = Flow_5_ErrorTest()
    flow.add_worker(
        "start",
        Flow5StartSyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow5Func1SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        Flow5Func2SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end",
        Flow5EndSyncWorker(),
        dependencies=["func_1", "func_2"],
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_5_run(flow_5_run):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("run() takes 1 positional argument but 3 were given")):
        await flow_5_run.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("run() takes 1 positional argument but 3 were given")):
        await flow_5_run.arun(x=2, y=3)

########################################################
######### Part Two: Test ArgsMappingRule.UNPACK ########
########################################################

class Flow_A(GraphAutoma):
    ...

class FlowAFunc1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        # Return a dict which is unpack-able.
        return {"x": x+1, "y": y+1}

class FlowAFunc2AsyncWorker(Worker):
    async def arun(self, x: int, y: int, z: int=0):
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

class FlowAFunc3AsyncWorker(Worker):
    async def arun(self, **kwargs):
        # Use **kwargs to receive a dict when args_mapping_rule=ArgsMappingRule.UNPACK.
        x = kwargs["x"]
        y = kwargs["y"]
        z = kwargs["z"]
        assert x == 4
        assert y == 5
        assert z == 0
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

class FlowAFunc4AsyncWorker(Worker):
    async def arun(self, y: int):
        # Test how to receive only parts of the dict.
        assert y == 6
        # Return a empty dict.
        return {}

class FlowAFunc5AsyncWorker(Worker):
    async def arun(self):
        # No arguments are needed for the return value of a empty dict.
        # Return a list.
        return [100, 200, 300]

class FlowAFunc6AsyncWorker(Worker):
    async def arun(self, x, y, z):
        # Return a tuple.
        return x+1, y+1, z+1

class FlowAFunc7AsyncWorker(Worker):
    async def arun(self, *args):
        # Use *args to receive a list.
        assert len(args) == 3
        assert args[0] == 101
        assert args[1] == 201
        assert args[2] == 301
        # Return a list again.
        return [w+1 for w in args]

class FlowAFunc8AsyncWorker(Worker):
    async def arun(self, x=0, *args):
        # Use positional parameters + *args to receive a list.
        assert len(args) == 2
        y = args[0]
        z = args[1]
        assert x == 102
        assert y == 202
        assert z == 302
        # Return a list again.
        return [x+1, y+1, z+1]

@pytest.fixture
def flow_A_arun():
    flow = Flow_A()
    flow.add_worker(
        "func_1",
        FlowAFunc1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowAFunc2AsyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_3",
        FlowAFunc3AsyncWorker(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_4",
        FlowAFunc4AsyncWorker(),
        dependencies=["func_3"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_5",
        FlowAFunc5AsyncWorker(),
        dependencies=["func_4"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_6",
        FlowAFunc6AsyncWorker(),
        dependencies=["func_5"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_7",
        FlowAFunc7AsyncWorker(),
        dependencies=["func_6"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_8",
        FlowAFunc8AsyncWorker(),
        dependencies=["func_7"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "func_8"
    return flow

@pytest.mark.asyncio
async def test_flow_A_arun(flow_A_arun):
    # Test case for positional input arguments.
    coordinates = await flow_A_arun.arun(2, 3)
    assert coordinates == [103, 203, 303]
    # Test case for keyword input arguments.
    coordinates = await flow_A_arun.arun(x=2, y=3)
    assert coordinates == [103, 203, 303]

class FlowAFunc1SyncWorker(Worker):
    def run(self, x: int, y: int):
        # Return a dict which is unpack-able.
        return {"x": x+1, "y": y+1}

class FlowAFunc2SyncWorker(Worker):
    def run(self, x: int, y: int, z: int=0):
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

class FlowAFunc3SyncWorker(Worker):
    def run(self, **kwargs):
        # Use **kwargs to receive a dict when args_mapping_rule=ArgsMappingRule.UNPACK.
        x = kwargs["x"]
        y = kwargs["y"]
        z = kwargs["z"]
        assert x == 4
        assert y == 5
        assert z == 0
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

class FlowAFunc4SyncWorker(Worker):
    def run(self, y: int):
        # Test how to receive only parts of the dict.
        assert y == 6
        # Return a empty dict.
        return {}

class FlowAFunc5SyncWorker(Worker):
    def run(self):
        # No arguments are needed for the return value of a empty dict.
        # Return a list.
        return [100, 200, 300]

class FlowAFunc6SyncWorker(Worker):
    def run(self, x, y, z):
        # Return a tuple.
        return x+1, y+1, z+1

class FlowAFunc7SyncWorker(Worker):
    def run(self, *args):
        # Use *args to receive a list.
        assert len(args) == 3
        assert args[0] == 101
        assert args[1] == 201
        assert args[2] == 301
        # Return a list again.
        return [w+1 for w in args]

class FlowAFunc8SyncWorker(Worker):
    def run(self, x=0, *args):
        # Use positional parameters + *args to receive a list.
        assert len(args) == 2
        y = args[0]
        z = args[1]
        assert x == 102
        assert y == 202
        assert z == 302
        # Return a list again.
        return [x+1, y+1, z+1]

@pytest.fixture
def flow_A_run():
    flow = Flow_A()
    flow.add_worker(
        "func_1",
        FlowAFunc1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowAFunc2SyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_3",
        FlowAFunc3SyncWorker(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_4",
        FlowAFunc4SyncWorker(),
        dependencies=["func_3"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_5",
        FlowAFunc5SyncWorker(),
        dependencies=["func_4"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_6",
        FlowAFunc6SyncWorker(),
        dependencies=["func_5"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_7",
        FlowAFunc7SyncWorker(),
        dependencies=["func_6"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_8",
        FlowAFunc8SyncWorker(),
        dependencies=["func_7"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "func_8"
    return flow

@pytest.mark.asyncio
async def test_flow_A_run(flow_A_run):
    # Test case for positional input arguments.
    coordinates = await flow_A_run.arun(2, 3)
    assert coordinates == [103, 203, 303]
    # Test case for keyword input arguments.
    coordinates = await flow_A_run.arun(x=2, y=3)
    assert coordinates == [103, 203, 303]

class Flow_B_ErrorTest(GraphAutoma):
    ...

class FlowBFunc1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

class FlowBFunc2AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

class FlowBEndAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        # This will raise an error due to muiltiple dependencies in the setting of args_mapping_rule=ArgsMappingRule.UNPACK.
        return {"x": x+1, "y": y+1}

@pytest.fixture
def flow_B_arun():
    flow = Flow_B_ErrorTest()
    flow.add_worker(
        "func_1",
        FlowBFunc1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowBFunc2AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowBEndAsyncWorker(),
        dependencies=["func_1", "func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_B_arun(flow_B_arun):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B_arun.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B_arun.arun(x=2, y=3)

class FlowBFunc1SyncWorker(Worker):
    def run(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

class FlowBFunc2SyncWorker(Worker):
    def run(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

class FlowBEndSyncWorker(Worker):
    def run(self, x: int, y: int):
        # This will raise an error due to muiltiple dependencies in the setting of args_mapping_rule=ArgsMappingRule.UNPACK.
        return {"x": x+1, "y": y+1}

@pytest.fixture
def flow_B_run():
    flow = Flow_B_ErrorTest()
    flow.add_worker(
        "func_1",
        FlowBFunc1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowBFunc2SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowBEndSyncWorker(),
        dependencies=["func_1", "func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_B_run(flow_B_run):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B_run.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B_run.arun(x=2, y=3)

class Flow_C_ErrorTest(GraphAutoma):
    ...

class FlowCFunc1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        # Return a value that is not unpack-able.
        return x

class FlowCFunc2AsyncWorker(Worker):
    async def arun(self, x: int):
        # This will raise an error because the return value is not unpack-able.
        return x

@pytest.fixture
def flow_C_arun():
    flow = Flow_C_ErrorTest()
    flow.add_worker(
        "func_1",
        FlowCFunc1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowCFunc2AsyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "func_2"
    return flow

@pytest.mark.asyncio
async def test_flow_C_arun(flow_C_arun):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C_arun.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C_arun.arun(x=2, y=3)

class FlowCFunc1SyncWorker(Worker):
    def run(self, x: int, y: int):
        # Return a value that is not unpack-able.
        return x

class FlowCFunc2SyncWorker(Worker):
    def run(self, x: int):
        # This will raise an error because the return value is not unpack-able.
        return x

@pytest.fixture
def flow_C_run():
    flow = Flow_C_ErrorTest()
    flow.add_worker(
        "func_1",
        FlowCFunc1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        FlowCFunc2SyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.output_worker_key = "func_2"
    return flow

@pytest.mark.asyncio
async def test_flow_C_run(flow_C_run):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C_run.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C_run.arun(x=2, y=3)

########################################################
######## Part Three: Test ArgsMappingRule.MERGE ########
########################################################

class Flow_I(GraphAutoma):
    """
    Test case for one/multiple depenencies when args_mapping_rule is ArgsMappingRule.MERGE.
    """
    ...

class FlowIStartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return Coordinate(x, y)

class FlowIFunc1AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x + 1, coord.y + 1)

class FlowIFunc2AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test
        return None

class FlowIFunc3AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x * coord.y, coord.x * coord.y)

class FlowIEnd1AsyncWorker(Worker):
    async def arun(self, coordinates: List[Coordinate]):
        # !!! The values of coordinates are consistent with the order of dependencies.
        assert len(coordinates) == 3
        assert coordinates[0] is None
        assert coordinates[1].x == 3
        assert coordinates[1].y == 4
        assert coordinates[2].x == 6
        assert coordinates[2].y == 6
        return coordinates

class FlowIEnd2AsyncWorker(Worker):
    async def arun(self, *args):
        assert len(args) == 1
        coordinates = args[0]
        return coordinates

@pytest.fixture
def flow_I_arun():
    flow = Flow_I()
    flow.add_worker(
        "start",
        FlowIStartAsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        FlowIFunc1AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        FlowIFunc2AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_3",
        FlowIFunc3AsyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end_1",
        FlowIEnd1AsyncWorker(),
        #Note: The order of dependencies is important for args mapping!!!
        dependencies=["func_2", "func_1", "func_3"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.add_worker(
        "end_2",
        FlowIEnd2AsyncWorker(),
        dependencies=["func_1", "func_2", "func_3"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end_2"
    return flow

@pytest.mark.asyncio
async def test_flow_I_arun(flow_I_arun):
    # Test case for positional input arguments.
    coordinates = await flow_I_arun.arun(2, 3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6
    # Test case for keyword input arguments.
    coordinates = await flow_I_arun.arun(x=2, y=3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6

class FlowIStartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return Coordinate(x, y)

class FlowIFunc1SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x + 1, coord.y + 1)

class FlowIFunc2SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test
        return None

class FlowIFunc3SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x * coord.y, coord.x * coord.y)

class FlowIEnd1SyncWorker(Worker):
    def run(self, coordinates: List[Coordinate]):
        # !!! The values of coordinates are consistent with the order of dependencies.
        assert len(coordinates) == 3
        assert coordinates[0] is None
        assert coordinates[1].x == 3
        assert coordinates[1].y == 4
        assert coordinates[2].x == 6
        assert coordinates[2].y == 6
        return coordinates

class FlowIEnd2SyncWorker(Worker):
    def run(self, *args):
        assert len(args) == 1
        coordinates = args[0]
        return coordinates

@pytest.fixture
def flow_I_run():
    flow = Flow_I()
    flow.add_worker(
        "start",
        FlowIStartSyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        FlowIFunc1SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_2",
        FlowIFunc2SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "func_3",
        FlowIFunc3SyncWorker(),
        dependencies=["start"],
    )
    flow.add_worker(
        "end_1",
        FlowIEnd1SyncWorker(),
        #Note: The order of dependencies is important for args mapping!!!
        dependencies=["func_2", "func_1", "func_3"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.add_worker(
        "end_2",
        FlowIEnd2SyncWorker(),
        dependencies=["func_1", "func_2", "func_3"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end_2"
    return flow

@pytest.mark.asyncio
async def test_flow_I_run(flow_I_run):
    # Test case for positional input arguments.
    coordinates = await flow_I_run.arun(2, 3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6
    # Test case for keyword input arguments.
    coordinates = await flow_I_run.arun(x=2, y=3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6

class Flow_II(GraphAutoma):
    """
    Test case for dynamically adding mutiple depenency workers when args_mapping_rule is ArgsMappingRule.MERGE.
    """
    ...

class FlowIIFunc1AsyncWorker(Worker):
    async def arun(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

class FlowIIMergeAsyncWorker(Worker):
    async def arun(self, coordinates: List[Coordinate]):
        return coordinates

class FlowIIStartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.parent.add_worker(
                worker_key,
                FlowIIFunc1AsyncWorker(),
            )
            self.ferry_to(worker_key, coord)
        self.parent.add_worker(
            "merge",
            FlowIIMergeAsyncWorker(),
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            args_mapping_rule=ArgsMappingRule.MERGE,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.parent.output_worker_key = "merge"

@pytest.fixture
def flow_II_arun():
    flow = Flow_II()
    flow.add_worker(
        "start",
        FlowIIStartAsyncWorker(),
        is_start=True,
    )
    return flow

@pytest.mark.asyncio
async def test_flow_II_arun_positional_inputs(flow_II_arun):
    coordinates = await flow_II_arun.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_II_arun_keyword_inputs(flow_II_arun):
    coordinates = await flow_II_arun.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class FlowIIFunc1SyncWorker(Worker):
    def run(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

class FlowIIMergeSyncWorker(Worker):
    def run(self, coordinates: List[Coordinate]):
        return coordinates

class FlowIIStartSyncWorker(Worker):
    def run(self, x: int, y: int):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.parent.add_worker(
                worker_key,
                FlowIIFunc1SyncWorker(),
            )
            self.ferry_to(worker_key, coord)
        self.parent.add_worker(
            "merge",
            FlowIIMergeSyncWorker(),
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            args_mapping_rule=ArgsMappingRule.MERGE,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.parent.output_worker_key = "merge"

@pytest.fixture
def flow_II_run():
    flow = Flow_II()
    flow.add_worker(
        "start",
        FlowIIStartSyncWorker(),
        is_start=True,
    )
    return flow

@pytest.mark.asyncio
async def test_flow_II_run_positional_inputs(flow_II_run):
    coordinates = await flow_II_run.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_II_run_keyword_inputs(flow_II_run):
    coordinates = await flow_II_run.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class Flow_III(GraphAutoma):
    """
    Test case for only one depenency when args_mapping_rule is ArgsMappingRule.MERGE.
    This case is valid.
    """
    ...

class FlowIIIStartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return [x, y]

class FlowIIIEndAsyncWorker(Worker):
    async def arun(self, my_list: List[int]):
        assert len(my_list) == 1
        coord = my_list[0]
        return Coordinate(coord[0], coord[1])

@pytest.fixture
def flow_III_arun():
    flow = Flow_III()
    flow.add_worker(
        "start",
        FlowIIIStartAsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowIIIEndAsyncWorker(),
        dependencies=["start"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_III_arun(flow_III_arun):
    # Test case for positional input arguments.
    coord = await flow_III_arun.arun(2, 3)
    assert coord.x == 2
    assert coord.y == 3
    # Test case for keyword input arguments.
    coord = await flow_III_arun.arun(x=2, y=3)
    assert coord.x == 2
    assert coord.y == 3

class FlowIIIStartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return [x, y]

class FlowIIIEndSyncWorker(Worker):
    def run(self, my_list: List[int]):
        return Coordinate(my_list[0], my_list[1])

class FlowIIIStartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return [x, y]

class FlowIIIEndSyncWorker(Worker):
    def run(self, my_list: List[int]):
        assert len(my_list) == 1
        coord = my_list[0]
        return Coordinate(coord[0], coord[1])

@pytest.fixture
def flow_III_run():
    flow = Flow_III()
    flow.add_worker(
        "start",
        FlowIIIStartSyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowIIIEndSyncWorker(),
        dependencies=["start"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_III_run(flow_III_run):
    # Test case for positional input arguments.
    coord = await flow_III_run.arun(2, 3)
    assert coord.x == 2
    assert coord.y == 3
    # Test case for keyword input arguments.
    coord = await flow_III_run.arun(x=2, y=3)
    assert coord.x == 2
    assert coord.y == 3

class Flow_IV_ErrorTest(GraphAutoma):
    ...

class FlowIVStart1AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return [x, y]

class FlowIVStart2AsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return [x, y]

class FlowIVEndAsyncWorker(Worker):
    async def arun(self, x2: int, y2: int):
        # This will raise an error due to too many parameters (x, y).
        return Coordinate(x2, y2)

@pytest.fixture
def flow_IV_arun():
    flow = Flow_IV_ErrorTest()
    flow.add_worker(
        "start1",
        FlowIVStart1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "start2",
        FlowIVStart2AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowIVEndAsyncWorker(),
        dependencies=["start1", "start2"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_IV_arun(flow_IV_arun):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() missing 1 required positional argument: 'y2'")):
        await flow_IV_arun.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("arun() missing 1 required positional argument: 'y2'")):
        await flow_IV_arun.arun(x=2, y=3)

class FlowIVStart1SyncWorker(Worker):
    def run(self, x: int, y: int):
        return [x, y]

class FlowIVStart2SyncWorker(Worker):
    def run(self, x: int, y: int):
        return [x, y]

class FlowIVEndSyncWorker(Worker):
    def run(self, x2: int, y2: int):
        # This will raise an error due to too many parameters (x, y).
        return Coordinate(x2, y2)

@pytest.fixture
def flow_IV_run():
    flow = Flow_IV_ErrorTest()
    flow.add_worker(
        "start1",
        FlowIVStart1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "start2",
        FlowIVStart2SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        FlowIVEndSyncWorker(),
        dependencies=["start1", "start2"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_IV_run(flow_IV_run):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("run() missing 1 required positional argument: 'y2'")):
        await flow_IV_run.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("run() missing 1 required positional argument: 'y2'")):
        await flow_IV_run.arun(x=2, y=3)

########################################################
##### Part Four: Test ArgsMappingRule.SUPPRESSED #######
########################################################

class Flow_101_Test(GraphAutoma):
    ...

class Flow101StartAsyncWorker(Worker):
    async def arun(self, x: int, y: int):
        return [x, y]

class Flow101EndAsyncWorker(Worker):
    async def arun(self):
        x, y = self.parent.start.output_buffer
        return x, y

@pytest.fixture
def flow_101_arun():
    flow = Flow_101_Test()
    flow.add_worker(
        "start",
        Flow101StartAsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        Flow101EndAsyncWorker(),
        dependencies=["start"],
        args_mapping_rule=ArgsMappingRule.SUPPRESSED,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_101_arun(flow_101_arun):
    # Test case for positional input arguments.
    result = await flow_101_arun.arun(2, 3)
    assert result == (2, 3)
    # Test case for keyword input arguments.
    result = await flow_101_arun.arun(x=2, y=3)
    assert result == (2, 3)

class Flow101StartSyncWorker(Worker):
    def run(self, x: int, y: int):
        return [x, y]

class Flow101EndSyncWorker(Worker):
    def run(self):
        x, y = self.parent.start.output_buffer
        return x, y

@pytest.fixture
def flow_101_run():
    flow = Flow_101_Test()
    flow.add_worker(
        "start",
        Flow101StartSyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "end",
        Flow101EndSyncWorker(),
        dependencies=["start"],
        args_mapping_rule=ArgsMappingRule.SUPPRESSED,
    )
    flow.output_worker_key = "end"
    return flow

@pytest.mark.asyncio
async def test_flow_101_run(flow_101_run):
    # Test case for positional input arguments.
    result = await flow_101_run.arun(2, 3)
    assert result == (2, 3)
    # Test case for keyword input arguments.
    result = await flow_101_run.arun(x=2, y=3)
    assert result == (2, 3)
