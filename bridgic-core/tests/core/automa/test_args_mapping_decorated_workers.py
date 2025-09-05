"""
Test cases for the arguments mapping mechanism of Bridgic framework.

This file only covers test cases for workers decorated by @worker, while test cases for custom workers are in another file, i.e., test_args_mapping_custom_workers.py.
"""
import pytest
import re

from bridgic.core.automa import GraphAutoma, worker, WorkerArgsMappingError, ArgsMappingRule
from typing import List, Tuple

class Coordinate:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

########################################################
######### Part One: Test ArgsMappingRule.AS_IS #########
########################################################

class Flow_1_Test_AS_IS(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y):
        # Return a dict.
        return {"x": x, "y": y}

    @worker(dependencies=["func_1"])
    async def func_2(self, map: dict):
        assert map == {"x": 1, "y": 3}
        # Return a list.
        return [map["x"], map["y"]]

    @worker(dependencies=["func_2"])
    async def func_3(self, my_list):
        assert my_list == [1, 3]
        # Return a list.
        return my_list

    @worker(dependencies=["func_3"])
    async def func_4(self, *args):
        # Use *args to receive a list as is.
        assert args[0] == [1, 3]
        # Return a list.
        return args[0]

    @worker(dependencies=["func_4"])
    async def func_5(self, my_list: List[int], *args):
        # Test case that both my_list and *args are used.
        assert my_list == [1, 3]
        assert args == ()
        # Return a custom object.
        return Coordinate(*my_list)

    @worker(dependencies=["func_5"])
    async def func_6(self, coord: Coordinate):
        assert coord.x == 1
        assert coord.y == 3
        return coord

    @worker(dependencies=["func_6"])
    async def func_7(self, coord: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        return coord

    @worker(dependencies=["func_7"])
    async def func_8(self, coord, coord2: Coordinate=Coordinate(11, 33)):
        # Test case that a default value is provided for the parameter.
        assert coord.x == 1
        assert coord.y == 3
        assert coord2.x == 11
        assert coord2.y == 33
        # return None test

    @worker(dependencies=["func_8"])
    async def func_9(self, value):
        assert value is None
        # return None again

    @worker(dependencies=["func_9"])
    async def func_10(self, *args):
        # Use args to receive None.
        assert len(args) == 1
        assert args[0] is None
        # return None again

    @worker(dependencies=["func_10"])
    async def func_11(self):
        # Test the special case of returning None and no arguments are expected.
        assert self.func_10.output_buffer is None

@pytest.fixture
def flow_1():
    flow = Flow_1_Test_AS_IS(output_worker_key="func_11")
    return flow

@pytest.mark.asyncio
async def test_flow_1(flow_1):
    # Test case for positional input arguments.
    result = await flow_1.arun(1, 3)
    assert result is None
    # Test case for keyword input arguments.
    result = await flow_1.arun(x=1, y=3)
    assert result is None

class Flow_2_Test_AS_IS(GraphAutoma):
    """
    Test case for mutiple depenencies when args_mapping_rule is ArgsMappingRule.AS_IS.
    """
    @worker(is_start=True)
    async def start(self, x: int, y):
        return Coordinate(x, y)

    @worker(dependencies=["start"])
    async def func_1(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x + coord.y

    @worker(dependencies=["start"])
    async def func_2(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test

    @worker(dependencies=["start"])
    async def func_3(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return coord.x * coord.y

    #Note: The order of dependencies is important for args mapping!!!
    @worker(dependencies=["func_2", "func_1", "func_3"])
    async def end(self, a, b, c):
        # !!! The values of a,b,c are consistent with the order of dependencies.
        assert a is None
        assert b == 5
        assert c == 6
        return (a, b, c)

@pytest.fixture
def flow_2():
    flow = Flow_2_Test_AS_IS(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_2(flow_2):
    # Test case for positional input arguments.
    result = await flow_2.arun(2, 3)
    assert result == (None, 5, 6)
    # Test case for keyword input arguments.
    result = await flow_2.arun(x=2, y=3)
    assert result == (None, 5, 6)

class Flow_3_Test_AS_IS(GraphAutoma):
    """
    Test case for dynamically adding mutiple depenency workers when args_mapping_rule is ArgsMappingRule.AS_IS.
    """
    @worker(is_start=True)
    async def start(self, x: int, y):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.add_func_as_worker(
                key=worker_key,
                func=self.func_1,
            )
            self.ferry_to(worker_key, coord=coord)
        self.add_func_as_worker(
            key="merge",
            func=self.merge,
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            #args_mapping_rule is the default value, which is ArgsMappingRule.AS_IS,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.set_output_worker("merge")
    
    async def func_1(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

    async def merge(self, *args):
        # For args_mapping_rule=ArgsMappingRule.AS_IS, you can use *args to receive the return values of dynamically added workers.
        return [*args]

@pytest.fixture
def flow_3():
    flow = Flow_3_Test_AS_IS()
    return flow

@pytest.mark.asyncio
async def test_flow_3_positional_inputs(flow_3):
    # Test case for positional input arguments.
    coordinates = await flow_3.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_3_keyword_inputs(flow_3):
    # Test case for keyword input arguments.
    coordinates = await flow_3.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class Flow_4_ErrorTest_AS_IS(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y):
        return {"x": x, "y": y}

    @worker(dependencies=["func_1"])
    async def func_2(self, x2: int, y2: int):
        # This will raise an error due to too much non-default parameters.
        return [x2, y2]

@pytest.fixture
def flow_4():
    flow = Flow_4_ErrorTest_AS_IS(output_worker_key="func_2")
    return flow

@pytest.mark.asyncio
async def test_flow_4(flow_4):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("func_2() missing 1 required positional argument: 'y2'")):
        await flow_4.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("func_2() missing 1 required positional argument: 'y2'")):
        await flow_4.arun(x=2, y=3)

class Flow_5_ErrorTest_AS_IS(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int, y: int):
        return {"x": x, "y": y}

    @worker(dependencies=["start"])
    async def func_1(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["x"]

    @worker(dependencies=["start"])
    async def func_2(self, map: dict):
        assert map == {"x": 2, "y": 3}
        return map["y"]

    @worker(dependencies=["func_1", "func_2"])
    async def end(self):
        # This will raise an error due to no positional parameters are provided.
        return None

@pytest.fixture
def flow_5():
    flow = Flow_5_ErrorTest_AS_IS(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_5(flow_5):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("end() takes 1 positional argument but 3 were given")):
        await flow_5.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("end() takes 1 positional argument but 3 were given")):
        await flow_5.arun(x=2, y=3)

########################################################
######### Part Two: Test ArgsMappingRule.UNPACK ########
########################################################

class Flow_A_Test_UNPACK(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        # Return a dict which is unpack-able.
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int, y: int, z: int=0):
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

    @worker(dependencies=["func_2"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_3(self, **kwargs):
        # Use **kwargs to receive a dict when args_mapping_rule=ArgsMappingRule.UNPACK.
        x = kwargs["x"]
        y = kwargs["y"]
        z = kwargs["z"]
        assert x == 4
        assert y == 5
        assert z == 0
        # Return a dict again.
        return {"x": x+1, "y": y+1, "z": z}

    @worker(dependencies=["func_3"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_4(self, y: int):
        # Test how to receive only parts of the dict.
        assert y == 6
        # Return a empty dict.
        return {}

    @worker(dependencies=["func_4"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_5(self):
        # No arguments are needed for the return value of a empty dict.
        # Return a list.
        return [100, 200, 300]

    @worker(dependencies=["func_5"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_6(self, x, y, z):
        # Return a tuple.
        return x+1, y+1, z+1

    @worker(dependencies=["func_6"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_7(self, *args):
        # Use *args to receive a list.
        assert len(args) == 3
        assert args[0] == 101
        assert args[1] == 201
        assert args[2] == 301
        # Return a list again.
        return [w+1 for w in args]

    @worker(dependencies=["func_7"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_8(self, x=0, *args):
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
def flow_A():
    flow = Flow_A_Test_UNPACK(output_worker_key="func_8")
    return flow

@pytest.mark.asyncio
async def test_flow_A(flow_A):
    # Test case for positional input arguments.
    coordinates = await flow_A.arun(2, 3)
    assert coordinates == [103, 203, 303]
    # Test case for keyword input arguments.
    coordinates = await flow_A.arun(x=2, y=3)
    assert coordinates == [103, 203, 303]

class Flow_B_ErrorTest_UNPACK(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(is_start=True)
    async def func_2(self, x: int, y: int):
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_1", "func_2"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def end(self, x: int, y: int):
        # This will raise an error due to muiltiple dependencies.
        return {"x": x+1, "y": y+1}

@pytest.fixture
def flow_B():
    flow = Flow_B_ErrorTest_UNPACK(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_B(flow_B):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has exactly one dependency"):
        await flow_B.arun(x=2, y=3)

class Flow_C_ErrorTest_UNPACK(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x: int, y: int):
        # Return a value that is not unpack-able.
        return x

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x: int):
        # This will raise an error because the return value is not unpack-able.
        return x

@pytest.fixture
def flow_C():
    flow = Flow_C_ErrorTest_UNPACK(output_worker_key="func_2")
    return flow

@pytest.mark.asyncio
async def test_flow_C(flow_C):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="only valid for tuple/list, or dict"):
        await flow_C.arun(x=2, y=3)

########################################################
######## Part Three: Test ArgsMappingRule.MERGE ########
########################################################

class Flow_I_Test_MERGE(GraphAutoma):
    """
    Test case for mutiple depenencies when args_mapping_rule is ArgsMappingRule.MERGE.
    """
    @worker(is_start=True)
    async def start(self, x: int, y):
        return Coordinate(x, y)

    @worker(dependencies=["start"])
    async def func_1(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x + 1, coord.y + 1)
        
    @worker(dependencies=["start"])
    async def func_2(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        # return None test
        return None

    @worker(dependencies=["start"])
    async def func_3(self, coord: Coordinate):
        assert coord.x == 2
        assert coord.y == 3
        return Coordinate(coord.x * coord.y, coord.x * coord.y)

    #Note: The order of dependencies is important for args mapping!!!
    @worker(dependencies=["func_2", "func_1", "func_3"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def end1(self, coordinates: List[Coordinate]):
        # !!! The values of coordinates are consistent with the order of dependencies.
        assert len(coordinates) == 3
        assert coordinates[0] is None
        assert coordinates[1].x == 3
        assert coordinates[1].y == 4
        assert coordinates[2].x == 6
        assert coordinates[2].y == 6
        return coordinates

    @worker(dependencies=["func_1", "func_2", "func_3"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def end2(self, *args):
        # Test case for *args to receive a list when args_mapping_rule=ArgsMappingRule.MERGE.
        assert len(args) == 1
        coordinates = args[0]
        return coordinates

@pytest.fixture
def flow_I():
    flow = Flow_I_Test_MERGE(output_worker_key="end2")
    return flow

@pytest.mark.asyncio
async def test_flow_I(flow_I):
    # Test case for positional input arguments.
    coordinates = await flow_I.arun(2, 3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6
    # Test case for keyword input arguments.
    coordinates = await flow_I.arun(x=2, y=3)
    assert len(coordinates) == 3
    assert coordinates[0].x == 3
    assert coordinates[0].y == 4
    assert coordinates[1] is None
    assert coordinates[2].x == 6
    assert coordinates[2].y == 6

class Flow_II_Test_MERGE(GraphAutoma):
    """
    Test case for dynamically adding mutiple depenency workers when args_mapping_rule is ArgsMappingRule.MERGE.
    """
    @worker(is_start=True)
    async def start(self, x: int, y):
        coord1 = Coordinate(x, y)
        coord2 = Coordinate(x + 1, y + 1)
        coord3 = Coordinate(x - 1, y - 1)
        coordinates = [coord1, coord2, coord3]
        # Dynamically add workers to the flow
        # The number of added workers may be determined dynamically according to the input data.
        for coord in coordinates:
            worker_key = f"func_coord_{coord.x}_{coord.y}"
            self.add_func_as_worker(
                key=worker_key,
                func=self.func_1,
            )
            self.ferry_to(worker_key, coord)
        self.add_func_as_worker(
            key="merge",
            func=self.merge,
            dependencies=[f"func_coord_{coord.x}_{coord.y}" for coord in coordinates],
            args_mapping_rule=ArgsMappingRule.MERGE,
        )
        # Dynamically set the output worker to the 'merge' worker which is dynamically added.
        self.set_output_worker("merge")
    
    async def func_1(self, coord: Coordinate):
        assert coord.x in [2, 3, 1]
        assert coord.y in [3, 4, 2]
        return coord

    async def merge(self, coordinates: List[Coordinate]):
        return coordinates

@pytest.fixture
def flow_II():
    flow = Flow_II_Test_MERGE()
    return flow

@pytest.mark.asyncio
async def test_flow_II_positional_inputs(flow_II):
    coordinates = await flow_II.arun(2, 3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

@pytest.mark.asyncio
async def test_flow_II_keyword_inputs(flow_II):
    coordinates = await flow_II.arun(x=2, y=3)
    coord1, coord2, coord3 = coordinates
    assert coord1.x == 2
    assert coord1.y == 3
    assert coord2.x == 3
    assert coord2.y == 4
    assert coord3.x == 1
    assert coord3.y == 2

class Flow_III_ErrorTest_MERGE(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int, y):
        return [x, y]

    @worker(dependencies=["start"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def end(self, my_list: Tuple[int, int]):
        # This will raise an error due to only one dependency.
        return Coordinate(my_list[0], my_list[1])

@pytest.fixture
def flow_III():
    flow = Flow_III_ErrorTest_MERGE(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_III(flow_III):
    # Test case for positional input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has at least 2 dependencies"):
        await flow_III.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(WorkerArgsMappingError, match="must has at least 2 dependencies"):
        await flow_III.arun(x=2, y=3)

class Flow_IV_ErrorTest_MERGE(GraphAutoma):
    @worker(is_start=True)
    async def start1(self, x: int, y):
        return [x, y]

    @worker(is_start=True)
    async def start2(self, x: int, y):
        return [x, y]

    @worker(dependencies=["start1", "start2"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def end(self, x2: int, y2: int):
        # This will raise an error due to too many parameters (x, y).
        return Coordinate(x2, y2)

@pytest.fixture
def flow_IV():
    flow = Flow_IV_ErrorTest_MERGE(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_IV(flow_IV):
    # Test case for positional input arguments.
    with pytest.raises(TypeError, match=re.escape("end() missing 1 required positional argument: 'y2'")):
        await flow_IV.arun(2, 3)
    # Test case for keyword input arguments.
    with pytest.raises(TypeError, match=re.escape("end() missing 1 required positional argument: 'y2'")):
        await flow_IV.arun(x=2, y=3)

########################################################
##### Part Four: Test ArgsMappingRule.SUPPRESSED #######
########################################################

class Flow_101_Test_SUPPRESSED(GraphAutoma):
    @worker(is_start=True)
    async def start(self, x: int, y):
        return [x, y]

    @worker(dependencies=["start"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
    async def end(self):
        x, y = self.start.output_buffer
        return x, y

@pytest.fixture
def flow_101():
    flow = Flow_101_Test_SUPPRESSED(output_worker_key="end")
    return flow

@pytest.mark.asyncio
async def test_flow_101(flow_101):
    # Test case for positional input arguments.
    result = await flow_101.arun(2, 3)
    assert result == (2, 3)
    # Test case for keyword input arguments.
    result = await flow_101.arun(x=2, y=3)
    assert result == (2, 3)
