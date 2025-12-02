"""
Test cases for the inputs propagation mechanism of Bridgic framework.
"""

import pytest

from bridgic.core.config import GlobalSetting
from bridgic.core.automa import GraphAutoma, worker, RunningOptions
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from typing import Tuple, Any, Dict, Optional

###########################################################
###### Part One: Test decorated workers -- async def ######
###########################################################

class Flow_1(GraphAutoma):
    @worker(is_start=True)
    async def start_1(self, a, b, /, x, user_input: str):
        # Test case for multiple start workers.
        assert x == 2
        assert user_input == "hi"
        return a, b, x + 1

    @worker(is_start=True)
    async def start_2(self, a, b, /, y, user_input: str):
        # Test case for multiple start workers.
        assert y == 3
        assert user_input == "hi"
        return a, b, y + 2

    @worker(dependencies=["start_1", "start_2"], args_mapping_rule=ArgsMappingRule.MERGE)
    async def func_1(self, my_list: Tuple[Any, Any], user_input: str):
        # Test case for args_mapping_rule=ArgsMappingRule.MERGE.
        start_1_output = my_list[0]
        start_2_output = my_list[1]
        assert start_1_output[0] == 11
        assert start_1_output[1] == 22
        assert start_2_output[0] == 11
        assert start_2_output[1] == 22
        assert user_input == "hi"
        x = start_1_output[2]
        y = start_2_output[2]
        assert x == 3
        assert y == 5
        return [x+1, y+1]

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_2(self, x, y, user_input: str):
        # Test case for UNPACK list.
        assert x == 4
        assert y == 6
        assert user_input == "hi"
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_2"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def func_3(self, x, y, user_input: str):
        # Test case for UNPACK dict.
        assert x == 5
        assert y == 7
        assert user_input == "hi"
        self.add_func_as_worker(
            "func_4",
            self.func_4,
        )
        # Test case for ferry_to() with positional arguments.
        self.ferry_to("func_4", x+1, y+1)

    async def func_4(self, x, y, user_input: str):
        assert x == 6
        assert y == 8
        assert user_input == "hi"
        self.add_func_as_worker(
            "func_5",
            self.func_5,
        )
        # Test case for ferry_to() with keyword arguments.
        self.ferry_to("func_5", x=x+1, y=y+1)
    
    async def func_5(self, x, y, user_input: str):
        assert x == 7
        assert y == 9
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments and keyword arguments.
        self.ferry_to("end", x+1, y+1, user_input=user_input)
        # Note: the `end` worker will be a netsted Automa.

class Flow_1_Nested(GraphAutoma):
    @worker(is_start=True)
    async def func_1(self, x, y, user_input: str):
        assert x == 8
        assert y == 10
        assert user_input == "hi"
        return x+1, y+1
    
    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK, is_output=True)
    async def func_2(self, x, y, user_input: str):
        assert x == 9
        assert y == 11
        assert user_input == "hi"
        return x, y

@pytest.fixture
def flow_1():
    flow = Flow_1()
    flow_nested = Flow_1_Nested()
    # Test case for inputs propagation in nested Automa.
    flow.add_worker(
        "end",
        flow_nested,
        is_output=True
    )
    return flow

@pytest.mark.asyncio
async def test_flow_1(flow_1):
    result = await flow_1.arun(11, 22, x=2, y=3, user_input="hi")
    assert result == (9, 11)

###########################################################
###### Part Two: Test decorated workers -- sync def ######
###########################################################

class Flow_2(GraphAutoma):
    @worker(is_start=True)
    def start_1(self, a, b, /, x, user_input: str):
        # Test case for multiple start workers.
        assert x == 2
        assert user_input == "hi"
        return a, b, x + 1

    @worker(is_start=True)
    def start_2(self, a, b, /, y, user_input: str):
        # Test case for multiple start workers.
        assert y == 3
        assert user_input == "hi"
        return a, b, y + 2

    @worker(dependencies=["start_1", "start_2"], args_mapping_rule=ArgsMappingRule.MERGE)
    def func_1(self, my_list: Tuple[Any, Any], user_input: str):
        # Test case for args_mapping_rule=ArgsMappingRule.MERGE.
        start_1_output = my_list[0]
        start_2_output = my_list[1]
        assert start_1_output[0] == 11
        assert start_1_output[1] == 22
        assert start_2_output[0] == 11
        assert start_2_output[1] == 22
        assert user_input == "hi"
        x = start_1_output[2]
        y = start_2_output[2]
        assert x == 3
        assert y == 5
        return [x+1, y+1]

    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    def func_2(self, x, y, user_input: str):
        # Test case for UNPACK list.
        assert x == 4
        assert y == 6
        assert user_input == "hi"
        return {"x": x+1, "y": y+1}

    @worker(dependencies=["func_2"], args_mapping_rule=ArgsMappingRule.UNPACK)
    def func_3(self, x, y, user_input: str):
        # Test case for UNPACK dict.
        assert x == 5
        assert y == 7
        assert user_input == "hi"
        self.add_func_as_worker(
            "func_4",
            self.func_4,
        )
        # Test case for ferry_to() with positional arguments.
        self.ferry_to("func_4", x+1, y+1)

    def func_4(self, x, y, user_input: str):
        assert x == 6
        assert y == 8
        assert user_input == "hi"
        self.add_func_as_worker(
            "func_5",
            self.func_5,
        )
        # Test case for ferry_to() with keyword arguments.
        self.ferry_to("func_5", x=x+1, y=y+1)
    
    def func_5(self, x, y, user_input: str):
        assert x == 7
        assert y == 9
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments and keyword arguments.
        self.ferry_to("end", x+1, y+1, user_input=user_input)
        # Note: the `end` worker will be a netsted Automa.

class Flow_2_Nested(GraphAutoma):
    @worker(is_start=True)
    def func_1(self, x, y, user_input: str):
        assert x == 8
        assert y == 10
        assert user_input == "hi"
        return x+1, y+1
    
    @worker(dependencies=["func_1"], args_mapping_rule=ArgsMappingRule.UNPACK, is_output=True)
    def func_2(self, x, y, user_input: str):
        assert x == 9
        assert y == 11
        assert user_input == "hi"
        return x, y

@pytest.fixture
def flow_2():
    flow = Flow_2()
    flow_nested = Flow_2_Nested()
    # Test case for inputs propagation in nested Automa.
    flow.add_worker(
        "end",
        flow_nested,
        is_output=True
    )
    return flow

@pytest.mark.asyncio
async def test_flow_2(flow_2):
    result = await flow_2.arun(11, 22, x=2, y=3, user_input="hi")
    assert result == (9, 11)

###########################################################
##### Part Three: Test custom workers -- async arun() #####
###########################################################

class Flow_3(GraphAutoma):
    ...

class Flow3Start1AsyncWorker(Worker):
    async def arun(self, a, b, /, x, user_input: str):
        # Test case for multiple start workers.
        assert x == 2
        assert user_input == "hi"
        return a, b, x + 1

class Flow3Start2AsyncWorker(Worker):
    async def arun(self, a, b, /, y, user_input: str):
        # Test case for multiple start workers.
        assert y == 3
        assert user_input == "hi"
        return a, b, y + 2

class Flow3Func1AsyncWorker(Worker):
    async def arun(self, my_list: Tuple[Any, Any], user_input: str):
        # Test case for args_mapping_rule=ArgsMappingRule.MERGE.
        start_1_output = my_list[0]
        start_2_output = my_list[1]
        assert start_1_output[0] == 11
        assert start_1_output[1] == 22
        assert start_2_output[0] == 11
        assert start_2_output[1] == 22
        assert user_input == "hi"
        x = start_1_output[2]
        y = start_2_output[2]
        assert x == 3
        assert y == 5
        return [x+1, y+1]

class Flow3Func2AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        # Test case for UNPACK list.
        assert x == 4
        assert y == 6
        assert user_input == "hi"
        return {"x": x+1, "y": y+1}

class Flow3Func3AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        # Test case for UNPACK dict.
        assert x == 5
        assert y == 7
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments.
        self.ferry_to("func_4", x+1, y+1)

class Flow3Func4AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        assert x == 6
        assert y == 8
        assert user_input == "hi"
        # Test case for ferry_to() with keyword arguments.
        self.ferry_to("func_5", x=x+1, y=y+1)

class Flow3Func5AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        assert x == 7
        assert y == 9
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments and keyword arguments.
        self.ferry_to("end", x+1, y+1, user_input=user_input)
        # Note: the `end` worker will be a netsted Automa.

class Flow_3_Nested(GraphAutoma):
    ...

class Flow3NestedFunc1AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        assert x == 8
        assert y == 10
        assert user_input == "hi"
        return x+1, y+1

class Flow3NestedFunc2AsyncWorker(Worker):
    async def arun(self, x, y, user_input: str):
        assert x == 9
        assert y == 11
        assert user_input == "hi"
        return x, y

@pytest.fixture
def flow_3_nested():
    flow = Flow_3_Nested()
    flow.add_worker(
        "func_1",
        Flow3NestedFunc1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow3NestedFunc2AsyncWorker(),
        dependencies=["func_1"],
        is_output=True,
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    return flow

@pytest.fixture
def flow_3(flow_3_nested):
    flow = Flow_3()
    flow.add_worker(
        "start_1",
        Flow3Start1AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "start_2",
        Flow3Start2AsyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow3Func1AsyncWorker(),
        dependencies=["start_1", "start_2"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.add_worker(
        "func_2",
        Flow3Func2AsyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_3",
        Flow3Func3AsyncWorker(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_4",
        Flow3Func4AsyncWorker(),
    )
    flow.add_worker(
        "func_5",
        Flow3Func5AsyncWorker(),
    )
    flow.add_worker(
        "end",
        flow_3_nested,
        is_output=True
    )
    return flow

@pytest.mark.asyncio
async def test_flow_3(flow_3):
    result = await flow_3.arun(11, 22, x=2, y=3, user_input="hi")
    assert result == (9, 11)

###########################################################
##### Part Four: Test custom workers -- sync run() #####
###########################################################

class Flow_4(GraphAutoma):
    ...

class Flow4Start1SyncWorker(Worker):
    def run(self, a, b, /, x, user_input: str):
        # Test case for multiple start workers.
        assert x == 2
        assert user_input == "hi"
        return a, b, x + 1

class Flow4Start2SyncWorker(Worker):
    def run(self, a, b, /, y, user_input: str):
        # Test case for multiple start workers.
        assert y == 3
        assert user_input == "hi"
        return a, b, y + 2

class Flow4Func1SyncWorker(Worker):
    def run(self, my_list: Tuple[Any, Any], user_input: str):
        # Test case for args_mapping_rule=ArgsMappingRule.MERGE.
        start_1_output = my_list[0]
        start_2_output = my_list[1]
        assert start_1_output[0] == 11
        assert start_1_output[1] == 22
        assert start_2_output[0] == 11
        assert start_2_output[1] == 22
        assert user_input == "hi"
        x = start_1_output[2]
        y = start_2_output[2]
        assert x == 3
        assert y == 5
        return [x+1, y+1]

class Flow4Func2SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        # Test case for UNPACK list.
        assert x == 4
        assert y == 6
        assert user_input == "hi"
        return {"x": x+1, "y": y+1}

class Flow4Func3SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        # Test case for UNPACK dict.
        assert x == 5
        assert y == 7
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments.
        self.ferry_to("func_4", x+1, y+1)

class Flow4Func4SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        assert x == 6
        assert y == 8
        assert user_input == "hi"
        # Test case for ferry_to() with keyword arguments.
        self.ferry_to("func_5", x=x+1, y=y+1)

class Flow4Func5SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        assert x == 7
        assert y == 9
        assert user_input == "hi"
        # Test case for ferry_to() with positional arguments and keyword arguments.
        self.ferry_to("end", x+1, y+1, user_input=user_input)
        # Note: the `end` worker will be a netsted Automa.

class Flow_4_Nested(GraphAutoma):
    ...

class Flow4NestedFunc1SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        assert x == 8
        assert y == 10
        assert user_input == "hi"
        return x+1, y+1

class Flow4NestedFunc2SyncWorker(Worker):
    def run(self, x, y, user_input: str):
        assert x == 9
        assert y == 11
        assert user_input == "hi"
        return x, y

@pytest.fixture
def flow_4_nested():
    flow = Flow_4_Nested()
    flow.add_worker(
        "func_1",
        Flow4NestedFunc1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_2",
        Flow4NestedFunc2SyncWorker(),
        dependencies=["func_1"],
        is_output=True,
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    return flow

@pytest.fixture
def flow_4(flow_4_nested):
    flow = Flow_4()
    flow.add_worker(
        "start_1",
        Flow4Start1SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "start_2",
        Flow4Start2SyncWorker(),
        is_start=True,
    )
    flow.add_worker(
        "func_1",
        Flow4Func1SyncWorker(),
        dependencies=["start_1", "start_2"],
        args_mapping_rule=ArgsMappingRule.MERGE,
    )
    flow.add_worker(
        "func_2",
        Flow4Func2SyncWorker(),
        dependencies=["func_1"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_3",
        Flow4Func3SyncWorker(),
        dependencies=["func_2"],
        args_mapping_rule=ArgsMappingRule.UNPACK,
    )
    flow.add_worker(
        "func_4",
        Flow4Func4SyncWorker(),
    )
    flow.add_worker(
        "func_5",
        Flow4Func5SyncWorker(),
    )
    flow.add_worker(
        "end",
        flow_4_nested,
        is_output=True
    )
    return flow

@pytest.mark.asyncio
async def test_flow_4(flow_4):
    result = await flow_4.arun(11, 22, x=2, y=3, user_input="hi")
    assert result == (9, 11)

###########################################################
##### Part Five: Test Nested Automa propagation #####
###########################################################


class GlobalLogCallback(WorkerCallback):
    def __init__(self, tag: str = None):
        self._tag = tag or ""

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        if key == "nested_automa_as_worker":
            assert 11 in arguments["args"]
            assert "x" in arguments["kwargs"]
            assert arguments["kwargs"]["x"] == 10
        elif key == "inner_worker":
            assert 11 in arguments["args"]
            assert "x" not in arguments["kwargs"]

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional[GraphAutoma] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        if key == "nested_automa_as_worker":
            assert 11 in arguments["args"]
            assert "x" in arguments["kwargs"]
            assert arguments["kwargs"]["x"] == 10
        elif key == "inner_worker":
            assert 11 in arguments["args"]
            assert "x" not in arguments["kwargs"]


@pytest.fixture
def graph_with_global_setting_inputs_propagation():
    # Top-level automa
    class TopAutoma(GraphAutoma):
        @worker(is_start=True)
        async def top_worker(self, x: int) -> int:
            return x + 1

    # Inner automa (will be used as a nested worker)
    class InnerAutoma(GraphAutoma):
        @worker(is_start=True, is_output=True)
        async def inner_worker(self, x: int) -> int:
            return x * 2

    # Configure callback at global setting, with <Global> tag.
    GlobalSetting.set(
        callback_builders=[
            WorkerCallbackBuilder(GlobalLogCallback, init_kwargs={"tag": "<Global>"}),
        ]
    )

    # Configure callback at top-level automa, with <Automa> tag.
    running_options = RunningOptions(
        callback_builders=[
            WorkerCallbackBuilder(GlobalLogCallback, init_kwargs={"tag": "<Automa>"})
        ]
    )
    automa = TopAutoma(name="top-automa", running_options=running_options)

    # Add a instance of InnerAutoma as a worker.
    automa.add_worker("nested_automa_as_worker", InnerAutoma(name="inner-automa"), dependencies=["top_worker"], is_output=True)

    return automa

@pytest.mark.asyncio
async def test_global_setting_callback_inputs_propagation(graph_with_global_setting_inputs_propagation: GraphAutoma):
    result = await graph_with_global_setting_inputs_propagation.arun(x=10)
    assert result == 22

    # Clean up: reset global setting
    GlobalSetting.set(callback_builders=[])