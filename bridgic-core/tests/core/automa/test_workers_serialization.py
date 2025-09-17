"""
Test cases for serializing and deserializing various types of workers, using msgpackx.
"""

import pytest
from typing import Optional, Dict, Any
from typing_extensions import override
from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
from bridgic.core.automa.worker import Worker, CallableWorker
from bridgic.core.serialization import msgpackx
from bridgic.core.types.error import WorkerRuntimeError

################## Test cases for Basic Worker && Customized Worker ####################

@pytest.fixture
def worker_1():
    # Test a basic Worker object.
    w = Worker()
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    return w

def test_basic_worker_serialization(worker_1: Worker):
    data = msgpackx.dump_bytes(worker_1)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is Worker
    assert obj.output_buffer == "Hello, Bridgic in Output Buffer!"

class MyCustomWorker(Worker):
    # Custom fields needs to be processed to support serialization.
    _x: int
    _y: int

    def __init__(
            self, 
            x: int = 0,
            y: int = 0,
        ):
        super().__init__()
        self._x = x
        self._y = y
    
    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["x"] = self._x
        state_dict["y"] = self._y
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._x = state_dict["x"]
        self._y = state_dict["y"]
   
@pytest.fixture
def worker_2():
    # Test a customized Worker object.
    w = MyCustomWorker(x=11, y=23)
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    return w

def test_customized_worker_serialization(worker_2: MyCustomWorker):
    data = msgpackx.dump_bytes(worker_2)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is MyCustomWorker
    assert obj.output_buffer == "Hello, Bridgic in Output Buffer!"
    assert obj._x == 11
    assert obj._y == 23

################## Test cases for Callable Worker ####################

def func_a():
    pass

@pytest.fixture
def worker_3():
    # Test a CallableWorker with a normal function.
    w = CallableWorker(func_a)
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    return w

def test_callable_worker_serialization_1(worker_3: CallableWorker):
    data = msgpackx.dump_bytes(worker_3)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is CallableWorker
    assert obj.output_buffer == "Hello, Bridgic in Output Buffer!"
    assert obj.callable is func_a

class MyClass:
    def func_a(self):
        pass

@pytest.fixture
def my_obj():
    return MyClass()

@pytest.fixture
def worker_4():
    # Test a CallableWorker with a unbound method of a class.
    w = CallableWorker(MyClass.func_a)
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    return w

def test_callable_worker_serialization_2(worker_4: CallableWorker):
    data = msgpackx.dump_bytes(worker_4)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is CallableWorker
    assert obj.output_buffer == "Hello, Bridgic in Output Buffer!"
    assert obj.callable is MyClass.func_a

@pytest.fixture
def worker_5(my_obj: MyClass):
    # Test a CallableWorker with a bound method of a class.
    w = CallableWorker(my_obj.func_a)
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    return w

def test_callable_worker_serialization_3(worker_5: CallableWorker, my_obj: MyClass):
    data = msgpackx.dump_bytes(worker_5)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is CallableWorker
    assert obj.output_buffer == "Hello, Bridgic in Output Buffer!"
    # Note: the bounded object is not the same as the original object.
    assert obj.callable != my_obj.func_a
    assert obj.callable.__func__ is my_obj.func_a.__func__
    assert obj.callable.__self__ != my_obj

class TopAutoma(GraphAutoma):
    @worker(is_start=True)
    async def add1(self, x: int):
        return x + 1
    
    async def add2(self, x: int):
        return x + 2

@pytest.fixture
def top_automa():
    return TopAutoma()

@pytest.fixture
def worker_6(top_automa: TopAutoma):
    # Test a CallableWorker with a parent of Automa.
    w = CallableWorker(top_automa.add2)
    top_automa.add_worker("add2", w, dependencies=["add1"])
    top_automa.output_worker_key = "add2"
    return w

@pytest.fixture
def worker_6_partially_deserialized(worker_6: CallableWorker, top_automa: TopAutoma):
    data = msgpackx.dump_bytes(worker_6)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is CallableWorker
    # Test partially deserialized data in the CallableWorker.
    assert obj._expected_bound_parent is True
    assert obj.callable is TopAutoma.add2
    return obj

@pytest.mark.asyncio
async def test_callable_worker_serialization_4(worker_6_partially_deserialized: CallableWorker):
    with pytest.raises(WorkerRuntimeError, match="not bounded yet"):
        await worker_6_partially_deserialized.arun(x=90)

@pytest.fixture
def automa_and_worker(worker_6_partially_deserialized: CallableWorker):
    top_automa2 = TopAutoma()
    top_automa2.add_worker("add2", worker_6_partially_deserialized, dependencies=["add1"])
    top_automa2.output_worker_key = "add2"
    # Fully deserialized after being added to a Automa.
    worker_6_deserialized = worker_6_partially_deserialized
    return top_automa2, worker_6_deserialized

@pytest.mark.asyncio
async def test_callable_worker_serialization_5(automa_and_worker):
    top_automa2, worker_6_deserialized = automa_and_worker
    assert worker_6_deserialized._expected_bound_parent is False
    # assert worker_6_deserialized.callable is top_automa2.add2 # TODO: fix this maybe later
    assert worker_6_deserialized.callable.__self__ is top_automa2
    result = await top_automa2.arun(x=90)
    assert result == 93

################## Test cases GraphAutoma ####################

class AdderAutoma(GraphAutoma):
    @worker(is_start=True)
    async def add1(self, x: int, y: int):
        return {
            "x": x + 1,
            "y": y + 1
        }
    
    @worker(dependencies=["add1"], args_mapping_rule=ArgsMappingRule.UNPACK)
    async def add2(self, x: int, y: int):
        return x + 2, y + 2

@pytest.fixture
def adder_automa():
    return AdderAutoma(output_worker_key="add2")

@pytest.fixture
def deserialized_adder_automa(adder_automa: AdderAutoma):
    data = msgpackx.dump_bytes(adder_automa)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is AdderAutoma
    return obj

@pytest.mark.asyncio
async def test_automa_serialization(deserialized_adder_automa: AdderAutoma):
    result = await deserialized_adder_automa.arun(x=10, y=20)
    assert result == (13, 23)