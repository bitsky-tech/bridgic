"""
Test cases for serializing and deserializing various types of workers, using msgpackx.
"""

import pytest
from typing import Optional, Dict, Any
from typing_extensions import override
from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
from bridgic.core.automa.worker import Worker, CallableWorker
from bridgic.core.utils import msgpackx
from bridgic.core.types.error import WorkerRuntimeError

################## Test cases for Customized Worker ####################

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