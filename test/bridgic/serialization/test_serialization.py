import msgpack
from typing import Dict, List
import pytest
from bridgic.automa.worker import Worker
from bridgic.serialization import Serializable, Picklable, JsonExtSerializer
import json

@pytest.fixture
def serder():
    return JsonExtSerializer()

def test_basic_types_serialization(serder: JsonExtSerializer):
    # str serialization test
    obj: str = "Hello, Bridgic!"
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # str deserialization test
    assert serder.loads(expected_data) == obj

    # bytes serialization test
    obj: bytes = b'Hello, Bridgic!'
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # bytes deserialization test
    assert serder.loads(expected_data) == obj

    #bytearray serialization test
    obj: bytearray = bytearray(b'Hello, Bridgic!')
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # bytearray deserialization test
    assert serder.loads(expected_data) == obj

    # dict serialization test
    obj: Dict = {
        "a": 1,
        "b": {
            "b1": 2,
            "b2": [3, 4, 5],
        },
        "c": "Hello, Bridgic!",
    }
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # dict deserialization test
    assert serder.loads(expected_data) == obj

    # list serialization test
    obj: List = [1, 2, {"a": 1, "b": 2}, 4, 5]
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # list deserialization test
    assert serder.loads(expected_data) == obj

    # None serialization test
    obj: None = None
    expected_data = msgpack.packb(obj)
    assert serder.dumps(obj) == expected_data
    # None deserialization test
    assert serder.loads(expected_data) is None

# Test a custom Serializable object
class MyWorker1(Worker, Serializable):
    def dumps(self) -> bytes:
        return json.dumps({
            "outbuf": self.output_buffer,
            "local_space": self.local_space,
        }).encode("utf-8")
    
    @classmethod
    def loads(cls, data: bytes) -> "MyWorker1":
        obj_dict = json.loads(data.decode("utf-8"))
        w = MyWorker1()
        w.output_buffer = obj_dict["outbuf"]
        w.local_space = obj_dict["local_space"]
        return w


@pytest.fixture
def worker_a():
    w = MyWorker1()
    # Initialize just for test
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    w.local_space = {"a": 1, "b": 2}
    return w

def test_custom_serialization(serder: JsonExtSerializer, worker_a: MyWorker1):
    # Test a pure Serializable object
    data = serder.dumps(worker_a)
    assert type(data) is bytes
    obj = serder.loads(data)
    assert type(obj) is MyWorker1
    assert obj.output_buffer == worker_a.output_buffer
    assert obj.local_space == worker_a.local_space

    # Test a json including a Serializable object
    orig = {
        "key1": 11,
        "key2": worker_a,
    }
    data = serder.dumps(orig)
    assert type(data) is bytes
    obj = serder.loads(data)
    assert type(obj) is dict
    assert obj["key1"] == orig["key1"]
    assert obj["key2"].output_buffer == orig["key2"].output_buffer
    assert obj["key2"].local_space == orig["key2"].local_space

# Test a Picklable object
class MyWorker2(Worker, Picklable):
    pass

@pytest.fixture
def worker_b():
    w = MyWorker2()
    # Initialize just for test
    w.output_buffer = ["Hello, Bridgic in Output Buffer!", "(Picklable)"]
    w.local_space = {"x": 100, "y": 333, "c": "Hello, Bridgic!"}
    return w

def test_pickle_serialization(serder: JsonExtSerializer, worker_a: MyWorker1, worker_b: MyWorker2):
    # Test a pure Picklable object
    data = serder.dumps(worker_b)
    assert type(data) is bytes
    obj = serder.loads(data)
    assert type(obj) is MyWorker2
    assert obj.output_buffer == worker_b.output_buffer
    assert obj.local_space == worker_b.local_space

    # Test a json including a Picklable object and a Serializable object!!
    orig = {
        "key1": {
            "a": 1,
            "b": worker_b,
        },
        "key2": worker_a,
    }
    data = serder.dumps(orig)
    assert type(data) is bytes
    obj = serder.loads(data)
    assert type(obj) is dict
    assert obj["key1"]["a"] == orig["key1"]["a"]
    assert obj["key1"]["b"].output_buffer == orig["key1"]["b"].output_buffer
    assert obj["key1"]["b"].local_space == orig["key1"]["b"].local_space
    assert obj["key2"].output_buffer == orig["key2"].output_buffer
    assert obj["key2"].local_space == orig["key2"].local_space

# Test an object whose serialization is not supported.
# Neither Serializable nor Picklable is implemented.
class MyWorker3(Worker):
    pass

@pytest.fixture
def worker_c():
    w = MyWorker3()
    # Initialize just for test
    w.output_buffer = "Not serializable"
    w.local_space = {"x": 100, "y": 333, "c": "Hello, Bridgic!"}
    return w

def test_unsupported_serialization(serder: JsonExtSerializer, worker_c: MyWorker3):
    with pytest.raises(TypeError, match="can not serialize"):
        serder.dumps(worker_c)

@pytest.fixture
def serder2():
    return JsonExtSerializer(pickle_fallback=True)

def test_pickle_fallback(serder2: JsonExtSerializer, worker_c: MyWorker3):
    data = serder2.dumps(worker_c)
    assert type(data) is bytes
    obj = serder2.loads(data)
    assert type(obj) is MyWorker3
    assert obj.output_buffer == worker_c.output_buffer
    assert obj.local_space == worker_c.local_space
