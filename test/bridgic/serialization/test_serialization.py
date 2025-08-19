import msgpack
from typing import Dict, List, Any
import pytest
from bridgic.automa.worker import Worker
from bridgic.serialization import Serializable, Picklable
import bridgic.serialization.msgpackx as msgpackx
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

def test_basic_types_serialization():
    # str serialization test
    obj: str = "Hello, Bridgic!"
    expected_data = msgpack.packb(obj)
    assert msgpackx.dumps(obj) == expected_data
    # str deserialization test
    assert msgpackx.loads(expected_data) == obj

    # bytes serialization test
    obj1: bytes = b'Hello, Bridgic!'
    expected_data = msgpack.packb(obj1)
    assert msgpackx.dumps(obj1) == expected_data
    # bytes deserialization test
    assert msgpackx.loads(expected_data) == obj1

    #bytearray serialization test
    obj2: bytearray = bytearray(b'Hello, Bridgic!')
    expected_data = msgpack.packb(obj2)
    assert msgpackx.dumps(obj2) == expected_data
    # bytearray deserialization test
    assert msgpackx.loads(expected_data) == obj2

    # dict serialization test
    obj3: Dict = {
        "a": 1,
        "b": {
            "b1": 2,
            "b2": [3, 4, 5],
        },
        "c": "Hello, Bridgic!",
    }
    expected_data = msgpack.packb(obj3)
    assert msgpackx.dumps(obj3) == expected_data
    # dict deserialization test
    assert msgpackx.loads(expected_data) == obj3

    # list serialization test
    obj4: List = [1, 2, {"a": 1, "b": 2}, 4, 5]
    expected_data = msgpack.packb(obj4)
    assert msgpackx.dumps(obj4) == expected_data
    # list deserialization test
    assert msgpackx.loads(expected_data) == obj4

    # None serialization test
    obj5: None = None
    expected_data = msgpack.packb(obj5)
    assert msgpackx.dumps(obj5) == expected_data
    # None deserialization test
    assert msgpackx.loads(expected_data) is None

def test_datetime_serialization():
    # A datetime object with timezone
    # 2025-08-18 01:31:45.564287+08:00
    dt1: datetime = datetime(2025, 8, 18, 1, 31, 45, microsecond=564287, tzinfo=timezone(timedelta(hours=8)), fold=0)
    data = msgpackx.dumps(dt1)
    assert msgpackx.loads(data) == dt1

    # A datetime object without timezone
    dt2 = datetime.today()
    data = msgpackx.dumps(dt2)
    assert msgpackx.loads(data) == dt2

# Test a custom Serializable object
class MyWorker1(Worker, Serializable):
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "outbuf": self.output_buffer,
            "local_space": self.local_space,
        }
    
    @classmethod
    def load_from_dict(cls, state_dict: Dict[str, Any]) -> "MyWorker1":
        w = MyWorker1()
        w.output_buffer = state_dict["outbuf"]
        w.local_space = state_dict["local_space"]
        return w


@pytest.fixture
def worker_a():
    w = MyWorker1()
    # Initialize just for test
    w.output_buffer = "Hello, Bridgic in Output Buffer!"
    w.local_space = {"a": 1, "b": 2}
    return w

def test_custom_serialization(worker_a: MyWorker1):
    # Test a pure Serializable object
    data = msgpackx.dumps(worker_a)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
    assert type(obj) is MyWorker1
    assert obj.output_buffer == worker_a.output_buffer
    assert obj.local_space == worker_a.local_space

    # Test a json including a Serializable object
    orig = {
        "key1": 11,
        "key2": worker_a,
    }
    data = msgpackx.dumps(orig)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
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

def test_pickle_serialization(worker_a: MyWorker1, worker_b: MyWorker2):
    # Test a pure Picklable object
    data = msgpackx.dumps(worker_b)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
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
    data = msgpackx.dumps(orig)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
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

def test_unsupported_serialization(worker_c: MyWorker3):
    with pytest.raises(TypeError, match="can not serialize"):
        msgpackx.dumps(worker_c)

def test_pickle_fallback(worker_c: MyWorker3):
    data = msgpackx.dumps(worker_c, pickle_fallback=True)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
    assert type(obj) is MyWorker3
    assert obj.output_buffer == worker_c.output_buffer
    assert obj.local_space == worker_c.local_space

# Test a Pydantic BaseModel object
class Dog(BaseModel):
    name: str
    age: int

class Person(BaseModel):
    name: str
    age: int
    birthday: datetime

def test_pydantic_serialization():
    # Test a pure Pydantic BaseModel object
    dog = Dog(name="Buddy", age=3)
    data = msgpackx.dumps(dog)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
    assert type(obj) is Dog
    assert obj.name == dog.name
    assert obj.age == dog.age

    # Test Pydantic BaseModel object including a datetime object
    person = Person(name="John", age=30, birthday=datetime(2010, 3, 12, 12, 33, 0, tzinfo=timezone(timedelta(hours=8))))
    data = msgpackx.dumps(person)
    assert type(data) is bytes
    obj = msgpackx.loads(data)
    assert type(obj) is Person
    assert obj.name == person.name
    assert obj.age == person.age
    assert obj.birthday == person.birthday
