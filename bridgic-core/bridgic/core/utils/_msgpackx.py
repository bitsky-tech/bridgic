"""
This module provides a serialization mechanism, which extends default msgpack serialization.

The strategy it will adopt the serialization and deserialization strategies is in the following order:

1. Checking if the data type belongs to common basic data types, and if so, using the special implementation for it.
2. Checking if the data type implements Serializable protocol, and if so, using its customized implementation.
3. Checking if the data type implements Picklable protocol, and if so, using its pickle's implementation.
4. Otherwise, the serialization behavior is determined by the pickle_fallback parameter.
    - If pickle_fallback is True, the data will be serialized / deserialized using pickle.
    - If pickle_fallback is False, serialization / deserialization will be tried by msgpack, which may raise a TypeError if failed.

See more about [Python's common basic data types](https://docs.python.org/3/library/datatypes.html).
"""
import msgpack # type: ignore
import pickle

from typing import Any, Optional
from enum import Enum
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func
from bridgic.core.types._serialization import Serializable, Picklable
from datetime import datetime
from pydantic import BaseModel

# TODO: It may be supported for more data types in the future.

def dump_bytes(obj: Any, pickle_fallback: bool = False) -> bytes:
    def _custom_encode(obj: Any) -> Any:
        ser_type: Optional[str] = None
        ser_data: Optional[bytes] = None
        obj_type: Optional[str] = None
        # If both Serializable and Picklable are implemented, prefer using the implementation of Serializable.
        if isinstance(obj, datetime):
            # TODO: Try serializing using Unix timestamp + timezone offset, and check if there is any loss of precision
            ser_type = "datetime"
            ser_data = obj.isoformat()
        elif isinstance(obj, BaseModel):
            ser_type = "pydantic"
            ser_data = obj.model_dump()
            obj_type = type(obj).__module__ + "." + type(obj).__qualname__
        elif isinstance(obj, Enum):
            ser_type = "enum"
            ser_data = obj.value
            obj_type = type(obj).__module__ + "." + type(obj).__qualname__
        elif isinstance(obj, set):
            # msgpack does not support set natively, so we need to convert it to a list.
            ser_type = "set"
            ser_data = list(obj)
        elif hasattr(obj, "dump_to_dict") and hasattr(obj, "load_from_dict"):
            # Use hasattr() instead of isinstance(obj, Serializable) for performance reasons.
            # Refer to: https://docs.python.org/3/library/typing.html#typing.runtime_checkable
            ser_type = type(obj).__module__ + "." + type(obj).__qualname__
            ser_data = obj.dump_to_dict()
        elif pickle_fallback or hasattr(obj, "__picklable_marker__"):
            # The type information is INCLUDED in the serialized data when pickle is used.
            ser_type = "pickled"
            ser_data = pickle.dumps(obj)
        
        if ser_type is not None and ser_data is not None:
            obj_dict = {
                "t": ser_type,
                "d": ser_data
            }
            if obj_type is not None:
                obj_dict["ot"] = obj_type
            return obj_dict
        return obj

    return msgpack.packb(obj, default=_custom_encode)

def load_bytes(data: bytes) -> Any:
    def _custom_decode(dict_obj: Any) -> Any:
        if "t" in dict_obj and "d" in dict_obj:
            if dict_obj["t"] == "datetime":
                return datetime.fromisoformat(dict_obj["d"])
            elif dict_obj["t"] == "pydantic":
                qualified_class_name = dict_obj["ot"]
                cls: BaseModel = load_qualified_class_or_func(qualified_class_name)
                return cls.model_validate(dict_obj["d"])
            elif dict_obj["t"] == "enum":
                qualified_class_name = dict_obj["ot"]
                cls: BaseModel = load_qualified_class_or_func(qualified_class_name)
                return cls(dict_obj["d"])
            elif dict_obj["t"] == "set":
                # list => set
                return set(dict_obj["d"])
            elif dict_obj["t"] == "pickled":
                return pickle.loads(dict_obj["d"])
            else:
                # Serializable is assumed here
                qualified_class_name = dict_obj["t"]
                cls: Serializable = load_qualified_class_or_func(qualified_class_name)
                # Note: Use the __init__ method with all default arguments to initialize the object.
                obj = cls.__new__(cls)
                obj.load_from_dict(dict_obj["d"])
                return obj
        return dict_obj

    return msgpack.unpackb(data, object_hook=_custom_decode)
