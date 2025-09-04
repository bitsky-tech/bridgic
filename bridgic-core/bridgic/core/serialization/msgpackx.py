"""
msgpackx is a serialization library that extends msgpack to provide enhanced data serialization capabilities.
It supports serializing various common data types that are not natively handled by msgpack, and also allows customization of serialization methods for different data types.

If the data type implements Serializable, the customized serialization method will be used.
If the data type implements Picklable, the data will be serialized using pickle.
If the data type does not implement Serializable or Picklable, the serialization behavior is determined by the pickle_fallback parameter. If pickle_fallback is True, the data will be serialized using pickle. Otherwise, serialization will be tried by msgpack, which may raise a TypeError if failed.

TODO: Add default serialization support for some common data types. Which data types should be supported?
Ref: https://docs.python.org/3/library/datatypes.html
"""
from typing import Any, Optional
from enum import Enum
import msgpack # type: ignore
from .base import Serializable, Picklable
import pickle
from bridgic.core.utils.inspect_tools import load_qualified_class_or_func
from datetime import datetime
from pydantic import BaseModel

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
    def _custom_decode(obj: Any) -> Any:
        if "t" in obj and "d" in obj:
            if obj["t"] == "datetime":
                return datetime.fromisoformat(obj["d"])
            elif obj["t"] == "pydantic":
                qualified_class_name = obj["ot"]
                cls: BaseModel = load_qualified_class_or_func(qualified_class_name)
                return cls.model_validate(obj["d"])
            elif obj["t"] == "enum":
                qualified_class_name = obj["ot"]
                cls: BaseModel = load_qualified_class_or_func(qualified_class_name)
                return cls(obj["d"])
            elif obj["t"] == "set":
                # list => set
                return set(obj["d"])
            elif obj["t"] == "pickled":
                return pickle.loads(obj["d"])
            else:
                # Serializable is assumed here
                qualified_class_name = obj["t"]
                cls: Serializable = load_qualified_class_or_func(qualified_class_name)
                return cls.load_from_dict(obj["d"])
        return obj

    return msgpack.unpackb(data, object_hook=_custom_decode)
