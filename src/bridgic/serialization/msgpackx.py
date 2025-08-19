"""
JsonExtSerializer is a serializer that mainly uses msgpack to serialize the data.
It allows to customize the serialization method of the data type that is not currently supported by msgpack.

If the data type implements Serializable, the customized serialization method will be used.
If the data type implements Picklable, the data will be serialized using pickle.
If the data type does not implement Serializable or Picklable, the serialization behavior is determined by the pickle_fallback parameter. If pickle_fallback is True, the data will be serialized using pickle. Otherwise, serialization will be tried by msgpack, which may raise a TypeError if failed.

TODO: Add default serialization support for some common data types. Which data types should be supported?
Ref: https://docs.python.org/3/library/datatypes.html
"""
from typing import Any, Optional
import msgpack # type: ignore
from .base import Serializable, Picklable
import pickle
from bridgic.utils.inspect_tools import load_qualified_class
from datetime import datetime
from pydantic import BaseModel

def dumps(obj: Any, pickle_fallback: bool = False) -> bytes:
    def _custom_encode(obj: Any) -> Any:
        ser_type: Optional[str] = None
        ser_data: Optional[bytes] = None
        model_type: Optional[str] = None
        # If both Serializable and Picklable are implemented, prefer using the implementation of Serializable.
        if isinstance(obj, datetime):
            # TODO: Try serializing using Unix timestamp + timezone offset, and check if there is any loss of precision
            ser_type = "datetime"
            ser_data = obj.isoformat()
        elif isinstance(obj, BaseModel):
            ser_type = "pydantic"
            ser_data = obj.model_dump()
            model_type = type(obj).__module__ + "." + type(obj).__qualname__
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
            if model_type is not None:
                obj_dict["m"] = model_type
            return obj_dict
        return obj

    return msgpack.packb(obj, default=_custom_encode)

def loads(data: bytes) -> Any:
    def _custom_decode(obj: Any) -> Any:
        if "t" in obj and "d" in obj:
            if obj["t"] == "datetime":
                return datetime.fromisoformat(obj["d"])
            elif obj["t"] == "pydantic":
                qualified_class_name = obj["m"]
                cls: BaseModel = load_qualified_class(qualified_class_name)
                return cls.model_validate(obj["d"])
            elif obj["t"] == "pickled":
                return pickle.loads(obj["d"])
            else:
                # Serializable is assumed here
                qualified_class_name = obj["t"]
                cls: Serializable = load_qualified_class(qualified_class_name)
                return cls.load_from_dict(obj["d"])
        return obj

    return msgpack.unpackb(data, object_hook=_custom_decode)
