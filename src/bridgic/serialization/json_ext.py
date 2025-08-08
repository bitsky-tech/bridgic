from typing import Any, Optional
import msgpack # type: ignore
from .base import Serializable, Picklable
import pickle
from bridgic.utils.inspect_tools import load_qualified_class

class JsonExtSerializer:
    """
    JsonExtSerializer is a serializer that mainly uses msgpack to serialize the data.
    It allows to customize the serialization method of the data type that is not currently supported by msgpack.

    If the data type implements Serializable, the customized serialization method will be used.
    If the data type implements Picklable, the data will be serialized using pickle.
    If the data type does not implement Serializable or Picklable, the serialization behavior is determined by the pickle_fallback parameter. If pickle_fallback is True, the data will be serialized using pickle. Otherwise, serialization will be tried by msgpack, which may raise a TypeError if failed.

    TODO: Add default serialization support for some common data types. Which data types should be supported?
    Ref: https://docs.python.org/3/library/datatypes.html
    """
    
    def __init__(self, pickle_fallback: bool = False):
        self.pickle_fallback = pickle_fallback

    def custom_encode(self, obj: Any) -> Any:
        ser_type: Optional[str] = None
        ser_data: Optional[bytes] = None
        # If both Serializable and Picklable are implemented, prefer using the implementation of Serializable.
        if isinstance(obj, Serializable):
            ser_type = type(obj).__module__ + "." + type(obj).__qualname__
            ser_data = obj.dumps()
        elif self.pickle_fallback or isinstance(obj, Picklable):
            # The type information is INCLUDED in the serialized data when pickle is used.
            ser_type = "pickled"
            ser_data = pickle.dumps(obj)
        
        if ser_type is not None and ser_data is not None:
            return {
                "t": ser_type,
                "b": ser_data
            }
        return obj

    def custom_decode(self, obj: Any) -> Any:
        if "t" in obj and "b" in obj:
            if obj["t"] == "pickled":
                return pickle.loads(obj["b"])
            else:
                # Serializable is assumed here
                qualified_class_name = obj["t"]
                cls: Serializable = load_qualified_class(qualified_class_name)
                return cls.loads(obj["b"])
        return obj

    def dumps(self, obj: Any) -> bytes:
        return msgpack.packb(obj, default=self.custom_encode)

    def loads(self, data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=self.custom_decode)

