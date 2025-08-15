from pydantic import BaseModel

class Snapshot(BaseModel):
    """
    Snapshot is a class that represents the current state of an Automa.
    """
    serialized_bytes: bytes
    serialization_version: str
