from pydantic import BaseModel
from typing import Protocol, runtime_checkable, Dict, Any
from abc import abstractmethod

@runtime_checkable
class Serializable(Protocol):
    """
    Serializable is a protocol that defines the interfaces that customizes serialization.
    """
    @abstractmethod
    def dump_to_dict(self) -> Dict[str, Any]:
        """
        Dump the object to a dictionary, which will finally be serialized to bytes.
        """
        ...

    @abstractmethod
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the object state from a dictionary previously obtained by deserializing from bytes.
        """
        ...

@runtime_checkable
class Picklable(Protocol):
    """
    Picklable is a protocol that defines the interfaces that customizes serialization using pickle.

    Notes
    -----
    If a class implements both Serializable and Picklable, the object of the class will be 
    serialized using the implementation provided by Serializable instead of using pickle.
    """

    def __picklable_marker__(self) -> None:
        """
        This is just a marker method to distinguish Picklable objects from other objects.
        Since it is not necessary to implement this method in the subclass, thus no 
        @abstractmethod is used here.
        """
        ...
