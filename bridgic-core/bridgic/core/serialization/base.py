from typing import Protocol, TypeVar, runtime_checkable, Dict, Any
from abc import abstractmethod

T = TypeVar('T', covariant=True)

@runtime_checkable
class Serializable(Protocol[T]):
    """
    Serializable is a protocol that defines the interface for objects that customizes serialization.
    The type parameter T is the type of the object to be serialized and deserialized, typically the subclass itself that implements Serializable.
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
        Load the object state from a dictionary, which is previously deserialized from bytes.
        """
        ...

@runtime_checkable
class Picklable(Protocol):
    """
    Picklable is a protocol that defines the interface for objects that can be serialized and deserialized using pickle.

    If a class implements both Serializable and Picklable, the object of the class will be serialized using the implementation provided by Serializable.
    """

    def __picklable_marker__(self) -> None:
        """
        This is just a marker method to distinguish Picklable implementations.
        Since it is not necessary to implement this method in the subclass, no @abstractmethod is used here.
        """
        ...
