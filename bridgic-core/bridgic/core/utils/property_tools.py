from typing import Callable, TypeVar

T = TypeVar("T")

def property_decorator(func: Callable, initial_value: T) -> property:
    attr_name = f"_{func.__name__}"
    
    def getter(self) -> T:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, initial_value() if callable(initial_value) else initial_value)
        return getattr(self, attr_name)
    
    def setter(self, value: T):
        setattr(self, attr_name, value)
    
    return property(getter, setter)

def property_decorator_factory(obj: object, key: str, initial_value: T) -> None:
    """Create a property for an object instance.

    Args:
        obj (object): the object to create a property for.
        key (str): the name of the property.
        initial_value (T): the initial value of the property.

    Returns:
        None
    """
    attr_name = f"__{key}"
    
    def getter(self) -> T:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, initial_value() if callable(initial_value) else initial_value)
        return getattr(self, attr_name)
    
    def setter(self, value: T):
        setattr(self, attr_name, value)
    
    # Create property and set it on the object's class
    prop = property(getter, setter)
    setattr(obj.__class__, key, prop)

def create_property_factory(obj: object, key: str, initial_value: T) -> None:
    """Create a property for an object instance.

    Args:
        obj (object): the object to create a property for.
        key (str): the name of the property.
        initial_value (T): the initial value of the property.

    Returns:
        None
    """
    property_decorator_factory(obj, key, initial_value)
