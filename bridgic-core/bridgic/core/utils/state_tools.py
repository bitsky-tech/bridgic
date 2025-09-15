from typing import TypeVar, Generic, Callable, Tuple, Any, Type, Union, Protocol, runtime_checkable, overload, cast, List, Dict
import sys
import threading

# define type variables
T = TypeVar('T')


@runtime_checkable
class StateAccessorProtocol(Protocol[T]):
    """A protocol for state accessors.

    Args:
        Protocol (_type_): A protocol for state accessors.
    """
    @property
    def value(self) -> T:
        """The value of the state.

        Returns:
            T: The value of the state.
        """
        ...
    @value.setter
    def value(self, value: T) -> None:
        """Set the value of the state

        Args:
            value (T): The value of the state.
        """
        ...

StateAccessorType = TypeVar("StateAccessorType", bound=Any)
StateSetter = Callable[[T], None]

def useState(initial_value: T) -> Tuple[T, Callable[[T], None]]:
    """
    A React-like state utility.
    
    Overview:
    - Returns a pair `(state, setState)`.
      - `state` is a dynamic accessor object that behaves like the original value type
        (its class name and module mirror the original type). Most operators, indexing,
        iteration, formatting, attribute and method access are delegated to the internal
        underlying value. Reading `state.value` returns the current underlying value;
        writing `state.value = x` is equivalent to calling `setState(x)`.
      - `setState(new_value)` replaces the underlying value. Type changes are allowed
        (for example, transitioning from `int` to `str`).
    - In-place mutations on mutable objects (e.g., list `append`, dict item assignment) are
      applied directly to the current underlying value because calls are delegated to it.
      Use `setState(new_value)` when you need to replace the whole value.
    
    Persistence and scope:
    - Fast path (with `@stateful`): when used inside a function decorated with `@stateful`,
      the state container is preserved per-thread and per-function, keyed by the call order
      index within that function (top-to-bottom, increasing index). Repeated calls reuse
      previously created state entries.
    - Slow path (without `@stateful`): the state container is preserved per call site using
      the tuple `(code object, line number, thread id)` as the key. Re-invoking from the same
      line in the same thread reuses the same state; different threads are isolated.
    - Thread isolation: in both paths, state is isolated per thread.
    
    Args:
        initial_value: The initial state value, of any type.
    
    Returns:
        (state, setState)
        - state: A dynamic accessor object that closely mimics the original value.
                 Supports arithmetic/comparison, indexing/iteration, formatting, and
                 common protocols such as context management when applicable.
        - setState: Function with signature `setState(new_value)` to update the underlying value.
    
    Examples:
        >>> count, setCount = useState(0)
        >>> count  # 0
        0
        >>> isinstance(count, int)
        True
        >>> setCount(5)
        >>> count
        5
        
        # Type can change
        >>> value, setValue = useState(42)
        >>> setValue(3.14)
        >>> (isinstance(value, float), value)
        (True, 3.14)
        >>> setValue("hello")
        >>> (isinstance(value, str), value)
        (True, 'hello')
        
        # In-place mutation affects the current underlying value
        >>> items, setItems = useState([1, 2])
        >>> items.append(3)
        >>> items
        [1, 2, 3]
        >>> setItems([9])
        >>> items
        [9]
        
        # Reuse state with @stateful (ordered by call index)
        >>> from bridgic.core.utils.state_tools import stateful
        >>> @stateful
        ... def counter():
        ...     c, setC = useState(0)
        ...     setC(c + 1)
        ...     return c
        >>> counter(); counter(); counter()
        1
        2
        3
        
        # Reuse state without @stateful (slow path by call site and thread)
        >>> def counter2():
        ...     c, setC = useState(0)
        ...     setC(c + 1)
        ...     return c
        >>> counter2(); counter2(); counter2()
        1
        2
        3
    """
    # Create a container object that contains the state
    class StateContainer:
        def __init__(self, value: T) -> None:
            self.value = value

    # Fast path: stateful context managed by decorator, avoids frame inspection entirely
    if not hasattr(useState, "_TLS"):
        setattr(useState, "_TLS", threading.local())
    _TLS = getattr(useState, "_TLS")

    state_container = None  # type: ignore
    in_stateful_context = hasattr(_TLS, "stack") and isinstance(_TLS.stack, list) and len(_TLS.stack) > 0
    if in_stateful_context:
        top = _TLS.stack[-1]
        registry: Dict[int, Any] = top["registry"]
        index: int = top["index"]
        if index in registry:
            state_container = registry[index]
        else:
            state_container = StateContainer(initial_value)
            registry[index] = state_container
        top["index"] = index + 1
    else:
        # Slow path: global registry for persistent states by call site (code object, lineno)
        if not hasattr(useState, "_STATE_REGISTRY"):
            setattr(useState, "_STATE_REGISTRY", {})
        _STATE_REGISTRY: Dict[Tuple[Any, int, int], Any] = getattr(useState, "_STATE_REGISTRY")

        # determine caller key with low overhead
        try:
            caller = sys._getframe(1)
            key = (caller.f_code, caller.f_lineno, threading.get_ident())
        except Exception:
            # fallback key when frame access fails; degrade gracefully
            key = (id(useState), id(initial_value), threading.get_ident())

        # fetch or create state container for this call site
        if key in _STATE_REGISTRY:
            state_container = _STATE_REGISTRY[key]
        else:
            state_container = StateContainer(initial_value)
            _STATE_REGISTRY[key] = state_container
    
    def setValue(new_value: T) -> None:
        """
        update the state value
        
        Args:
            new_value: The new state value, type must be the same as the initial value
        """
        state_container.value = new_value
    
    # Dynamically create a class with correct __name__ and __module__
    original_type = type(state_container.value)
    
    # create dynamic class
    def create_state_accessor_class():
        class StateAccessorBase:
            def __getattribute__(self, name):
                if name == 'value':
                    return state_container.value
                # for other attributes, delegate to the state value itself
                return getattr(state_container.value, name)
            
            def __setattr__(self, name, value):
                if name == 'value':
                    setValue(value)
                else:
                    # if the state value supports attribute setting, set it
                    setattr(state_container.value, name, value)
            
            def __str__(self):
                return str(state_container.value)
            
            def __repr__(self):
                return repr(state_container.value)
            
            def __int__(self):
                return int(state_container.value)
            
            def __float__(self):
                return float(state_container.value)
            
            def __add__(self, other):
                return state_container.value + other
            
            def __radd__(self, other):
                return other + state_container.value
            
            def __sub__(self, other):
                return state_container.value - other
            
            def __rsub__(self, other):
                return other - state_container.value
            
            def __mul__(self, other):
                return state_container.value * other
            
            def __rmul__(self, other):
                return other * state_container.value
            
            def __eq__(self, other):
                return state_container.value == other
            
            def __ne__(self, other):
                return state_container.value != other
            
            def __lt__(self, other):
                return state_container.value < other
            
            def __le__(self, other):
                return state_container.value <= other
            
            def __gt__(self, other):
                return state_container.value > other
            
            def __ge__(self, other):
                return state_container.value >= other
            
            def __len__(self):
                return len(state_container.value)
            
            def __getitem__(self, key):
                return state_container.value[key]
            
            def __setitem__(self, key, value):
                state_container.value[key] = value
            
            def __delitem__(self, key):
                del state_container.value[key]
            
            def __contains__(self, item):
                return item in state_container.value
            
            def __iter__(self):
                return iter(state_container.value)
            
            def __bool__(self):
                return bool(state_container.value)
            
            def __hash__(self):
                try:
                    return hash(state_container.value)
                except TypeError:
                    return hash(id(state_container.value))
            
            # Add more arithmetic operations support
            def __truediv__(self, other):
                return state_container.value / other
            
            def __rtruediv__(self, other):
                return other / state_container.value
            
            def __floordiv__(self, other):
                return state_container.value // other
            
            def __rfloordiv__(self, other):
                return other // state_container.value
            
            def __mod__(self, other):
                return state_container.value % other
            
            def __rmod__(self, other):
                return other % state_container.value
            
            def __pow__(self, other):
                return state_container.value ** other
            
            def __rpow__(self, other):
                return other ** state_container.value
            
            # Bitwise operations support
            def __and__(self, other):
                return state_container.value & other
            
            def __rand__(self, other):
                return other & state_container.value
            
            def __or__(self, other):
                return state_container.value | other
            
            def __ror__(self, other):
                return other | state_container.value
            
            def __xor__(self, other):
                return state_container.value ^ other
            
            def __rxor__(self, other):
                return other ^ state_container.value
            
            # Unary operators
            def __neg__(self):
                return -state_container.value
            
            def __pos__(self):
                return +state_container.value
            
            def __abs__(self):
                return abs(state_container.value)
            
            def __invert__(self):
                return ~state_container.value
            
            # Type conversion
            def __complex__(self):
                return complex(state_container.value)
            
            def __round__(self, ndigits=None):
                if ndigits is None:
                    return round(state_container.value)
                return round(state_container.value, ndigits)
            
            def __trunc__(self):
                import math
                return math.trunc(state_container.value)
            
            def __floor__(self):
                import math
                return math.floor(state_container.value)
            
            def __ceil__(self):
                import math
                return math.ceil(state_container.value)
            
            # Support built-in functions
            def __index__(self):
                """Support bin(), oct(), hex() etc. functions that require integer index"""
                return state_container.value.__index__() if hasattr(state_container.value, '__index__') else int(state_container.value)
            
            # Formatting support
            def __format__(self, format_spec):
                return format(state_container.value, format_spec)
            
            # Byte and index support (for sequences)
            def __reversed__(self):
                return reversed(state_container.value)
            
            # Context manager support (if the original object supports)
            def __enter__(self):
                if hasattr(state_container.value, '__enter__'):
                    return state_container.value.__enter__()
                raise AttributeError(f"'{type(state_container.value).__name__}' object has no attribute '__enter__'")
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if hasattr(state_container.value, '__exit__'):
                    return state_container.value.__exit__(exc_type, exc_val, exc_tb)
                raise AttributeError(f"'{type(state_container.value).__name__}' object has no attribute '__exit__'")
        
        # try to use the name and module of the original type but keep our own behavior
        StateAccessorClass = type(
            original_type.__name__,  # use the name of the original type
            (StateAccessorBase,), 
            {
                '__module__': original_type.__module__,  # use the module of the original type
                '__qualname__': original_type.__qualname__ if hasattr(original_type, '__qualname__') else original_type.__name__
            }
        )
        
        return StateAccessorClass
    
    StateAccessorClass = create_state_accessor_class()
    state_instance = StateAccessorClass()
    
    # use cast to make the dynamically created object be regarded as the original type T
    # so that the IDE will provide the correct type hints
    return cast(T, state_instance), setValue

# Decorator: provide zero-frame-overhead state context for a function
def stateful(func: Callable[..., T]) -> Callable[..., T]:
    if not hasattr(useState, "_TLS"):
        setattr(useState, "_TLS", threading.local())
    _TLS = getattr(useState, "_TLS")

    # per-thread, per-function registry persists across calls
    # stored on TLS to ensure thread isolation
    def _get_registry_for_current_thread() -> Dict[int, Any]:
        if not hasattr(_TLS, "func_registries"):
            _TLS.func_registries = {}
        func_registries = _TLS.func_registries
        reg = func_registries.get(func)
        if reg is None:
            reg = {}
            func_registries[func] = reg
        return reg

    def wrapper(*args, **kwargs):
        if not hasattr(_TLS, "stack"):
            _TLS.stack = []
        _TLS.stack.append({"registry": _get_registry_for_current_thread(), "index": 0})
        try:
            return func(*args, **kwargs)
        finally:
            _TLS.stack.pop()

    return wrapper  # type: ignore[return-value]

# export public API
__all__ = ['useState', 'StateSetter', 'StateAccessorProtocol', 'stateful']
