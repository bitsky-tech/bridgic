import inspect

from threading import Lock
from typing import Any, Dict, Union, List, Optional, Type, Generic, TypeVar, TYPE_CHECKING, get_origin, get_args
from typing_extensions import override
from bridgic.core.types._serialization import Serializable
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func

if TYPE_CHECKING:
    from bridgic.core.automa._automa import Automa

T_WorkerCallback = TypeVar("T_WorkerCallback", bound="WorkerCallback")
"""Type variable for WorkerCallback subclasses."""


class WorkerCallback(Serializable):
    """
    Callback for the execution of a worker instance during the running 
    of a prebult automa.

    This class defines the interfaces that will be called before or after 
    the execution of the corresponding worker. Callbacks are typically used 
    for validating input, monitoring execution, and collecting logs, etc.

    Methods
    -------
    on_worker_start(key, is_top_level, parent, arguments)
        Hook invoked before worker execution.
    on_worker_end(key, is_top_level, parent, arguments, result)
        Hook invoked after worker execution.
    on_worker_error(key, is_top_level, parent, arguments, error)
        Hook invoked when worker execution raises an exception.
    """
    async def on_worker_start(
        self, 
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        """
        Hook invoked before worker execution.

        Called immediately before the worker runs. Use for arguments
        validation, logging, or monitoring. Cannot modify execution
        arguments or logic.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level: bool = False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa] = None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Dict[str, Any] = None
            Execution parameters with keys "args" and "kwargs".
        """
        pass

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        """
        Hook invoked after worker execution.

        Called immediately after the worker completes. Use for result
        monitoring, logging, event publishing, or validation. Cannot
        modify execution results or logic.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level: bool = False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa] = None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Dict[str, Any] = None
            Execution arguments with keys "args" and "kwargs".
        result : Any = None
            Worker execution result.
        """
        pass

    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
        arguments: Dict[str, Any] = None,
        error: Exception = None,
    ) -> bool:
        """
        Hook invoked when worker execution raises an exception.

        Called when the worker execution raises an exception. Use for error handling, logging, 
        or event publishing. Cannot modify execution logic or arguments.

        **Exception Matching Mechanism: How to Handle a Specific Exception**

        The framework enable your callback to handle a given exception based on the 
        **type annotation** of the `error` parameter in your `on_worker_error` method.
        The matching follows these rules:

        - The parameter name MUST be `error` and the type annotation is critical for the 
          matching mechanism.
        - If you annotate `error: ValueError`, it will match `ValueError` and all its 
          subclasses (e.g., `UnicodeDecodeError`).
        - If you annotate `error: Exception`, it will match all exceptions (since all exceptions 
          inherit from Exception).
        - If you want to match multiple exception types, you can use `Union[Type1, Type2, ...]`.


        **Return Value: Whether to Suppress the Exception**

        - If `on_worker_error` returns `True`, the framework will suppress the exception. 
          The framework will then proceed as if there was no error, and the worker result 
          will be set to None.
        - If `on_worker_error` returns `False`, the framework will simply observe the error; 
          after all matching callbacks are called, the framework will re-raise the exception.

        **Special Case: Interaction Exceptions Cannot Be Suppressed**

        To ensure human-interaction mechanisms work correctly, exceptions of type
        `_InteractionEventException` or `InteractionException` (including their subclasses) 
        **CANNOT** be suppressed by any callback. Even if your callback returns `True`, the 
        framework will forcibly re-raise the exception. This ensures these exceptions always 
        propagate correctly through the automa hierarchy to trigger necessary human interactions.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level: bool = False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa] = None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Dict[str, Any] = None
            Execution arguments with keys "args" and "kwargs".
        error : Exception = None
            The exception raised during worker execution. The type annotation of this
            parameter determines which exceptions this callback will handle. The matching
            is based on inheritance relationship (using isinstance), so a callback with
            `error: ValueError` will match ValueError and all its subclasses.

        Returns
        -------
        bool
            True if the automa should suppress the exception (not re-raise it); False otherwise.
        """
        return False

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "callback_cls": self.__class__.__module__ + "." + self.__class__.__qualname__,
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

class WorkerCallbackBuilder(Generic[T_WorkerCallback]):
    """
    Builder class for creating instances of `WorkerCallback` subclasses.

    This builder is designed to construct instances of subclasses of `WorkerCallback`.
    The `_callback_type` parameter should be a subclass of `WorkerCallback`, and `build()` 
    will return an instance of that specific subclass. There is no need to call `build()` 
    directly. Instead, the framework calls the `build` method automatically to create 
    its own `WorkerCallback` instance for each worker instance.

    Notes
    -----
    **Register a Callback in Different Scope**

    There are three ways to register a callback for three levels of customization:

    - Case 1: Use in worker decorator to register the callback for a specific worker.
    - Case 2: Use in RunningOptions to register the callback for a specific Automa instance.
    - Case 3: Use in GlobalSetting to register the callback for all workers.

    Notes
    -----
    **Shared Instance Mode**

    - When `is_shared=True` (default), all workers within the same scope will share the same 
      callback instance. This is useful for scenarios where a single callback instance is needed 
      to maintain some state across workers within the same scope, such as the connection to 
      an external service. The scope is determined by where the builder is declared:
      - If declared in GlobalSetting: shared across all workers globally
      - If declared in RunningOptions: shared across all workers within that Automa instance
    - When `is_shared=False`, each worker will get its own callback instance. This is useful for 
      scenarios where a independent callback instance is needed for each worker.

    Examples
    --------
    There are three ways to use the builder, for different levels of customization:

    >>> # Define a custom callback class:
    >>> class MyEmptyCallback(WorkerCallback):
    ...     pass
    ...
    >>> # Case 1: Use in worker decorator to register the callback for a specific worker:
    >>> class MyGraphAutoma(GraphAutoma):
    ...     @worker(callback_builders=[WorkerCallbackBuilder(MyEmptyCallback)])
    ...     async def my_worker(self, x: int) -> int:
    ...         return x + 1
    ...
    >>> # Case 2: Use in RunningOptions to register the callback for a specific Automa instance:
    ...     running_options = RunningOptions(callback_builders=[WorkerCallbackBuilder(MyEmptyCallback)])
    ...     graph = MyGraphAutoma(running_options=running_options)
    ...
    >>> # Case 3: Use in GlobalSetting to register the callback for all workers:
    >>> GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(MyEmptyCallback)])
    """
    _callback_type: Type[T_WorkerCallback]
    """The specific subclass of `WorkerCallback` to instantiate."""
    _init_kwargs: Dict[str, Any]
    """The initialization arguments for the instance."""
    _is_shared: bool
    """Whether to use shared instance mode (reuse the same instance within the declaration scope)."""

    _shared_instance: Optional[T_WorkerCallback] = None
    """Shared instance of the callback within the declaration scope."""
    _shared_lock: Lock = Lock()
    """Lock for thread-safe shared instance creation."""

    def __init__(
        self,
        callback_type: Type[T_WorkerCallback],
        init_kwargs: Optional[Dict[str, Any]] = None,
        is_shared: bool = True,
    ):
        """
        Initialize the builder with a `WorkerCallback` subclass type.

        Parameters
        ----------
        callback_type : Type[T_WorkerCallback]
            A subclass of `WorkerCallback` to be instantiated.
        init_kwargs : Optional[Dict[str, Any]]
            Keyword arguments to pass to the subclass constructor.
        is_shared : bool, default True
            If True, the callback instance will be shared within the declaration scope:
            If False, each worker will get its own callback instance.
        """
        self._callback_type = callback_type
        self._init_kwargs = init_kwargs or {}
        self._is_shared = is_shared

    def build(self) -> T_WorkerCallback:
        """
        Build and return an instance of the specified `WorkerCallback` subclass.

        Returns
        -------
        T_WorkerCallback
            An instance of the `WorkerCallback` subclass specified during initialization.
        """
        if self._is_shared:
            if self._shared_instance is None:
                with self._shared_lock:
                    if self._shared_instance is None:
                        self._shared_instance = self._callback_type(**self._init_kwargs)
            return self._shared_instance
        else:
            return self._callback_type(**self._init_kwargs)

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "callback_type": self._callback_type.__module__ + "." + self._callback_type.__qualname__,
            "init_kwargs": self._init_kwargs,
            "is_shared": self._is_shared,
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        # Load the callback type from its fully qualified name
        callback_type_name = state_dict["callback_type"]
        self._callback_type = load_qualified_class_or_func(callback_type_name)

        # Load init_kwargs (default to empty dict if not present or None)
        init_kwargs = state_dict.get("init_kwargs")
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}

        # Load is_shared (default to True if not present for backward compatibility)
        self._is_shared = state_dict.get("is_shared", True)

        # Reset shared instance and lock (they will be recreated when needed)
        self._shared_instance = None
        self._shared_lock = Lock()


def can_handle_exception(callback: WorkerCallback, error: Exception) -> bool:
    """
    Check if a callback can handle a specific exception type based on its type annotation.

    The matching is based on inheritance relationship using `isinstance()`, not exact
    type matching. This means:
    - A callback with `error: ValueError` will match `ValueError` and all its subclasses.
    - A callback with `error: Exception` will match all exceptions (since all exceptions inherit from Exception).
    - Union types are supported: `error: Union[ValueError, TypeError]` will match both.

    Parameters
    ----------
    callback : WorkerCallback
        The callback instance to check.
    error : Exception
        The exception to check.

    Returns
    -------
    bool
        True if the callback can handle this exception type (based on inheritance),
        False otherwise.
    """
    try:
        sig = inspect.signature(callback.on_worker_error)
        error_param = sig.parameters.get('error')
        if error_param and error_param.annotation != inspect.Parameter.empty:
            error_type = error_param.annotation
            # Resolve forward references and get the actual type
            if isinstance(error_type, str):
                # Forward reference, skip for now
                return False
            
            # Handle Union types (e.g., Union[ValueError, TypeError])
            origin = get_origin(error_type)
            if origin is Union:
                union_args = get_args(error_type)
                # Check if exception is instance of any type in the Union
                return any(isinstance(error, t) for t in union_args if isinstance(t, type) and issubclass(t, Exception))
            
            # Handle single type annotation (including base Exception class)
            elif isinstance(error_type, type) and issubclass(error_type, Exception):
                # Match based on inheritance relationship using isinstance
                return isinstance(error, error_type)
    except (ValueError, TypeError, AttributeError):
        # If we can't determine the type, skip this callback
        pass
    
    return False


async def try_handle_error_with_callbacks(
    callbacks: List[WorkerCallback],
    key: str,
    is_top_level: bool = False,
    parent: Optional["Automa"] = None,
    arguments: Dict[str, Any] = None,
    error: Exception = None,
) -> bool:
    """
    Try to handle an exception using the provided callbacks.

    This function returns True if at least one callback's `on_worker_error` method returns True
    (indicating a request to suppress the exception) and the exception to be handled is not an 
    interaction exception. Otherwise the function returns False, which means the framework will 
    re-raise the exception as usual.

    **Important**: Interaction exceptions (`_InteractionEventException` or `InteractionException`
    and their subclasses) cannot be suppressed. If the exception is an interaction exception,
    this function will always return False regardless of callback return values.

    Parameters
    ----------
    callbacks : List[WorkerCallback]
        List of callbacks to check.
    key : str
        Worker identifier.
    is_top_level : bool, optional
        Whether the worker is the top-level automa. Default is False. When True, parent will be the automa itself (parent is self).
    parent : Optional[Automa], optional
        Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
    arguments : Dict[str, Any], optional
        Execution arguments with keys "args" and "kwargs".
    error : Exception, optional
        The exception to handle.

    Returns
    -------
    bool
        True if at least one callback requested to suppress the exception and the exception
        is not an interaction exception; False otherwise. The framework will re-raise the 
        exception if this returns False.
    """
    # Import here to avoid circular import
    from bridgic.core.automa._automa import _InteractionEventException
    from bridgic.core.automa.interaction._human_interaction import InteractionException
    
    should_suppress = False
    is_interaction_exception = isinstance(error, (_InteractionEventException, InteractionException))

    for callback in callbacks:
        if can_handle_exception(callback, error):
            suppress_request = await callback.on_worker_error(
                key=key,
                is_top_level=is_top_level,
                parent=parent,
                arguments=arguments,
                error=error,
            )
            if suppress_request and not is_interaction_exception:
                should_suppress = True
    
    return should_suppress