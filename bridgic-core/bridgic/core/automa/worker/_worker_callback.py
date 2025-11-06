import inspect

from typing import Any, Dict, Union, List, Optional, Type, TypeVar, TYPE_CHECKING, get_origin, get_args
from typing_extensions import override
from bridgic.core.types._serialization import Serializable

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma

# Type variable for WorkerCallback subclasses
T_WorkerCallback = TypeVar("T_WorkerCallback", bound="WorkerCallback")


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
        parent: Optional["GraphAutoma"] = None,
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
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma] = None
            Parent automa instance containing this worker. Will be None when is_top_level is True.
        arguments : Dict[str, Any] = None
            Execution parameters with keys "args" and "kwargs".
        """
        pass

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["GraphAutoma"] = None,
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
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma] = None
            Parent automa instance containing this worker. Will be None when is_top_level is True.
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
        parent: Optional["GraphAutoma"] = None,
        arguments: Dict[str, Any] = None,
        error: Exception = None,
    ) -> None:
        """
        Hook invoked when worker execution raises an exception.

        Called when the worker execution raises an exception. Use for error
        handling, logging, or event publishing. Cannot modify execution
        logic or arguments.

        **Important: Exception Matching Mechanism**

        The exception matching is based on the **type annotation** of the `error`
        parameter, using **inheritance relationship** (not exact type matching).
        This means:
        - The parameter name MUST be `error` and the type annotation is critical 
          for the matching mechanism.
        - If you annotate `error: ValueError`, it will match `ValueError` and all
          its subclasses (e.g., `UnicodeDecodeError`).
        - If you annotate `error: Exception`, it will match all exceptions (since
          all exceptions inherit from Exception).
        - You can use `Union[Type1, Type2, ...]` to match multiple exception types.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level: bool = False
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma] = None
            Parent automa instance containing this worker. Will be None when is_top_level is True.
        arguments : Dict[str, Any] = None
            Execution arguments with keys "args" and "kwargs".
        error : Exception = None
            The exception raised during worker execution. The type annotation of this
            parameter determines which exceptions this callback will handle. The matching
            is based on inheritance relationship (using isinstance), so a callback with
            `error: ValueError` will match ValueError and all its subclasses.
        """
        pass

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "callback_cls": self.__class__.__module__ + "." + self.__class__.__qualname__,
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

class WorkerCallbackBuilder:
    """
    Builder class for creating instances of `WorkerCallback` subclasses.

    This builder is designed to construct instances of subclasses of `WorkerCallback`.
    The `_callback_type` parameter should be a subclass of `WorkerCallback`, and `build()` 
    will return an instance of that specific subclass.
    """
    _callback_type: Type[T_WorkerCallback]
    """The specific subclass of `WorkerCallback` to instantiate."""
    _init_kwargs: Dict[str, Any]
    """The initialization arguments for the instance."""

    def __init__(self, callback_type: Type[T_WorkerCallback], init_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the builder with a `WorkerCallback` subclass type.

        Parameters
        ----------
        callback_type : Type[T_WorkerCallback]
            A subclass of `WorkerCallback` to be instantiated.
        init_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the subclass constructor.
        """
        self._callback_type = callback_type
        self._init_kwargs = init_kwargs or {}

    def build(self) -> T_WorkerCallback:
        """
        Build and return an instance of the specified `WorkerCallback` subclass.

        Returns
        -------
        T_WorkerCallback
            An instance of the `WorkerCallback` subclass specified during initialization.
        """
        return self._callback_type(**self._init_kwargs)


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
    parent: Optional["GraphAutoma"] = None,
    arguments: Dict[str, Any] = None,
    error: Exception = None,
) -> bool:
    """
    Try to handle an exception using the provided callbacks.

    This function checks each callback's type annotation for the `on_worker_error` method.
    If a callback's annotation matches the exception type, it will be called.
    The function returns True if at least one callback handled the exception, False otherwise.

    Parameters
    ----------
    callbacks : List[WorkerCallback]
        List of callbacks to check.
    key : str
        Worker identifier.
    is_top_level : bool, optional
        Whether the worker is the top-level automa. Default is False. When True, parent will be None.
    parent : Optional[GraphAutoma], optional
        Parent automa instance containing this worker. Will be None when is_top_level is True.
    arguments : Dict[str, Any], optional
        Execution arguments with keys "args" and "kwargs".
    error : Exception, optional
        The exception to handle.

    Returns
    -------
    bool
        True if at least one callback handled the exception, False otherwise.
    """
    handled = False
    for callback in callbacks:
        if can_handle_exception(callback, error):
            await callback.on_worker_error(
                key=key,
                is_top_level=is_top_level,
                parent=parent,
                arguments=arguments,
                error=error,
            )
            handled = True
    
    return handled