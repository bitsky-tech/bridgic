import inspect
import pickle

from typing import Callable, Dict, Optional, TYPE_CHECKING, Tuple, List, Any
from types import MethodType
from inspect import _ParameterKind
from typing_extensions import override
from bridgic.core.automa.worker import Worker
from bridgic.core.types._error import WorkerRuntimeError
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func, get_param_names_all_kinds

if TYPE_CHECKING:
    from bridgic.core.automa._automa import Automa

class CallableWorker(Worker):
    """
    This class is a worker that wraps a callable object, such as functions or methods.

    Parameters
    ----------
    func_or_method : Optional[Callable]
        The callable to be wrapped by the worker. If `func_or_method` is None, 
        `state_dict` must be provided.
    """
    _is_async: bool
    _callable: Callable
    # Used to deserialization.
    _expected_bound_parent: bool

    # Cached method signatures, with no need for serialization.
    __cached_param_names_of_callable: Dict[_ParameterKind, List[str]]

    def __init__(
        self, 
        func_or_method: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        func_or_method : Optional[Callable]
            The callable to be wrapped by the worker. If `func_or_method` is None, 
            `state_dict` must be provided.
        """
        super().__init__()
        self._is_async = inspect.iscoroutinefunction(func_or_method)
        self._callable = func_or_method
        self._expected_bound_parent = False
        
        # Cached method signatures, with no need for serialization.
        self.__cached_param_names_of_callable = None

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        if self._expected_bound_parent:
            raise WorkerRuntimeError(
                f"The callable is expected to be bound to the parent, "
                f"but not bounded yet: {self._callable}"
            )
        if self._is_async:
            return await self._callable(*args, **kwargs)
        return await super().arun(*args, **kwargs)

    def run(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        assert self._is_async is False
        return self._callable(*args, **kwargs)

    @override
    def get_input_param_names(self) -> Dict[_ParameterKind, List[str]]:
        """
        Get the names of input parameters of this callable worker.
        Use cached result if available in order to improve performance.

        Returns
        -------
        Dict[_ParameterKind, List[str]]
            A dictionary of input parameter names by the kind of the parameter.
            The key is the kind of the parameter, which is one of five possible values:

            - inspect.Parameter.POSITIONAL_ONLY
            - inspect.Parameter.POSITIONAL_OR_KEYWORD
            - inspect.Parameter.VAR_POSITIONAL
            - inspect.Parameter.KEYWORD_ONLY
            - inspect.Parameter.VAR_KEYWORD
        """
        if self.__cached_param_names_of_callable is None:
            self.__cached_param_names_of_callable = get_param_names_all_kinds(self._callable)
        return self.__cached_param_names_of_callable

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["is_async"] = self._is_async
        # Note: Not to use pickle to serialize the callable here.
        # We customize the serialization method of the callable to avoid creating instance multiple times and to minimize side effects.
        bounded = isinstance(self._callable, MethodType)
        state_dict["bounded"] = bounded
        if bounded:
            if self._callable.__self__ is self.parent:
                state_dict["callable_name"] = self._callable.__module__ + "." + self._callable.__qualname__
            else:
                state_dict["pickled_callable"] = pickle.dumps(self._callable)
        else:
            state_dict["callable_name"] = self._callable.__module__ + "." + self._callable.__qualname__
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        # Deserialize from the state_dict.
        self._is_async = state_dict["is_async"]
        bounded = state_dict["bounded"]
        if bounded:
            pickled_callable = state_dict.get("pickled_callable", None)
            if pickled_callable is None:
                self._callable = load_qualified_class_or_func(state_dict["callable_name"])
                # Partially deserialized, need to be bound to the parent.
                self._expected_bound_parent = True
            else:
                self._callable = pickle.loads(pickled_callable)
                self._expected_bound_parent = False
        else:
            self._callable = load_qualified_class_or_func(state_dict["callable_name"])
            self._expected_bound_parent = False
        
        # Cached method signatures, with no need for serialization.
        self.__cached_param_names_of_callable = None

    @property
    def callable(self):
        return self._callable

    @property
    def parent(self) -> "Automa":
        return super().parent

    @parent.setter
    def parent(self, value: "Automa"):
        if self._expected_bound_parent:
            self._callable = MethodType(self._callable, value)
            self._expected_bound_parent = False
        Worker.parent.fset(self, value)

    @override
    def __str__(self) -> str:
        return f"CallableWorker(callable={self._callable.__name__})"
