from typing import Optional, Final, Callable, Any, Dict, Union, ClassVar
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import override

from bridgic.core.automa.worker import Worker
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.types._error import AutomaRuntimeError
from bridgic.core.types._common import AutomaType, ArgsMappingRule

class SequentialAutoma(GraphAutoma):
    """
    This class is to provide an easy way to orchestrate workers in a strictly 
    sequential manner.

    Each worker within the SequentialAutoma is invoked in the precise order determined 
    by their positional index, ensuring a linear workflow where the output of one worker 
    can serve as the input to the next.

    Upon the completion of all registered workers, the SequentialAutoma returns the output 
    produced by the final worker in the sequence as the overall result to the caller. This 
    design enforces ordered, step-wise processing, making the SequentialAutoma particularly 
    suitable for use cases that require strict procedural dependencies among constituent tasks.
    """

    # Automa type.
    AUTOMA_TYPE: ClassVar[AutomaType] = AutomaType.Sequential

    _TAIL_WORKER_KEY: Final[str] = "__tail__"
    _last_worker_key: Optional[str]

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        cls = type(self)
        self._last_worker_key = None
        if cls.AUTOMA_TYPE == AutomaType.Sequential:
            # The _registered_worker_funcs data are from @worker decorators.
            # Initialize the decorated sequential workers.
            for worker_key, worker_func in self._registered_worker_funcs.items():
                is_start = self._last_worker_key is None
                dependencies = [] if self._last_worker_key is None else [self._last_worker_key]
                super().add_func_as_worker(
                    key=worker_key,
                    func=worker_func,
                    dependencies=dependencies,
                    is_start=is_start,
                    args_mapping_rule=worker_func.__args_mapping_rule__,
                )
                self._last_worker_key = worker_key

        if self._last_worker_key is not None:
            # Add a hidden worker as the tail worker.
            super().add_func_as_worker(
                key=self._TAIL_WORKER_KEY,
                func=self._tail_worker,
                dependencies=[self._last_worker_key],
                is_output=True,
                args_mapping_rule=ArgsMappingRule.AS_IS,
            )

    def _tail_worker(self, result: Any) -> Any:
        # Return the result of the last worker without any modification.
        return result

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["last_worker_key"] = self._last_worker_key
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._last_worker_key = state_dict["last_worker_key"]

    def __add_worker_internal(
        self,
        key: str,
        func_or_worker: Union[Callable, Worker],
        *,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        is_start = self._last_worker_key is None
        dependencies = [] if self._last_worker_key is None else [self._last_worker_key]
        if isinstance(func_or_worker, Callable):
            super().add_func_as_worker(
                key=key, 
                func=func_or_worker,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
        else:
            super().add_worker(
                key=key, 
                worker=func_or_worker,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
        if self._last_worker_key is not None:
            # Remove the old hidden tail worker.
            super().remove_worker(self._TAIL_WORKER_KEY)

        # Add a new hidden tail worker.
        self._last_worker_key = key
        super().add_func_as_worker(
            key=self._TAIL_WORKER_KEY,
            func=self._tail_worker,
            dependencies=[self._last_worker_key],
            is_output=True,
            args_mapping_rule=ArgsMappingRule.AS_IS,
        )

    @override
    def add_worker(
        self,
        key: str,
        worker: Worker,
        *,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        Add a sequential worker to the sequential automa at the end of the automa.

        Parameters
        ----------
        key : str
            The key of the worker.
        worker : Worker
            The worker instance to be registered.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        """
        if key == self._TAIL_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `add_worker()`")

        self.__add_worker_internal(
            key, 
            worker, 
            args_mapping_rule=args_mapping_rule
        )

    @override
    def add_func_as_worker(
        self,
        key: str,
        func: Callable,
        *,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        Add a function or method as a sequential worker to the sequential automa at the end of the automa.

        Parameters
        ----------
        key : str
            The key of the worker.
        func : Callable
            The function to be added as a sequential worker to the automa.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        """
        if key == self._TAIL_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `add_func_as_worker()`")

        self.__add_worker_internal(
            key, 
            func, 
            args_mapping_rule=args_mapping_rule
        )

    @override
    def worker(
        self,
        *,
        key: Optional[str] = None,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> Callable:
        """
        This is a decorator to mark a function or method as a sequential worker of the sequential automa, at the end of the automa.

        Parameters
        ----------
        key : str
            The key of the worker. If not provided, the name of the decorated callable will be used.
        args_mapping_rule: ArgsMappingRule
            The rule of arguments mapping.
        """
        if key == self._TAIL_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `automa.worker()`")

        def wrapper(func: Callable):
            self.__add_worker_internal(
                key, 
                func, 
                args_mapping_rule=args_mapping_rule
            )

        return wrapper

    @override
    def remove_worker(self, key: str) -> None:
        raise AutomaRuntimeError(f"remove_worker() is not allowed to be called on a sequential automa")

    @override
    def add_dependency(
        self,
        key: str,
        depends: str,
    ) -> None:
        raise AutomaRuntimeError(f"add_dependency() is not allowed to be called on a sequential automa")

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        raise AutomaRuntimeError(f"ferry_to() is not allowed to be called on a sequential automa")
