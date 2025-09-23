from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import override
from typing import Callable

from bridgic.core.automa.worker import Worker
from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker_decorator import ArgsMappingRule
from bridgic.core.types.error import AutomaRuntimeError
from bridgic.core.types.common import AutomaType

class SequentialAutoma(GraphAutoma):
    """
    A sequential automa is a subclass of graph automa that execute multiple workers sequentially. Workers in a sequential automa are executed one by one, in the order of their position index in the automa.

    When all the workers have finished running, the sequential automa will return the output results of the last worker to the caller.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        super().__init__(name=name, output_worker_key=None, thread_pool=thread_pool)

        cls = type(self)
        if cls.worker_decorator_type() == AutomaType.Sequential:
            # The _registered_worker_funcs data are from @worker decorators.
            # Initialize the decorated sequential workers.
            last_worker_key = None
            for worker_key, worker_func in self._registered_worker_funcs.items():
                is_start = last_worker_key is None
                dependencies = [] if last_worker_key is None else [last_worker_key]
                super().add_func_as_worker(
                    key=worker_key,
                    func=worker_func,
                    dependencies=dependencies,
                    is_start=is_start,
                    args_mapping_rule=worker_func.__args_mapping_rule__,
                )
                last_worker_key = worker_key

        if last_worker_key is not None:
            GraphAutoma.output_worker_key.fset(self, last_worker_key)

    @classmethod
    def worker_decorator_type(cls) -> AutomaType:
        """
        Subclasses of GraphAutoma can declare this class method `worker_decorator_type` to specify the type of worker decorator.
        """
        return AutomaType.Sequential

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
        last_worker_key = super().output_worker_key
        is_start = last_worker_key is None
        dependencies = [] if last_worker_key is None else [last_worker_key]
        super().add_worker(
            key=key, 
            worker=worker,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )
        GraphAutoma.output_worker_key.fset(self, key)

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
        last_worker_key = super().output_worker_key
        is_start = last_worker_key is None
        dependencies = [] if last_worker_key is None else [last_worker_key]
        super().add_func_as_worker(
            key=key, 
            func=func,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )
        GraphAutoma.output_worker_key.fset(self, key)

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
        super_automa = super()
        def wrapper(func: Callable):
            last_worker_key = super_automa.output_worker_key
            is_start = last_worker_key is None
            dependencies = [] if last_worker_key is None else [last_worker_key]
            super_automa.add_func_as_worker(
                key=key, 
                func=func,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
            GraphAutoma.output_worker_key.fset(self, key)

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

    @override
    @GraphAutoma.output_worker_key.setter
    def output_worker_key(self, worker_key: str):
        raise AutomaRuntimeError(f"output_worker_key is not allowed to be set on a sequential automa")

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        raise AutomaRuntimeError(f"ferry_to() is not allowed to be called on a sequential automa")
