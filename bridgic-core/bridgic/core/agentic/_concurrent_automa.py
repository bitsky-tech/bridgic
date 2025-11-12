from typing import Optional, ClassVar
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Callable, Final, cast, Tuple, Dict, Union
from typing_extensions import override

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker
from bridgic.core.types._error import AutomaRuntimeError
from bridgic.core.types._common import AutomaType, ArgsMappingRule
from bridgic.core.automa.interaction import InteractionFeedback

class ConcurrentAutoma(GraphAutoma):
    """
    This class is to provide concurrent execution of multiple workers.

    In accordance with the defined "Concurrency Model of Worker", each worker within 
    a ConcurrentAutoma can be configured to operate in one of two concurrency modes:

    1. **Async Mode**: Workers execute concurrently in an asynchronous fashion, driven 
    by the event loop of the main thread. This execution mode corresponds to the `arun()` 
    method of the Worker.
    2. **Parallel Mode**: Workers execute synchronously, each running in a dedicated 
    thread within a thread pool managed by the ConcurrentAutoma. This execution mode 
    corresponds to the `run()` method of the Worker.

    Upon completion of all worker tasks, the concurrent automa instance aggregates 
    the result outputs from each worker into a single list, which is then returned 
    to the caller.
    """

    # Automa type.
    AUTOMA_TYPE: ClassVar[AutomaType] = AutomaType.Concurrent

    _MERGER_WORKER_KEY: Final[str] = "__merger__"
    
    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        # Implementation notes:
        # There are two types of workers in the concurrent automa:
        # 1. Concurrent workers: These workers will be concurrently executed with each other.
        # 2. The Merger worker: This worker will merge the results of all the concurrent workers.

        cls = type(self)
        if cls.AUTOMA_TYPE == AutomaType.Concurrent:
            # The _registered_worker_funcs data are from @worker decorators.
            # Initialize the decorated concurrent workers.
            for worker_key, worker_func in self._registered_worker_funcs.items():
                super().add_func_as_worker(
                    key=worker_key,
                    func=worker_func,
                    is_start=True,
                )

        # Add a hidden worker as the merger worker, which will merge the results of all the start workers.
        super().add_func_as_worker(
            key=self._MERGER_WORKER_KEY,
            func=self._merge_workers_results,
            dependencies=super().all_workers(),
            is_output=True,
            args_mapping_rule=ArgsMappingRule.MERGE,
        )

    def _merge_workers_results(self, results: List[Any]) -> List[Any]:
        return results

    @override
    def add_worker(
        self,
        key: str,
        worker: Worker,
    ) -> None:
        """
        Add a concurrent worker to the concurrent automa. This worker will be concurrently executed with other concurrent workers.

        Parameters
        ----------
        key : str
            The key of the worker.
        worker : Worker
            The worker instance to be registered.
        """
        if key == self._MERGER_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `add_worker()`")
        # Implementation notes:
        # Concurrent workers are implemented as start workers in the underlying graph automa.
        super().add_worker(key=key, worker=worker, is_start=True)
        super().add_dependency(self._MERGER_WORKER_KEY, key)

    @override
    def add_func_as_worker(
        self,
        key: str,
        func: Callable,
    ) -> None:
        """
        Add a function or method as a concurrent worker to the concurrent automa. This worker will be concurrently executed with other concurrent workers.

        Parameters
        ----------
        key : str
            The key of the function worker.
        func : Callable
            The function to be added as a concurrent worker to the automa.
        """
        if key == self._MERGER_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `add_func_as_worker()`")
        # Implementation notes:
        # Concurrent workers are implemented as start workers in the underlying graph automa.
        super().add_func_as_worker(key=key, func=func, is_start=True)
        super().add_dependency(self._MERGER_WORKER_KEY, key)

    @override
    def worker(
        self,
        *,
        key: Optional[str] = None,
    ) -> Callable:
        """
        This is a decorator to mark a function or method as a concurrent worker of the concurrent automa. This worker will be concurrently executed with other concurrent workers.

        Parameters
        ----------
        key : str
            The key of the worker. If not provided, the name of the decorated callable will be used.
        """
        if key == self._MERGER_WORKER_KEY:
            raise AutomaRuntimeError(f"the reserved key `{key}` is not allowed to be used by `automa.worker()`")

        super_automa = super()
        def wrapper(func: Callable):
            super_automa.add_func_as_worker(key=key, func=func, is_start=True)
            super_automa.add_dependency(self._MERGER_WORKER_KEY, key)

        return wrapper

    @override
    def remove_worker(self, key: str) -> None:
        """
        Remove a concurrent worker from the concurrent automa.

        Parameters
        ----------
        key : str
            The key of the worker to be removed.
        """
        if key == self._MERGER_WORKER_KEY:
            raise AutomaRuntimeError(f"the merge worker is not allowed to be removed from the concurrent automa")
        super().remove_worker(key=key)

    @override
    def add_dependency(
        self,
        key: str,
        dependency: str,
    ) -> None:
        raise AutomaRuntimeError(f"add_dependency() is not allowed to be called on a concurrent automa")

    def all_workers(self) -> List[str]:
        """
        Gets a list containing the keys of all concurrent workers registered in this concurrent automa.

        Returns
        -------
        List[str]
            A list of concurrent worker keys.
        """
        keys_list = super().all_workers()
        # Implementation notes:
        # Hide the merger worker from the list of concurrent workers.
        return list(filter(lambda key: key != self._MERGER_WORKER_KEY, keys_list))

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        raise AutomaRuntimeError(f"ferry_to() is not allowed to be called on a concurrent automa")

    async def arun(
        self, 
        *args: Tuple[Any, ...],
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[Any]:
        result = await super().arun(
            *args,
            feedback_data=feedback_data,
            **kwargs
        )
        return cast(List[Any], result)
