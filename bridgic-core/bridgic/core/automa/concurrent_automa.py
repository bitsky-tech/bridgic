from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Callable, Final, cast, Tuple, Dict
from typing_extensions import override

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.worker_decorator import ArgsMappingRule
from bridgic.core.types.error import AutomaRuntimeError
from bridgic.core.types.common import AutomaType
from bridgic.core.automa.interaction import InteractionFeedback

class ConcurrentAutoma(GraphAutoma):
    """
    A concurrent automa is a subclass of graph automa that execute multiple workers concurrently.

    According to the `Concurrency Model of Worker`, each worker in the concurrent automa may choose to run in one of the two concurrency modes:

    1. `Async Mode`: Different workers run asynchronously and concurrently, driven by the event loop of the main thread.
    2. `Parallelism Mode`: Different workers run synchronously in separate threads, driven by the thread pool of the concurrent automa.

    These two modes correspond to the `arun()` and `run()` methods of Worker, respectively.

    When all the workers have finished running, the concurrent automa will merge their output results into a list and return it to the caller.    
    """

    _MERGER_WORKER_KEY: Final[str] = "__merger__"
    
    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        super().__init__(name=name, thread_pool=thread_pool)

        # Implementation notes:
        # There are two types of workers in the concurrent automa:
        # 1. Concurrent workers: These workers will be concurrently executed with each other.
        # 2. The Merger worker: This worker will merge the results of all the concurrent workers.

        cls = type(self)
        if cls.automa_type() == AutomaType.Concurrent:
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

    @classmethod
    def automa_type(cls) -> AutomaType:
        """
        Subclasses of GraphAutoma can declare this class method `automa_type` to specify the type of worker decorator.
        """
        return AutomaType.Concurrent

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
        interaction_feedback: Optional[InteractionFeedback] = None,
        interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
        **kwargs: Dict[str, Any]
    ) -> List[Any]:
        result = await super().arun(
            *args,
            interaction_feedback=interaction_feedback,
            interaction_feedbacks=interaction_feedbacks,
            **kwargs
        )
        return cast(List[Any], result)
