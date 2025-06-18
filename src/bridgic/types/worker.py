import copy
import inspect
import uuid

from typing import Any, List, Dict, Callable, get_type_hints

class Worker:
    def __init__(self, name: str = None, as_thread: bool = False):
        """
        Parameters
        ----------
        name : str (default = None, then a generated name will be provided)
            The name of the worker.
        as_thread : bool (default = False)
            If True, the worker will run as a single thread when orchestrated.
            If False, the worker will run as a normal event in the current event loop.
        """
        self.name: str = name or f"unnamed-worker-{uuid.uuid4().hex[:8]}"
        self.as_thread: bool = as_thread
        self.__output_buffer: Any = None
        self.__output_setted: bool = False
        self.__local_space: Dict[str, Any] = {}

    async def process_async(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"process_async is not implemented in {type(self)}")

    @property
    def return_type(self) -> type:
        return get_type_hints(self.process_async).get('return', Any)

    @property
    def output_buffer(self) -> Any:
        if not self.__output_setted:
            raise RuntimeError(f"output of worker is not ready yet: worker_name={self.name}")
        return copy.deepcopy(self.__output_buffer)
    
    @output_buffer.setter
    def output_buffer(self, value: Any):
        self.__output_buffer = value
        self.__output_setted = True

    @property
    def local_space(self) -> Dict[str, Any]:
        return self.__local_space
    
    @local_space.setter
    def local_space(self, value: Dict[str, Any]):
        self.__local_space = value

class LandableWorker(Worker):
    def __init__(self, *, dependencies: List[str] = [], is_start: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.is_start = is_start
        self.dependencies = dependencies

class CallableWorker(Worker):
    def __init__(self, *, func: Callable, **kwargs):
        super().__init__(**kwargs)
        if not func:
            raise ValueError("func is required")
        self.func = func
        self.is_coro = inspect.iscoroutinefunction(func)

    async def process_async(self, *args, **kwargs) -> Any:
        if self.is_coro:
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

class CallableLandableWorker(CallableWorker, LandableWorker):
    def __init__(
        self,
        *,
        name: str = None,
        as_thread: bool = False,
        func: Callable,
        dependencies: List[str] = [],
        is_start: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            as_thread=as_thread,
            func=func,
            dependencies=dependencies,
            is_start=is_start,
            **kwargs,
        )