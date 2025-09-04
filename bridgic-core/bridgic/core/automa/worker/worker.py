"""
A worker is the basic building block of an Automa, responsible for executing tasks.

[The Concurrency Model of Worker]

A worker has two types of methods for running tasks:

1. `arun()`: This method is used for running tasks asynchronously. It is driven by the event loop of the main thread.
2. `run()`: This method is used for running tasks that are I/O-bound or CPU-bound. It is driven by the thread pool or the process pool (TODO:) of the Automa. More specifically, I/O-bound tasks are handled by the thread pool, while CPU-bound tasks are handled by the process pool.

"""

import copy
import asyncio
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple
from functools import partial
from bridgic.core.automa.interaction import Event, InteractionFeedback, Feedback
from bridgic.core.types.error import WorkerRuntimeError
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from bridgic.core.automa.automa import Automa

class Worker:
    __output_buffer: Any
    __output_setted: bool
    __local_space: Dict[str, Any]

    def __init__(self, state_dict: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        state_dict : Optional[Dict[str, Any]] (default = None)
            A dictionary for initializing the worker's runtime state. This parameter is intended for internal framework use only, specifically for deserialization, and should not be used by developers.
        """
        self.__parent: Automa = None
        if state_dict is None:
            self.__output_setted = False
            self.__local_space = {}
        else:
            self.__output_setted = state_dict["output_setted"]
            if self.__output_setted:
                self.__output_buffer = state_dict["output_buffer"]
            self.__local_space = state_dict["local_space"]

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        topest_automa = self._get_top_level_automa()
        if topest_automa:
            thread_pool = topest_automa.get_thread_pool()
            if thread_pool:
                # kwargs can only be passed by functools.partial.
                return await loop.run_in_executor(thread_pool, partial(self.run, *args, **kwargs))

        # Unexpected: No thread pool is available.
        # Case 1: the worker is not inside an Automa (uncommon case).
        # Case 2: no thread pool is setup by the top-level automa.
        raise WorkerRuntimeError(f"No thread pool is available for the worker {type(self)}")

    def run(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        raise NotImplementedError(f"run() is not implemented in {type(self)}")

    def _get_top_level_automa(self) -> Optional["Automa"]:
        """
        Get the top-level automa instance reference.
        """
        top_level_automa = self.parent
        while top_level_automa and (not top_level_automa.is_top_level()):
            top_level_automa = top_level_automa.parent
        return top_level_automa

    @property
    def parent(self) -> "Automa":
        return self.__parent

    @parent.setter
    def parent(self, value: "Automa"):
        self.__parent = value

    @property
    def output_buffer(self) -> Any:
        if not self.__output_setted:
            raise RuntimeError(f"output of worker is not ready yet")
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

    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["output_setted"] = self.__output_setted
        if self.__output_setted:
            state_dict["output_buffer"] = self.__output_buffer
        state_dict["local_space"] = self.__local_space
        return state_dict

    @classmethod
    def load_from_dict(cls, state_dict: Dict[str, Any]) -> "Worker":
        return cls(state_dict=state_dict)

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        """
        Handoff control flow to the specified worker, passing along any arguments as needed.
        The specified worker will always start to run asynchronously in the next event loop, regardless of its dependencies.

        Parameters
        ----------
        worker_key : str
            The key of the worker to run.
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.
        """
        if self.parent is None:
            raise WorkerRuntimeError(f"`ferry_to` method can only be called by a worker inside an Automa")
        self.parent.ferry_to(worker_key, *args, **kwargs)

    def post_event(self, event: Event) -> None:
        """
        Post an event to the application layer outside the Automa.

        The event handler implemented by the application layer will be called in the same thread as the worker (maybe the main thread or a new thread from the thread pool).
        
        Note that `post_event` can be called in a non-async method or an async method.

        The event will be bubbled up to the top-level Automa, where it will be processed by the event handler registered with the event type.

        Parameters
        ----------
        event: Event
            The event to be posted.
        """
        if self.parent is None:
            raise WorkerRuntimeError(f"`post_event` method can only be called by a worker inside an Automa")
        self.parent.post_event(event)

    def request_feedback(self, event: Event) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        Note that `post_event` should only be called from within a non-async method running in the new thread of the Automa thread pool.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        """
        if self.parent is None:
            raise WorkerRuntimeError(f"`request_feedback` method can only be called by a worker inside an Automa")
        return self.parent.request_feedback(event)

    async def request_feedback_async(self, event: Event) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        The event handler implemented by the application layer will be called in the next event loop, in the main thread.

        Note that `post_event` should only be called from within an asynchronous method running in the main event loop of the top-level Automa.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        """
        if self.parent is None:
            raise WorkerRuntimeError(f"`request_feedback_async` method can only be called by a worker inside an Automa")
        return await self.parent.request_feedback_async(event)

    def interact_with_human(self, event: Event) -> InteractionFeedback:
        if self.parent is None:
            raise WorkerRuntimeError(f"`interact_with_human` method can only be called by a worker inside an Automa")
        return self.parent.interact_with_human_from_worker(event, self)
