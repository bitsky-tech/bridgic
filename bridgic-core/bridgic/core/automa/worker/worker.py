"""
A worker is the basic building block of an Automa, responsible for executing tasks.

[The Concurrency Model of Worker]

A worker has two types of methods for running tasks:

1. `arun()`: This method is used for running tasks asynchronously. It is driven by the event loop of the main thread.
2. `run()`: This method is used for running tasks that are I/O-bound or CPU-bound. It is driven by the thread pool or the process pool (TODO:) of the Automa. More specifically, I/O-bound tasks are handled by the thread pool, while CPU-bound tasks are handled by the process pool.

"""

import copy
import asyncio
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Tuple, Callable
from functools import partial
from inspect import Parameter
from bridgic.core.automa.interaction import Event, InteractionFeedback, Feedback
from bridgic.core.types.error import WorkerRuntimeError
from bridgic.core.utils.inspect_tools import get_param_names_by_kind, get_param_names_all_kinds

if TYPE_CHECKING:
    from bridgic.core.automa.automa import Automa

class Worker:
    __output_buffer: Any
    __output_has_set: bool

    # Cached method signatures, with no need for serialization.
    __cached_param_names_of_arun: Dict[Parameter, List[str]]
    __cached_param_names_of_run: Dict[Parameter, List[str]]

    @staticmethod
    def safely_map_args(
            in_args: Tuple[Any, ...], 
            in_kwargs: Dict[str, Any],
            rx_param_names_dict: Dict[Parameter, List[str]],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        positional_only_param_names = rx_param_names_dict.get(Parameter.POSITIONAL_ONLY, [])
        positional_or_keyword_param_names = rx_param_names_dict.get(Parameter.POSITIONAL_OR_KEYWORD, [])

        # Resolve the positional arguments `rx_args`.
        positional_param_names = positional_only_param_names + positional_or_keyword_param_names
        var_positional_param_names = rx_param_names_dict.get(Parameter.VAR_POSITIONAL, [])
        if len(in_args) == 1 and in_args[0] is None and len(positional_param_names) == 0 and len(var_positional_param_names) == 0:
            # The special case of the predecessor returning None and the successor has no arguments expected.
            rx_args = ()
        else:
            # In normal cases, positional arguments are unchanged.
            rx_args = in_args

        # Resolve the keyword arguments `rx_kwargs`.
        # keyword arguments are firstly filtered by `masked_keyword_param_names`.
        if len(in_args) > len(positional_only_param_names):
            masked_keyword_param_names = positional_or_keyword_param_names[:len(in_args)-len(positional_only_param_names)]
        else:
            masked_keyword_param_names = []
        rx_kwargs = {k:v for k,v in in_kwargs.items() if k not in masked_keyword_param_names}
        var_keyword_param_names = rx_param_names_dict.get(Parameter.VAR_KEYWORD, [])
        if not var_keyword_param_names:
            # keyword arguments are secondly filtered by `keyword_param_names`.
            keyword_only_param_names = rx_param_names_dict.get(Parameter.KEYWORD_ONLY, [])
            keyword_param_names = positional_or_keyword_param_names + keyword_only_param_names
            rx_kwargs = {k:v for k,v in rx_kwargs.items() if k in keyword_param_names}
        return rx_args, rx_kwargs

    def __init__(self):
        """
        Parameters
        ----------
        state_dict : Optional[Dict[str, Any]] (default = None)
            A dictionary for initializing the worker's runtime state. This parameter is intended for internal framework use only, specifically for deserialization, and should not be used by developers.
        """
        self.__parent: Automa = None
        self.__output_has_set = False
        
        # Cached method signatures, with no need for serialization.
        self.__cached_param_names_of_arun = None
        self.__cached_param_names_of_run = None

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        topest_automa = self._get_top_level_automa()
        if topest_automa:
            thread_pool = topest_automa.thread_pool
            if thread_pool:
                rx_param_names_dict = self._get_param_names_of_run()
                rx_args, rx_kwargs = Worker.safely_map_args(args, kwargs, rx_param_names_dict)
                # kwargs can only be passed by functools.partial.
                return await loop.run_in_executor(thread_pool, partial(self.run, *rx_args, **rx_kwargs))

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
    
    def get_input_param_names(self) -> Dict[Parameter, List[str]]:
        """
        Get the names of input parameters of the worker.
        Use cached result if available in order to improve performance.

        Returns
        -------
        Dict[Parameter, List[str]]
            A dictionary of input parameter names by the kind of the parameter.
            The key is the kind of the parameter, which is one of five possible values:
            - inspect.Parameter.POSITIONAL_ONLY
            - inspect.Parameter.POSITIONAL_OR_KEYWORD
            - inspect.Parameter.VAR_POSITIONAL
            - inspect.Parameter.KEYWORD_ONLY
            - inspect.Parameter.VAR_KEYWORD
        """
        if self.__cached_param_names_of_arun is None:
            self.__cached_param_names_of_arun = get_param_names_all_kinds(self.arun)
        return self.__cached_param_names_of_arun

    def _get_param_names_of_run(self) -> Dict[Parameter, List[str]]:
        """
        The symmetric method of `get_input_param_names` for the `run` method. For internal use only.
        """
        if self.__cached_param_names_of_run is None:
            self.__cached_param_names_of_run = get_param_names_all_kinds(self.run)
        return self.__cached_param_names_of_run

    @property
    def parent(self) -> "Automa":
        return self.__parent

    @parent.setter
    def parent(self, value: "Automa"):
        self.__parent = value

    @property
    def output_buffer(self) -> Any:
        if not self.__output_has_set:
            raise RuntimeError(f"output of worker is not ready yet")
        return copy.deepcopy(self.__output_buffer)
    
    @output_buffer.setter
    def output_buffer(self, value: Any):
        self.__output_buffer = value
        self.__output_has_set = True

    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["output_has_set"] = self.__output_has_set
        if self.__output_has_set:
            state_dict["output_buffer"] = self.__output_buffer
        return state_dict

    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__output_has_set = state_dict["output_has_set"]
        if self.__output_has_set:
            self.__output_buffer = state_dict["output_buffer"]

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
