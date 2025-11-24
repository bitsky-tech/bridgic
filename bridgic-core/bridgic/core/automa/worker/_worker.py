import copy
import asyncio
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Tuple
from typing_extensions import override
from functools import partial
from inspect import _ParameterKind
from bridgic.core.automa.interaction import Event, InteractionFeedback, Feedback
from bridgic.core.types._error import WorkerRuntimeError
from bridgic.core.types._serialization import Serializable
from bridgic.core.utils._inspect_tools import get_param_names_all_kinds
from bridgic.core.utils._args_map import safely_map_args

if TYPE_CHECKING:
    from bridgic.core.automa._automa import Automa

class Worker(Serializable):
    """
    This class is the base class for all workers.

    `Worker` has two methods that may be overridden by the subclass:

    1. `arun()`: This asynchronous method should be implemented when your worker 
    does not require almost immediately scheduling after all its task dependencies 
    are fulfilled, and when overall workflow is not sensitive to the fair sharing 
    of CPU resources between workers. If workers can afford to retain and occupy 
    execution resources for their entire execution duration, and there is no 
    explicit need for fair CPU time-sharing or timely scheduling, you should 
    implement `arun()` and allow workers to run to completion as cooperative tasks 
    within the event loop.

    2. `run()`: This synchronous method should be implemented when either of the 
    following holds:

        - a. The automa includes other workers that require timely access to CPU 
        resources (for example, workers that must respond quickly or are sensitive 
        to scheduling latency).

        - b. The current worker itself should be scheduled as soon as all its task 
        dependencies are met, to maintain overall workflow responsiveness. In these 
        cases, `run()` enables the framework to offload your worker to a thread pool, 
        ensuring that CPU time is shared fairly among all such workers and the event 
        loop remains responsive.

    In summary, if you are unsure whether your task require quickly scheduling or not, 
    it is recommended to implement the `arun()` method. Otherwise, implement the 
    `run()` **ONLY** if you are certain that you agree to share CPU time slices 
    with other workers.
    """

    # TODO : Maybe process pool of the Automa is needed.

    __parent: "Automa"
    __local_space: Dict[str, Any]
    
    # Cached method signatures, with no need for serialization.
    __cached_param_names_of_arun: Dict[_ParameterKind, List[Tuple[str, Any]]]
    __cached_param_names_of_run: Dict[_ParameterKind, List[Tuple[str, Any]]]

    def __init__(self):
        self.__parent = self
        self.__local_space = {}

        # Cached method signatures, with no need for serialization.
        self.__cached_param_names_of_arun = None
        self.__cached_param_names_of_run = None

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        """
        The asynchronous method to run the worker.
        """
        loop = asyncio.get_running_loop()
        topest_automa = self._get_top_level_automa()
        if topest_automa:
            thread_pool = topest_automa.thread_pool
            if thread_pool:
                rx_param_names_dict = self.get_input_param_names()
                rx_args, rx_kwargs = safely_map_args(args, kwargs, rx_param_names_dict)
                # kwargs can only be passed by functools.partial.
                return await loop.run_in_executor(thread_pool, partial(self.run, *rx_args, **rx_kwargs))

        # Unexpected: No thread pool is available.
        # Case 1: the worker is not inside an Automa (uncommon case).
        # Case 2: no thread pool is setup by the top-level automa.
        raise WorkerRuntimeError(f"No thread pool is available for the worker {type(self)}")

    def run(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        """
        The synchronous method to run the worker.
        """
        raise NotImplementedError(f"run() is not implemented in {type(self)}")

    def is_top_level(self) -> bool:
        """
        Check if the current worker is the top-level worker.

        Returns
        -------
        bool
            True if the current worker is the top-level worker (parent is self), False otherwise.
        """
        return self.parent is self

    def _get_top_level_automa(self) -> Optional["Automa"]:
        """
        Get the top-level automa instance reference.
        """
        # If the current automa is the top-level automa, return itself.
        from bridgic.core.automa._automa import Automa
        if isinstance(self, Automa):
            top_level_automa = self
        else:
            top_level_automa = self.parent
        while top_level_automa and (not top_level_automa.is_top_level()):
            top_level_automa = top_level_automa.parent
        return top_level_automa
    
    def get_input_param_names(self) -> Dict[_ParameterKind, List[Tuple[str, Any]]]:
        """
        Get the names of input parameters of the worker.
        Use cached result if available in order to improve performance.
        
        This method intelligently detects whether the user has overridden the `run` method
        or is using the default `arun` method, and returns the appropriate parameter signature.

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
        # Check if user has overridden the arun method
        if self._is_arun_overridden():
            # User overrode arun method, return arun method parameters
            if self.__cached_param_names_of_arun is None:
                self.__cached_param_names_of_arun = get_param_names_all_kinds(self.arun)
            return self.__cached_param_names_of_arun
        else:
            # User is using run method, return run method parameters
            if self.__cached_param_names_of_run is None:
                self.__cached_param_names_of_run = get_param_names_all_kinds(self.run)
            return self.__cached_param_names_of_run

    def _is_arun_overridden(self) -> bool:
        """
        Check if the user has overridden the arun method.
        """
        # Compare method references - much faster than inspect.getsource()
        return self.arun.__func__ is not Worker.arun

    @property
    def parent(self) -> "Automa":
        return self.__parent

    @parent.setter
    def parent(self, value: "Automa"):
        self.__parent = value

    @property
    def local_space(self) -> Dict[str, Any]:
        return self.__local_space

    @local_space.setter
    def local_space(self, value: Dict[str, Any]):
        self.__local_space = value
    
    def get_report_info(self) -> Dict[str, Any]:
        report_info = {}
        report_info["local_space"] = self.__local_space
        return report_info

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["local_space"] = self.__local_space
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__parent = self
        self.__local_space = state_dict["local_space"]
        
        # Cached method signatures, with no need for serialization.
        self.__cached_param_names_of_arun = None
        self.__cached_param_names_of_run = None
        
    def ferry_to(self, key: str, /, *args, **kwargs):
        """
        Handoff control flow to the specified worker, passing along any arguments as needed.
        The specified worker will always start to run asynchronously in the next event loop, regardless of its dependencies.

        Parameters
        ----------
        key : str
            The key of the worker to run.
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.
        """
        if self.is_top_level():
            raise WorkerRuntimeError(f"`ferry_to` method can only be called by a worker inside an Automa")
        self.parent.ferry_to(key, *args, **kwargs)

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
        if self.is_top_level():
            raise WorkerRuntimeError(f"`post_event` method can only be called by a worker inside an Automa")
        self.parent.post_event(event)

    def request_feedback(
        self, 
        event: Event,
        timeout: Optional[float] = None
    ) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        Note that `post_event` should only be called from within a non-async method running in the new thread of the Automa thread pool.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        timeout: Optional[float]
            A float or int number of seconds to wait for if the feedback is not received. If None, then there is no limit on the wait time.

        Returns
        -------
        Feedback
            The feedback received from the application layer.

        Raises
        ------
        TimeoutError
            If the feedback is not received before the timeout. Note that the raised exception is the built-in `TimeoutError` exception, instead of asyncio.TimeoutError or concurrent.futures.TimeoutError!
        """
        if self.is_top_level():
            raise WorkerRuntimeError(f"`request_feedback` method can only be called by a worker inside an Automa")
        return self.parent.request_feedback(event, timeout)

    async def request_feedback_async(
        self, 
        event: Event,
        timeout: Optional[float] = None
    ) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        The event handler implemented by the application layer will be called in the next event loop, in the main thread.

        Note that `post_event` should only be called from within an asynchronous method running in the main event loop of the top-level Automa.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        timeout: Optional[float]
            A float or int number of seconds to wait for if the feedback is not received. If None, then there is no limit on the wait time.

        Returns
        -------
        Feedback
            The feedback received from the application layer.

        Raises
        ------
        TimeoutError
            If the feedback is not received before the timeout. Note that the raised exception is the built-in `TimeoutError` exception, instead of asyncio.TimeoutError!
        """
        if self.is_top_level():
            raise WorkerRuntimeError(f"`request_feedback_async` method can only be called by a worker inside an Automa")
        return await self.parent.request_feedback_async(event, timeout)

    def interact_with_human(self, event: Event) -> InteractionFeedback:
        if self.is_top_level():
            raise WorkerRuntimeError(f"`interact_with_human` method can only be called by a worker inside an Automa")
        return self.parent.interact_with_human(event, self)
