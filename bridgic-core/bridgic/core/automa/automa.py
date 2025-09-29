import asyncio
import uuid
import threading

from typing import List, Any, Optional, Dict
from typing_extensions import override
from abc import ABCMeta, abstractmethod
from inspect import Parameter, _ParameterKind
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa.worker import Worker
from bridgic.core.automa.interaction import Event, FeedbackSender, EventHandlerType, InteractionFeedback, Feedback, Interaction, InteractionException
from bridgic.core.automa.serialization import Snapshot
from bridgic.core.automa.arguments_descriptor import RuntimeContext
from bridgic.core.utils import msgpackx
from bridgic.core.utils.inspect_tools import get_param_names_by_kind
from bridgic.core.types.error import AutomaRuntimeError

class RunningOptions(BaseModel):
    debug: bool = False

class _InteractionAndFeedback(BaseModel):
    interaction: Interaction
    feedback: Optional[InteractionFeedback] = None

class _InteractionEventException(Exception):
    """
    Exception raised when the `interact_with_human` method is called.
    For internal use only.
    `Interaction` obects are stored in `self.args` of the exception.
    """

class Automa(Worker):
    _running_options: RunningOptions

    # For event handling.
    _event_handlers: Dict[str, EventHandlerType]
    _default_event_handler: EventHandlerType

    # For human interaction.
    _worker_interaction_indices: Dict[str, int]

    # Ongoing human interactions triggered by the `interact_with_human()` call from workers of the current Automa.
    # worker_key -> list of interactions.
    _ongoing_interactions: Dict[str, List[_InteractionAndFeedback]]

    _thread_pool: ThreadPoolExecutor
    _main_thread_id: int
    _main_loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        name: str = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        super().__init__()

        # Set the name of the Automa instance.
        self.name = name or f"automa-{uuid.uuid4().hex[:8]}"

        # Initialize the shared running options.
        self._running_options = RunningOptions()

        # Initialize data structures for event handling and human interactions
        self._event_handlers = {}
        self._default_event_handler = None
        self._worker_interaction_indices = {}
        self._ongoing_interactions = {}

        self._thread_pool = thread_pool

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["name"] = self.name
        state_dict["running_options"] = self._running_options
        state_dict["ongoing_interactions"] = self._ongoing_interactions
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.name = state_dict["name"]
        self._running_options = state_dict["running_options"]

        self._event_handlers = {}
        self._default_event_handler = None
        self._worker_interaction_indices = {}
        self._ongoing_interactions = state_dict["ongoing_interactions"]
        self._thread_pool = None

    @classmethod
    def load_from_snapshot(
        cls, 
        snapshot: Snapshot,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ) -> "Automa":
        # Here you can compare snapshot.serialization_version with SERIALIZATION_VERSION, and handle any necessary version compatibility issues if needed.
        automa = msgpackx.load_bytes(snapshot.serialized_bytes)
        if thread_pool:
            automa.thread_pool = thread_pool
        return automa

    @property
    def thread_pool(self) -> Optional[ThreadPoolExecutor]:
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, executor: ThreadPoolExecutor) -> None:
        """
        Set the thread pool for parallel running of I/O-bound tasks.

        If an Automa is nested within another Automa, the thread pool of the top-level Automa will be used, rather than the thread pool of the nested Automa.
        """
        self._thread_pool = executor

    @abstractmethod
    def _locate_interacting_worker(self) -> Optional[str]:
        """
        Locate the worker that is currently interacting with human.

        Returns
        -------
        Optional[str]
            The necessary identifier of the worker that is currently interacting with human.
        """
        ...

    @abstractmethod
    def _get_worker_key(self, worker: Worker) -> Optional[str]:
        """
        Identify the worker key by the worker instance.
        """
        ...

    @abstractmethod
    def _get_worker_instance(self, worker_key: str) -> Worker:
        """
        Get the worker instance by the worker key.
        """
        ...

    def is_top_level(self) -> bool:
        """
        Check if the current automa is the top-level automa.

        Returns
        -------
        bool
            True if the current automa is the top-level automa, False otherwise.
        """
        return self.parent is None

    def set_running_options(self, debug: bool = None):
        """
        Set running options for this Automa instance, and ensure these options propagate through (penetrate) all nested 
        Automa instances. For each option, if its value is None, the original value will be retained and not overwritten.

        When an option is set multiple times across different nested Automa instances, the setting from the outermost 
        (top-level) Automa will override the settings of all inner (nested) Automa instances.

        For example, if the top-level Automa instance sets `debug = True` and the nested instances sets `debug = False`, 
        then the nested Automa instance will run in debug mode, when the top-level Automa instance is executed.

        Parameters
        ----------
        debug : bool, optional
            Whether to enable debug mode. If not set, the effect is the same as setting `debug = False` by default.
        """
        if debug is not None:
            self._running_options.debug = debug

    def _get_top_running_options(self) -> RunningOptions:
        if self.parent is None:
            # Here we are at the top-level automa.
            return self._running_options
        return self.parent._get_top_running_options()

    ###############################################################
    ########## [Bridgic Event Handling Mechanism] starts ##########
    ###############################################################

    def register_event_handler(self, event_type: Optional[str], event_handler: EventHandlerType) -> None:
        """
        Register an event handler for the specified event type.

        Note: Only event handlers registered on the top-level Automa will be invoked to handle events.

        Parameters
        ----------
        event_type: Optional[str]
            The type of event to be handled. If set to None, the event handler will be registered as the default handler and will be used to handle all event types.
        event_handler: EventHandlerType
            The event handler to be registered.
        """
        if event_type is None:
            self._default_event_handler = event_handler
        else:
            self._event_handlers[event_type] = event_handler

    def unregister_event_handler(self, event_type: Optional[str]) -> None:
        """
        Unregister an event handler for the specified event type.

        Parameters
        ----------
        event_type: Optional[str]
            The type of event to be unregistered. If set to None, the default event handler will be unregistered.
        """
        if event_type in self._event_handlers:
            del self._event_handlers[event_type]
        if event_type is None:
            self._default_event_handler = None

    def unregister_all_event_handlers(self) -> None:
        """
        Unregister all event handlers.
        """
        self._event_handlers.clear()
        self._default_event_handler = None

    class _FeedbackSender(FeedbackSender):
        def __init__(
                self, 
                future: asyncio.Future[Feedback],
                post_loop: asyncio.AbstractEventLoop,
                ):
            self._future = future
            self._post_loop = post_loop
        
        def send(self, feedback: Feedback) -> None:
            try:
                current_loop = asyncio.get_running_loop()
            except Exception:
                current_loop = None
            try:
                if current_loop is self._post_loop:
                    self._future.set_result(feedback)
                else:
                    self._post_loop.call_soon_threadsafe(self._future.set_result, feedback)
            except asyncio.InvalidStateError:
                # Suppress the InvalidStateError to be raised, maybe due to timeout.
                import warnings
                warnings.warn(f"Feedback future already set. feedback: {feedback}", FutureWarning)

    @override
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
        def _handler_need_feedback_sender(handler: EventHandlerType):
            positional_param_names = get_param_names_by_kind(handler, Parameter.POSITIONAL_ONLY) + get_param_names_by_kind(handler, Parameter.POSITIONAL_OR_KEYWORD)
            var_positional_param_names = get_param_names_by_kind(handler, Parameter.VAR_POSITIONAL)
            return len(var_positional_param_names) > 0 or len(positional_param_names) > 1

        if self.parent is not None:
            # Bubble up the event to the top-level Automa.
            return self.parent.post_event(event)

        # Here is the top-level Automa.
        # Call event handlers
        if event.event_type in self._event_handlers:
            if _handler_need_feedback_sender(self._event_handlers[event.event_type]):
                self._event_handlers[event.event_type](event, feedback_sender=None)
            else:
                self._event_handlers[event.event_type](event)
        if self._default_event_handler is not None:
            if _handler_need_feedback_sender(self._default_event_handler):
                self._default_event_handler(event, feedback_sender=None)
            else:
                self._default_event_handler(event)

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
        if threading.get_ident() == self._main_thread_id:
            raise AutomaRuntimeError(
                f"`request_feedback` should only be called in a different thread from the main thread of the {self.name}. "
            )
        return asyncio.run_coroutine_threadsafe(
            self.request_feedback_async(event, timeout),
            self._main_loop
        ).result()

    async def request_feedback_async(
        self, 
        event: Event,
        timeout: Optional[float] = None
    ) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        The event handler implemented by the application layer will be called in the next event loop, in the main thread.

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
        if self.parent is not None:
            # Bubble up the event to the top-level Automa.
            return await self.parent.request_feedback_async(event, timeout)
        
        # Here is the top-level Automa.
        event_loop = asyncio.get_running_loop()
        future = event_loop.create_future()
        feedback_sender = self._FeedbackSender(future, event_loop)
        # Call event handlers
        if event.event_type in self._event_handlers:
            self._event_handlers[event.event_type](event, feedback_sender)
        if self._default_event_handler is not None:
            self._default_event_handler(event, feedback_sender)

        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as e:
            # When python >= 3.11 here.
            raise TimeoutError(f"No feedback is received before timeout in Automa[{self.name}]") from e
        except asyncio.TimeoutError as e:
            # Version compatibility resolution: asyncio.wait_for raises asyncio.TimeoutError before python 3.11.
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
            raise TimeoutError(f"No feedback is received before timeout in Automa[{self.name}]") from e

    ###############################################################
    ########### [Bridgic Event Handling Mechanism] ends ###########
    ###############################################################

    ###############################################################
    ######## [Bridgic Human Interaction Mechanism] starts #########
    ###############################################################

    def interact_with_human(self, event: Event) -> InteractionFeedback:
        """
        Trigger an interruption in the "human-computer interaction" during the execution of Automa.

        Parameters
        ----------
        event: Event
            The event that triggered the interaction.

        Returns
        -------
        InteractionFeedback
            The feedback received from the application layer.

        Raises
        ------
        _InteractionEventException
            If the Automa is not the top-level Automa and the `interact_with_human()` method is called by 
            one or more workers, this exception will be raised to the upper level Automa.
        """
        kickoff_worker_key: str = self._locate_interacting_worker()
        if kickoff_worker_key:
            return self.interact_with_human_from_worker_key(event, kickoff_worker_key)
        raise AutomaRuntimeError(
            f"Get kickoff worker failed in Automa[{self.name}] "
            f"when trying to interact with human with event: {event}"
        )

    def interact_with_human_from_worker(
        self,
        event: Event,
        worker: Worker
    ) -> InteractionFeedback:
        worker_key = self._get_worker_key(worker)
        if worker_key:
            return self.interact_with_human_from_worker_key(event, worker_key)
        raise AutomaRuntimeError(
            f"Not found worker[{worker}] in Automa[{self.name}] "
            f"when trying to interact with human with event: {event}"
        )

    def interact_with_human_from_worker_key(
        self,
        event: Event,
        worker_key: str
    ) -> InteractionFeedback:
        # Match interaction_feedback to see if it matches
        matched_feedback: _InteractionAndFeedback = None
        cur_interact_index = self._get_and_increment_interaction_index(worker_key)
        if worker_key in self._ongoing_interactions:
            interaction_and_feedbacks = self._ongoing_interactions[worker_key]
            if cur_interact_index < len(interaction_and_feedbacks):
                matched_feedback = interaction_and_feedbacks[cur_interact_index]
                # Check the event type
                if event.event_type != matched_feedback.interaction.event.event_type:
                    raise AutomaRuntimeError(
                        f"Event type mismatch! Automa[{self.name}-worker[{worker_key}]]. "
                        f"interact_with_human passed-in event: {event}\n"
                        f"ongoing interaction && feedback: {matched_feedback}\n"
                    )
        if matched_feedback is None or matched_feedback.feedback is None:
            # Important: The interaction_id should be unique for each human interaction.
            interaction_id = uuid.uuid4().hex if matched_feedback is None else matched_feedback.interaction.interaction_id
            # Match interaction_feedback failed, raise an exception to go into the human interactioin process.
            raise _InteractionEventException(Interaction(
                interaction_id=interaction_id,
                event=event,
            ))
        else:
            # Match interaction_feedback succeeded, return it.
            return matched_feedback.feedback

    def _get_and_increment_interaction_index(self, worker_key: str) -> int:
        if worker_key not in self._worker_interaction_indices:
            cur_index = 0
            self._worker_interaction_indices[worker_key] = 0
        else:
            cur_index = self._worker_interaction_indices[worker_key]
        self._worker_interaction_indices[worker_key] += 1
        return cur_index

    ###############################################################
    ######### [Bridgic Human Interaction Mechanism] ends ##########
    ###############################################################

    def get_local_space(self, runtime_context: RuntimeContext) -> Dict[str, Any]:
        """
        Retrieve the local execution context (local space) associated with the current worker. 
        If you require the local space to be cleared after the completion of `automa.arun()`, 
        you may customize this behavior by overriding the `should_reset_local_space()` method.

        Parameters
        ----------
        runtime_context : RuntimeContext
            The runtime context.

        Returns
        -------
        Dict[str, Any]
            The local space.
        """
        worker_key = runtime_context.worker_key
        worker_obj = self._get_worker_instance(worker_key)
        return worker_obj.local_space

    def should_reset_local_space(self) -> bool:
        """
        This method indicates whether to reset the local space at the end of the arun method of Automa. 
        By default, it returns True, standing for resetting. Otherwise, it means doing nothing.
        
        Examples:
        --------
        ```python
        class MyAutoma(Automa):
            def should_reset_local_space(self) -> bool:
                return False
        ```
        """
        return True
