"""
[Bridgic Event Handling Mechanism]

This module contains the base classes and types for the Bridgic event handling mechanism.

The event handling mechanism enables (quasi) real-time communication between workers inside the Automa and the application layer outside the Automa. Events can be sent from a worker to the application layer, and feedback can be sent in the opposite direction, from the application layer back to the worker, if needed.

A worker can use the `post_event()` method to send an event to the application layer. To handle these events, the application layer can register event handlers by calling the Automa's `register_event_handler()` method. Since an Automa can be nested inside another Automa, only the event handlers registered on the top-level Automa will be invoked to handle events.

Note that there are two types of events: those that require feedback from the application layer, and those that do not. An example of the former is `ProgressEvent`, which is used to send progress information from a worker to the application layer. An example of the latter is an event that asks the user for additional input.

If an event requires feedback, the application layer can provide feedback to the worker by invoking the `send` method of the `FeedbackSender` object. The worker will then `await` the future object returned by the `post_event` method to receive the feedback.

The Bridgic event handling mechanism requires the Automa process to remain alive in memory for the entire duration of the event and feedback interaction. If the interaction period is too long and the Automa process needs to be shut down in the middle, the Bridgic human interaction mechanism should be used instead.

"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Any, Callable, Union
from typing_extensions import TypeAlias
from abc import ABC, abstractmethod
from bridgic.core.types.common import ZeroToOne

class Event(BaseModel):
    """
    An event is a message that is sent from one worker inside the Automa to the application layer outside the Automa.

    Fields
    ------
    event_type: Optional[str]
        The type of the event. The type of the event is used to identify the event handler registered to handle the event.
    timestamp: datetime
        The timestamp of the event.
    data: Optional[Any]
        The data attached to the event.
    """
    event_type: Optional[str] = None
    timestamp: datetime = datetime.now()
    data: Optional[Any] = None

class ProgressEvent(Event):
    """
    A progress event is an event that indicates the progress of a worker task.

    Fields
    ------
    progress: ZeroToOne
        The progress of the task, represented as a value between 0 and 1.
    """
    progress: ZeroToOne

class Feedback(BaseModel):
    """
    A feedback is a message that is sent from the application layer outside the Automa to a worker inside the Automa.

    Fields
    ------
    data: Any
        The data attached to the feedback.
    """
    data: Any

class FeedbackSender(ABC):
    """
    The appliction layer must use `FeedbackSender` to send back feedback to the worker inside the Automa.
    """
    @abstractmethod
    def send(self, feedback: Feedback) -> None:
        """
        Send feedback to the Automa.
        This method can be called only once for each event.

        This `send` method can be safely called in several different scenarios:
        - In the same asyncio Task of the same event loop as the event handler.
        - In a different asyncio Task of the same event loop as the event handler.
        - In a different thread from the event handler.

        Parameters
        ----------
        feedback: Feedback
            The feedback to be sent.
        """
        ...

EventHandlerType: TypeAlias = Union[Callable[[Event, FeedbackSender], None], Callable[[Event], None]]


