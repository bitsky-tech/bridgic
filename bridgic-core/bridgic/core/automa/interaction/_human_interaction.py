"""
[Bridgic Human Interaction Mechanism]

This module contains the base classes and types for the Bridgic human interaction mechanism.

The human interaction mechanism provides the canonical "Human-in-the-loop" interaction pattern in Bridgic, enabling effective collaboration between Automas and humans within Agentic Systems.

The human interaction mechanism also enables communication between workers inside the Automa and the application layer outside the Automa. However, it differs from the event handling mechanism in several important ways:
- If the time interval between a worker sending an event and receiving feedback from the application layer is relatively short (for example, a few seconds or minutes), the event handling mechanism may be sufficient. In contrast, the human interaction mechanism is designed to support scenarios where user feedback may take a long time (such as several hours or even days) to arrive.
- If there is a persistent bidirectional connection between the user and the Automa process (such as a WebSocket), the event handling mechanism is usually sufficient. However, if there are information channels between the user and the Agentic system (such as email systems, IM platforms, or certain approval workflow systems), then only the human interaction mechanism can be used.
- The event handling mechanism requires the Automa process to remain alive in memory for the entire duration of the event and feedback interaction. In contrast, with the human interaction mechanism, when the Automa is waiting for user feedback, states of the entire Automa are serialized and persisted to external storage, and the Automa process itself can be completely shut down. Once user feedback arrives, the Automa process can be restarted, its state restored from external storage, and the interaction can continue from where it left off.
- When communicating between the inside and outside of the Automa, the event handling mechanism can be used for both one-way information transfer (such as sending a ProgressEvent) and two-way communication (such as sending an event and receiving feedback). In contrast, the human interaction mechanism is primarily designed for two-way communication between the Automa and the application layer.

[Description of the whole process of human interaction]

In the human interaction mechanism, a worker inside an Automa can initiate an interaction with a user by calling the `interact_with_human` method. This method can be called from both asynchronous and non-asynchronous (normal) methods. When `interact_with_human` is called, the Automa will immediately pause, serialize its current state, and raise an `InteractionException`. 

If multiple `interact_with_human` calls are made concurrently by different workers (for example, in parallel branches of the graph), all resulting interactions will be collected and included in the `InteractionException`. Each interaction will have its own `interaction_id`, which uniquely identifies it.

When the application layer catches an `InteractionException`, it receives a list of `Interaction`s that occurred during the most recent event loop, along with a `Snapshot` of the Automa's current state. The application layer should then save this snapshot to external storage and present a user-friendly interface (through some information channels) that allows the user to interact with the Automa remotely and asynchronously. 

When user feedback is received, the Automa's state is restored from the `Snapshot` retrived from external storage. The application layer should then call the `Automa.load_from_snapshot()` method to deserialize the Automa instance. After deserialization, the Automa's `arun` method is called to resume execution.

When `arun` is called, the application layer should provide an argument named `interaction_feedback` to this method. This argument must contain both the feedback data supplied by the user and the `interaction_id`, which uniquely identifies the specific interaction. If multiple interactions occurred simultaneously before the Automa was paused, the application layer should instead provide an argument named `interaction_feedbacks`, which should be a list of `InteractionFeedback` objects, allowing all feedback to be returned at once. Once a `interaction_feedback` or `interaction_feedbacks` argument is provided, there is no need to provide the other arguments again.

After that, the Automa will resume execution from the worker that was previously paused. However, it is important to note that the paused worker will be re-executed from the beginning. As a result, any code in the worker that appears before the call to `interact_with_human` will be executed again, so this part of the code must be idempotent. When `interact_with_human` is re-executed, it will return the correct `InteractionFeedback` as its return value. The worker can then use this feedback to continue executing the subsequent logic.

"""

from typing import List
from datetime import datetime
from bridgic.core.automa._serialization import Snapshot
from bridgic.core.automa.interaction._event_handling import Feedback, Event
from pydantic import BaseModel

class Interaction(BaseModel):
    """
    Represents a single interaction between the Automa and a human. 
    Each call to `interact_with_human` will generate an `Interaction` object.

    Fields
    ------
    interaction_id: str
        The unique identifier for the interaction.
    event: Event
        The event that triggered the interaction.
    """
    interaction_id: str
    event: Event

class InteractionException(Exception):
    """
    Exception raised when the `interact_with_human` method is called and a human interaction is triggered.
    """
    _interactions: List[Interaction]
    _snapshot: Snapshot

    def __init__(self, interactions: List[Interaction], snapshot: Snapshot):
        self._interactions = interactions
        self._snapshot = snapshot

    @property
    def interactions(self) -> List[Interaction]:
        """
        Returns a list of `Interaction` objects that occurred during the most recent event loop.

        Multiple `Interaction` objects may be generated because, within the same event loop, several workers calling the `interact_with_human` method might be running concurrently in parallel branches of the graph.
        """
        return self._interactions

    @property
    def snapshot(self) -> Snapshot:
        """
        Returns a `Snapshot` of the Automa's current state.
        The serialization is automatically triggered by the `interact_with_human` method.
        """
        return self._snapshot

class InteractionFeedback(Feedback):
    """
    A feedback object that contains both the data provided by the user and the `interaction_id`, which uniquely identifies the corresponding interaction.
    """
    interaction_id: str
    timestamp: datetime = datetime.now()
