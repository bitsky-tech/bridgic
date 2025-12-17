"""
[Bridgic Human Interaction Mechanism]

This module defines the core types for Bridgic's human interaction mechanism, which 
empowers seamless collaboration between Automas and human users within Agentic Systems.

**Human Interaction Process Overview:**

When an Automa worker requires to obtain human's input, it initiates the process by invoking 
the `interact_with_human` method in both async and synchronous contexts. This call 
instantaneously pauses the execution of the Automa, serializes all relevant state, and raises 
an `InteractionException`. If more than one `interact_with_human` were called concurrently 
(for example, on different workers that execute in parallel branches), all interactions are 
aggregated and included within a single raised `InteractionException`, each could be associated 
with its own `interaction_id`.

When the application layer catches an `InteractionException`, it receives both the list of 
pending `Interaction` objects from the latest dynamic step and a `Snapshot` object representing 
the state of the Automa. It is recommended to persist this snapshot externally and then expose 
user-facing interfaces through appropriate communication channels to collect human's feedbacks.

Receiving the necessary feedbacks from the application layer, the Automa object can reload 
itself by calling `load_from_snapshot` with the pre-generated `Snapshot` object. Then the 
execution can be resumed by calling the `arun` method. When resuming, feedback is passed through 
the `feedback_data` parameter (containing the data provided by the user and the 
`interaction_id`). In cases with multiple simultaneous interactions, the `feedback_data` 
parameter is also used as a list of `InteractionFeedback` objects.

The execution of the Automa instance will resume from the worker where the interruption 
occurred. The entire worker function will re-execute from the top; all code prior to 
the `interact_with_human` invocation will be run again. Therefore, all code up to the 
interaction point should be written without side effects. When execution reaches 
`interact_with_human` again, it will return the appropriate feedback results, allowing 
the workflow to continue based on the new user input.

**Differences between Human Interaction Mechanism and Event Handling Mechanism:**

While the human interaction mechanism also supports communication between Automa workers 
and the external application layer, it fundamentally differs from Bridgic’s event handling 
mechanism in several crucial ways:

- The event handling mechanism is suitable for interactions requiring a quick turnaround, 
typically seconds, such as progress updates. In contrast, the human interaction mechanism 
is purpose-built for cases where user input may be significantly delayed—ranging from 
minutes to hours to even days, such as email, instant messaging, or approval workflows.
- The event handling mechanism requires the Automa process to stay alive in memory throughout 
the event-feedback lifecycle. With the human interaction mechanism, once user input is needed, 
the complete state of the Automa is serialized and saved to external storage, allowing the 
process itself to be interrupted. When feedback is finally received, the Automa can be resumed 
from storage and the execution can pick up from where it left off.
- The event handling mechanism supports both one-way exchanges (e.g., progress updates) 
and two-way exchanges (e.g., requesting and receiving feedback). The human interaction 
mechanism, is primarily specialized for orchestrating two-way exchanges that may contain 
significant delays.
"""

from typing import List, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel
from bridgic.core.automa.interaction._event_handling import Feedback, Event

if TYPE_CHECKING:
    from bridgic.core.automa._automa import Snapshot

class Interaction(BaseModel):
    """
    An object that represents a single interaction between the Automa and a human. 
    Each call to `interact_with_human` will generate an `Interaction` object which will be included in the `InteractionException` raised.
    """
    interaction_id: str
    """ The unique identifier for the interaction."""
    event: Event
    """The event that triggered the interaction."""

class InteractionException(Exception):
    """
    An exception raised when the `interact_with_human` method is called one or more times within the latest event loop iteration, causing one or multiple human interactions to be triggered.
    """
    _interactions: List[Interaction]
    """The list of interactions that occurred during the latest event loop iteration."""
    _snapshot: "Snapshot"
    """The snapshot of the Automa's current state."""

    def __init__(self, interactions: List[Interaction], snapshot: "Snapshot"):
        self._interactions = interactions
        self._snapshot = snapshot

    @property
    def interactions(self) -> List[Interaction]:
        """
        A list of `Interaction` objects that occurred during the latest event loop iteration.

        Multiple `Interaction` objects may be generated because, within the latest event loop iteration, multiple workers calling the `interact_with_human` method might be running concurrently in parallel branches of the graph.
        """
        return self._interactions

    @property
    def snapshot(self) -> "Snapshot":
        """
        A `Snapshot` of the Automa's current state.
        The serialization is triggered automatically by the `interact_with_human` method.
        """
        return self._snapshot

class InteractionFeedback(Feedback):
    """
    A feedback object that contains both the data provided by the user and the `interaction_id`, which uniquely identifies the corresponding interaction.
    """
    interaction_id: str
    """ The unique identifier for the interaction."""
    timestamp: datetime = datetime.now()
    """The timestamp of the feedback."""