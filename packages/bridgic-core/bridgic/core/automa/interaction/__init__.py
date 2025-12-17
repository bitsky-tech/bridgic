"""
The Interaction module provides mechanisms for handling human-machine interactions in Automa.

This module contains several important interface definitions for implementing event handling, 
feedback collection, and interaction control during Automa execution.

There are two fundamental mechanisms for human-machine interactions in Automa:

- **Feedback Request Mechanism**: For simple interaction scenarios during Automa execution.
    - [`request_feedback_async()`](../../../../../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa.request_feedback_async), [`request_feedback()`](../../../../../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa.request_feedback)
- **Human Interaction Mechanism**: For long-running interaction scenarios that require interruption and resumption during Automa execution.
    - [`interact_with_human()`](../../../../../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa.interact_with_human), [`load_from_snapshot()`](../../../../../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa.load_from_snapshot)
"""

from ._event_handling import Event, Feedback, FeedbackSender, EventHandlerType    
from ._human_interaction import InteractionFeedback, InteractionException, Interaction

__all__ = [
    "Event",
    "Feedback",
    "InteractionFeedback",
    "FeedbackSender",
    "EventHandlerType",
    "InteractionException",
    "Interaction",
]
