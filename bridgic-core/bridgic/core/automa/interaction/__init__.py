"""
The Interaction module provides human-machine interaction mechanisms for Automa.

This module contains several important interface definitions for implementing event 
listening, feedback collection, and interactive control during Automa execution.

There are two fundamental mechanisms for human-machine interaction in Automa:
- [Event and Feedback Mechanism]: For simple interaction scenarios during Automa execution.
- [Human Interaction Mechanism]: For longer-running interaction scenarios that require 
  interruption and resumption during Automa execution.
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
