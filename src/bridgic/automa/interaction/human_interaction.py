"""
[Bridgic Human Interaction Mechanism]

This module contains the base classes and types for the Bridgic human interaction mechanism.

"""

from typing import List
from bridgic.automa.serialization import Snapshot
from bridgic.automa.interaction.event_handling import Feedback, Event
from pydantic import BaseModel

class Interaction(BaseModel):
    interaction_id: str
    event: Event

class InteractionException(Exception):
    @property
    def interactions(self) -> List[Interaction]:
        ...
        #TODO: implement...

    @property
    def snapshot(self) -> Snapshot:
        ...
        #TODO: implement...

class InteractionFeedback(Feedback):
    interaction_id: str
