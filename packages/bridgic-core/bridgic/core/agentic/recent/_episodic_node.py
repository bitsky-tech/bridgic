from abc import ABC
from enum import Enum
from typing import List, Dict, Any, Optional
from typing_extensions import override
from datetime import datetime

from bridgic.core.types._serialization import Serializable
from bridgic.core.model.types import Message


class NodeType(str, Enum):
    GOAL = "goal"
    LEAF = "leaf"
    COMPRESSED = "compressed"


class BaseEpisodicNode(Serializable, ABC):
    """
    BaseEpisodicNode represents a single memory unit in the memory sequence in the ReCENT Algorithm.
    """

    node_type: NodeType
    """The type of the node."""

    timestep: int
    """The timestep of the node."""
    timestamp: str
    """The timestamp of the node."""

    def __init__(self, timestep: int):
        self.timestep = timestep
        self.timestamp = datetime.now().isoformat()

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "timestep": self.timestep,
            "timestamp": self.timestamp,
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.node_type = NodeType(state_dict["node_type"])
        self.timestep = state_dict["timestep"]
        self.timestamp = state_dict["timestamp"]


class GoalEpisodicNode(BaseEpisodicNode):

    goal: str
    """The content of the goal."""
    guidance: str
    """The guidance to achieve the goal."""

    shifted_goal_node_timestep: int
    """The timestep of the shifted goal node."""

    def __init__(
        self,
        timestep: int,
        goal: str,
        guidance: Optional[str] = None,
        shifted_goal_node_timestep: Optional[int] = None,
    ):
        super().__init__(timestep)
        self.node_type = NodeType.GOAL
        self.goal = goal
        self.guidance = guidance if guidance is not None else ""
        self.shifted_goal_node_timestep = shifted_goal_node_timestep if shifted_goal_node_timestep is not None else -1

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        result = super().dump_to_dict()
        result["content"] = self.goal
        result["shifted_goal_node_timestep"] = self.shifted_goal_node_timestep
        return result

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.goal = state_dict["content"]
        self.shifted_goal_node_timestep = state_dict.get("shifted_goal_node_timestep")

class LeafEpisodicNode(BaseEpisodicNode):

    messages: List[Message]
    """The original messages of the leaf episodic node."""

    message_appendable: bool
    """Whether new message could be appended to the leaf episodic node."""

    def __init__(self, timestep: int, messages: Optional[List[Message]] = None):
        super().__init__(timestep)
        self.node_type = NodeType.LEAF
        self.messages = messages if messages is not None else []
        self.message_appendable = True

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        result = super().dump_to_dict()
        result["messages"] = self.messages
        result["message_appendable"] = self.message_appendable
        return result

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.messages = state_dict.get("messages")
        self.message_appendable = state_dict.get("message_appendable")

class CompressedEpisodicNode(BaseEpisodicNode):

    summary: str
    """The summary of the compressed episodic node."""

    compressed_node_timesteps: List[int]
    """The timesteps of the compressed episodic node."""

    def __init__(self, timestep: int, summary: str, compressed_timesteps: Optional[List[int]] = None):
        super().__init__(timestep)
        self.summary = summary
        self.compressed_node_timesteps = compressed_timesteps if compressed_timesteps is not None else []

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        result = super().dump_to_dict()
        result["summary"] = self.summary
        result["compressed_node_timesteps"] = self.compressed_node_timesteps
        return result

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.summary = state_dict.get("summary", "")
        self.compressed_node_timesteps = state_dict.get("compressed_node_timesteps", [])
