import asyncio
from typing import List, Dict, Any, Optional, Union, cast
from threading import RLock

from bridgic.core.model.types import Message
from bridgic.core.types._serialization import Serializable
from bridgic.core.agentic.recent._episodic_node import (
    BaseEpisodicNode,
    GoalEpisodicNode,
    LeafEpisodicNode,
    CompressionEpisodicNode,
    NodeType,
)


class EpisodicNodeTree(Serializable):
    """
    EpisodicNodeTree is a data structure responsible for managing the sequence of episodic memory nodes, 
    which is the core data structure of the ReCENT Algorithm.

    **ReCENT Algorithm** (Recursive Compressed Episodic Node Tree Algorithm) is an algorithm designed to 
    address issues such as context explosion and goal drift, by employing a recursive memory compression 
    mechanism. In this algorithm, each episodic node will serve as a container of memory and could be 
    tightly organized together to form a more efficient and reliable memory for the higher agentic system.

    Notes:
    ------
    - This data structure only supports appending new nodes; deletion or insertion is not allowed.
    - All write operations are protected by a lock to ensure atomicity and preserve ordered nature of the structure.
    - The data structure does not and should not perform any computationally expensive operations such as summarization.
    """

    _lock: RLock
    """Reentrant lock for thread-safe node write operations."""

    _node_sequence: List[BaseEpisodicNode]
    """The sequence of nodes in the episodic node tree."""

    _goal_node_timestep: int
    """The timestep of the current goal node. If no goal node exists, it will be -1."""
    _non_goal_node_timesteps: List[int]
    """The timesteps of the non-goal nodes."""

    def __init__(self):
        self._node_sequence = []
        self._goal_node_timestep = -1
        self._non_goal_node_timesteps = []
        self._lock = RLock()

    def get_node(self, timestep: int) -> Optional[BaseEpisodicNode]:
        """
        Get a node by its timestep.

        Parameters
        ----------
        timestep : int
            The timestep of the node.

        Returns
        -------
        Optional[BaseEpisodicNode]
            The node with the given timestep, or None if not found.
        """
        if timestep < len(self._node_sequence):
            return self._node_sequence[timestep]
        return None

    def get_goal_node(self) -> Optional[GoalEpisodicNode]:
        """
        Get the current goal node.

        Returns
        -------
        Optional[GoalEpisodicNode]
            The current goal node, or None if no goal node exists.
        """
        if self._goal_node_timestep != -1:
            return cast(GoalEpisodicNode, self.get_node(self._goal_node_timestep))
        return None

    def get_non_goal_nodes(self) -> List[BaseEpisodicNode]:
        """
        Get all directly accessible non-goal nodes (sorted by timestep).

        Returns
        -------
        List[BaseEpisodicNode]
            List of non-goal nodes sorted by timestep.
        """
        nodes = []
        for timestep in self._non_goal_node_timesteps:
            nodes.append(self.get_node(timestep))
        return nodes
 
    def _get_next_timestep(self) -> int:
        """
        Get the next available timestep.

        Returns
        -------
        int
            The next available timestep.
        """
        return len(self._node_sequence)

    def _mark_tail_leaf_node_not_appendable(self) -> None:
        """
        Mark the tail appendable leaf node as not appendable.
        """
        if self._node_sequence:
            last_node = self._node_sequence[-1]
            if isinstance(last_node, LeafEpisodicNode) and last_node.message_appendable:
                last_node.message_appendable = False

    def add_goal_node(self, goal: str, guidance: Optional[str] = None) -> int:
        """
        Add a goal node.

        If an old goal node exists, the new goal node's shifted_goal_timestep 
        will be set to the old goal node's timestep. The tail appendable leaf 
        node will be closed before adding the new goal node.

        Parameters
        ----------
        goal : str
            The goal content.
        guidance : Optional[str]
            Optional execution guidance.

        Returns
        -------
        int
            The timestep of the new goal node.
        """
        with self._lock:
            # Close the tail appendable leaf node.
            self._mark_tail_leaf_node_not_appendable()

            # Get the next timestep.
            new_timestep = self._get_next_timestep()

            # Create the new goal node.
            goal_node = GoalEpisodicNode(
                timestep=new_timestep,
                goal=goal,
                guidance=guidance,
                shifted_goal_node_timestep=self._goal_node_timestep
            )

            # Add the new goal node to the sequence and update the goal timestep.
            self._node_sequence.append(goal_node)
            self._goal_node_timestep = new_timestep

        return new_timestep

    def add_leaf_node(self, messages: List[Message]) -> int:
        """
        Add a new leaf node that is appendable to new messages.

        The tail appendable leaf node will be closed before adding the new node.

        Parameters
        ----------
        messages : List[Message]
            The original message sequence.

        Returns
        -------
        int
            The timestep of the new leaf node.
        """
        with self._lock:
            # Close the tail appendable leaf node.
            self._mark_tail_leaf_node_not_appendable()

            # Get the next timestep.
            new_timestep = self._get_next_timestep()

            # Create a new appendable leaf node.
            leaf_node = LeafEpisodicNode(
                timestep=new_timestep,
                messages=messages
            )

            # Add the new leaf node to the sequence and update the non-goal node timesteps.
            self._node_sequence.append(leaf_node)
            self._non_goal_node_timesteps.append(new_timestep)

        return new_timestep

    def add_compression_node(self, compressed_timesteps: List[int], summary: Optional[str] = None) -> int:
        """
        Add a new compression node that summarizes the given non-goal nodes.

        Before creating the compression node, close the last leaf node if it is still appendable.
        The `compressed_timesteps` list tells which nodes to summarize. Those nodes are then removed 
        from the active list, and the new compression node replaces them in the active non-goal node list.

        Parameters
        ----------
        compressed_timesteps : List[int]
            List of timesteps of the compressed nodes.
        summary : Optional[str]
            The compression summary content. If not provided, an unset asyncio.Future of summary will be created.

        Returns
        -------
        int
            The timestep of the new compression node.
        """
        with self._lock:
            # Close the tail appendable leaf node.
            self._mark_tail_leaf_node_not_appendable()

            # Get the next timestep.
            new_timestep = self._get_next_timestep()

            # Create the compression node.
            compression_node = CompressionEpisodicNode(
                timestep=new_timestep,
                compressed_timesteps=compressed_timesteps,
                summary=summary,
            )

            # Add the new compression node to the sequence and update the non-goal node timesteps.
            self._node_sequence.append(compression_node)
            self._non_goal_node_timesteps.append(new_timestep)

            # Remove the timesteps of the compressed nodes from _non_goal_node_timesteps.
            self._non_goal_node_timesteps = [
                t for t in self._non_goal_node_timesteps 
                if t not in compressed_timesteps
            ]

        return new_timestep

    def get_tail_appendable_leaf_node(self) -> Optional[LeafEpisodicNode]:
        """
        Get the tail appendable leaf node if it exists.

        Returns
        -------
        Optional[LeafEpisodicNode]
            The tail appendable leaf node, or None if not found.
        """
        if self._node_sequence:
            last_node = self._node_sequence[-1]
            if isinstance(last_node, LeafEpisodicNode) and last_node.message_appendable:
                return last_node
        return None

    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "node_sequence": [node.dump_to_dict() for node in self._node_sequence],
            "goal_node_timestep": self._goal_node_timestep,
            "non_goal_node_timesteps": self._non_goal_node_timesteps,
        }

    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self._node_sequence = []
        node_dicts = state_dict.get("node_sequence", [])

        for node_dict in node_dicts:
            node_type = NodeType(node_dict["node_type"])
            node = None

            if node_type == NodeType.GOAL:
                node = GoalEpisodicNode(timestep=0, goal="")
            elif node_type == NodeType.LEAF:
                node = LeafEpisodicNode(timestep=0)
            elif node_type == NodeType.COMPRESSED:
                node = CompressionEpisodicNode(timestep=0, compressed_timesteps=[])
            else:
                raise ValueError(f"Invalid node type: {node_type}")

            node.load_from_dict(node_dict)
            self._node_sequence.append(node)

        self._goal_node_timestep = state_dict.get("goal_node_timestep", -1)
        self._non_goal_node_timesteps = state_dict.get("non_goal_node_timesteps", [])

