"""
The module provides core memory data structures for the ReCENT Algorithm.

**ReCENT Algorithm** (Recursive Compressed Episodic Node Tree Algorithm) is an algorithm designed to 
address issues such as context explosion and goal drift, by employing a recursive memory compression 
mechanism. In this algorithm, each episodic node will serve as a container of memory and could be 
tightly organized together to form a more efficient and reliable memory for the higher agentic system.

There are three types of episodic nodes:
- GoalEpisodicNode: A goal node that represents the goal of the agent.
- LeafEpisodicNode: A leaf node that represents a sequence of messages.
- CompressedEpisodicNode: A compressed node that represents a compressed memory unit.
"""

from ._episodic_node import (
    BaseEpisodicNode,
    GoalEpisodicNode,
    LeafEpisodicNode,
    CompressedEpisodicNode,
    NodeType,
)

__all__ = [
    "BaseEpisodicNode",
    "GoalEpisodicNode",
    "LeafEpisodicNode",
    "CompressedEpisodicNode",
    "NodeType",
]

