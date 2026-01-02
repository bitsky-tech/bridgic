"""
The module provides core components for the ReCENT Algorithm.

**ReCENT Algorithm** (Recursive Compressed Episodic Node Tree Algorithm) is an algorithm designed to 
address issues such as context explosion and goal drift, by employing a recursive memory compression 
mechanism.

This module provides an agentic automa and its corresponding memory configuration:
- `ReCentAutoma`: The main automaton that implements the ReCENT algorithm.
- `ReCentMemoryConfig`: Configuration for ReCENT memory management.

The core data structures are:
- `EpisodicNodeTree`: Tree of episodic nodes which is the core data structure of ReCENT.
- `BaseEpisodicNode`: Base class for all episodic nodes. It is inherited by:
    + `GoalEpisodicNode`: A goal node that represents the goal of the agent.
    + `LeafEpisodicNode`: A leaf node that represents a sequence of messages.
    + `CompressionEpisodicNode`: A compression node that summarizes a sequence of episodic nodes.
"""

from bridgic.core.agentic.recent._recent_automa import ReCentAutoma
from bridgic.core.agentic.recent._recent_memory_config import ReCentMemoryConfig
from bridgic.core.agentic.recent._episodic_node_tree import EpisodicNodeTree
from bridgic.core.agentic.recent._episodic_node import (
    BaseEpisodicNode,
    GoalEpisodicNode,
    LeafEpisodicNode,
    CompressionEpisodicNode,
    NodeType,
)

__all__ = [
    "ReCentMemoryConfig",
    "ReCentAutoma",
    "EpisodicNodeTree",
    "BaseEpisodicNode",
    "GoalEpisodicNode",
    "LeafEpisodicNode",
    "CompressionEpisodicNode",
    "NodeType",
]

