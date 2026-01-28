"""
Unit tests for EpisodicNodeTree.
"""
import pytest
from threading import Thread
from bridgic.core.agentic.recent._episodic_node_tree import EpisodicNodeTree
from bridgic.core.agentic.recent._episodic_node import (
    GoalEpisodicNode,
    LeafEpisodicNode,
    CompressionEpisodicNode,
    NodeType,
)
from bridgic.core.model.types import Message, Role


class TestEpisodicNodeTreeInit:
    """Test initialization of EpisodicNodeTree."""

    def test_episodic_node_tree_init(self):
        """Test empty tree initialization, verify initial state."""
        tree = EpisodicNodeTree()
        assert tree.get_node(0) is None
        assert tree.get_goal_node() is None
        assert tree.get_non_goal_nodes() == []
        assert tree.get_tail_appendable_leaf_node() is None


class TestEpisodicNodeTreeAddNodes:
    """Test adding nodes to EpisodicNodeTree."""

    def test_add_goal_node(self):
        """Test adding goal node (including previous_goal_node_timestep link and closing appendable leaf node)."""
        tree = EpisodicNodeTree()
        
        # Add first goal node
        timestep1 = tree.add_goal_node(goal="Goal 1", guidance="Guidance 1")
        assert timestep1 == 0
        goal_node1 = tree.get_goal_node()
        assert goal_node1 is not None
        assert goal_node1.goal == "Goal 1"
        assert goal_node1.guidance == "Guidance 1"
        assert goal_node1.previous_goal_node_timestep == -1
        
        # Add second goal node (should link to first)
        timestep2 = tree.add_goal_node(goal="Goal 2", guidance="Guidance 2")
        assert timestep2 == 1
        goal_node2 = tree.get_goal_node()
        assert goal_node2 is not None
        assert goal_node2.goal == "Goal 2"
        assert goal_node2.previous_goal_node_timestep == timestep1
        
        # Add goal node after leaf node (should close appendable leaf)
        leaf_timestep = tree.add_leaf_node(messages=[])
        leaf_node = tree.get_node(leaf_timestep)
        assert isinstance(leaf_node, LeafEpisodicNode)
        assert leaf_node.message_appendable is True
        
        timestep3 = tree.add_goal_node(goal="Goal 3")
        assert leaf_node.message_appendable is False
        goal_node3 = tree.get_goal_node()
        assert goal_node3 is not None
        assert goal_node3.goal == "Goal 3"
        assert goal_node3.previous_goal_node_timestep == timestep2

    def test_add_leaf_node(self):
        """Test adding leaf node (including closing previous appendable node)."""
        tree = EpisodicNodeTree()
        
        # Add first leaf node
        timestep1 = tree.add_leaf_node(messages=[])
        assert timestep1 == 0
        leaf_node1 = tree.get_node(timestep1)
        assert isinstance(leaf_node1, LeafEpisodicNode)
        assert leaf_node1.message_appendable is True
        assert leaf_node1.messages == []
        
        # Add leaf node with messages
        timestep2 = tree.add_leaf_node(messages=[
            Message.from_text(text="Hello", role=Role.USER),
            Message.from_text(text="Hi", role=Role.AI),
        ])
        leaf_node2 = tree.get_node(timestep2)
        assert isinstance(leaf_node2, LeafEpisodicNode)
        assert leaf_node1.message_appendable is False
        assert leaf_node2.message_appendable is True
        assert len(leaf_node2.messages) == 2

    def test_add_compression_node(self):
        """Test adding compression node (including removing compressed nodes from non_goal_node_timesteps)."""
        tree = EpisodicNodeTree()
        
        # Add some leaf nodes
        timestep1 = tree.add_leaf_node(messages=[])
        timestep2 = tree.add_leaf_node(messages=[])
        timestep3 = tree.add_leaf_node(messages=[])
        
        # Verify non-goal nodes
        non_goal_nodes = tree.get_non_goal_nodes()
        assert len(non_goal_nodes) == 3
        assert [node.timestep for node in non_goal_nodes] == [timestep1, timestep2, timestep3]
        
        # Add compression node
        compressed_timesteps = [timestep2, timestep3]
        compression_timestep = tree.add_compression_node(
            compressed_timesteps=compressed_timesteps,
            summary="Compressed summary"
        )
        
        # Verify compression node
        compression_node = tree.get_node(compression_timestep)
        assert isinstance(compression_node, CompressionEpisodicNode)
        assert compression_node.compressed_node_timesteps == compressed_timesteps
        assert compression_node.summary.done()
        assert compression_node.summary.result() == "Compressed summary"
        
        # Verify compressed nodes are removed from non_goal_node_timesteps
        non_goal_nodes_after = tree.get_non_goal_nodes()
        assert len(non_goal_nodes_after) == 2  # timestep3 and compression_timestep
        assert timestep1 in [node.timestep for node in non_goal_nodes_after]
        assert timestep2 not in [node.timestep for node in non_goal_nodes_after]
        assert timestep3 not in [node.timestep for node in non_goal_nodes_after]
        assert compression_timestep in [node.timestep for node in non_goal_nodes_after]


class TestEpisodicNodeTreeQuery:
    """Test querying nodes from EpisodicNodeTree."""

    def test_get_goal_node(self):
        """Test getting goal node (existing and non-existing cases)."""
        tree = EpisodicNodeTree()
        
        # Test with no goal node
        assert tree.get_goal_node() is None
        
        # Add goal node
        timestep = tree.add_goal_node(goal="Test goal")
        goal_node = tree.get_goal_node()
        assert goal_node is not None
        assert goal_node.timestep == timestep
        assert goal_node.goal == "Test goal"
        
        # Add another goal node (should replace previous)
        timestep2 = tree.add_goal_node(goal="New goal")
        goal_node2 = tree.get_goal_node()
        assert goal_node2 is not None
        assert goal_node2.timestep == timestep2
        assert goal_node2.goal == "New goal"

    def test_get_non_goal_nodes(self):
        """Test getting non-goal nodes list."""
        tree = EpisodicNodeTree()
        
        # Test with empty tree
        assert tree.get_non_goal_nodes() == []
        
        # Add goal node (should not appear in non-goal nodes)
        goal_timestep = tree.add_goal_node(goal="Goal")
        assert len(tree.get_non_goal_nodes()) == 0
        
        # Add leaf nodes
        leaf_timestep1 = tree.add_leaf_node(messages=[])
        leaf_timestep2 = tree.add_leaf_node(messages=[])
        
        non_goal_nodes = tree.get_non_goal_nodes()
        assert len(non_goal_nodes) == 2
        assert [node.timestep for node in non_goal_nodes] == [leaf_timestep1, leaf_timestep2]
        
        # Add compression node
        compression_timestep = tree.add_compression_node(
            compressed_timesteps=[leaf_timestep1],
            summary="Summary"
        )
        non_goal_nodes_after = tree.get_non_goal_nodes()
        assert len(non_goal_nodes_after) == 2  # leaf_timestep2 and compression_timestep
        assert compression_timestep in [node.timestep for node in non_goal_nodes_after]

    def test_get_tail_appendable_leaf_node(self):
        """Test getting tail appendable leaf node."""
        tree = EpisodicNodeTree()
        
        # Test with empty tree
        assert tree.get_tail_appendable_leaf_node() is None
        
        # Add goal node (not appendable)
        tree.add_goal_node(goal="Goal")
        assert tree.get_tail_appendable_leaf_node() is None
        
        # Add leaf node (should be appendable)
        leaf_timestep_1 = tree.add_leaf_node(messages=[])
        tail_node_1 = tree.get_tail_appendable_leaf_node()
        assert tail_node_1 is not None
        assert tail_node_1.timestep == leaf_timestep_1
        assert tail_node_1.message_appendable is True
        
        # Add another leaf node (first should be closed)
        leaf_timestep_2 = tree.add_leaf_node(messages=[])
        tail_node_2 = tree.get_tail_appendable_leaf_node()
        assert tail_node_1.message_appendable is False
        assert tail_node_2 is not None
        assert tail_node_2.timestep == leaf_timestep_2
        assert tail_node_2.message_appendable is True


class TestEpisodicNodeTreeSerialization:
    """Test serialization of EpisodicNodeTree."""

    def test_serialization_roundtrip(self):
        """Test serialization-deserialization roundtrip, verify data consistency."""
        tree = EpisodicNodeTree()
        
        # Build a complex tree
        goal_timestep1 = tree.add_goal_node(goal="Goal 1", guidance="Guidance 1")
        leaf_timestep1 = tree.add_leaf_node(messages=[
            Message.from_text(text="Hello", role=Role.USER),
        ])
        leaf_timestep2 = tree.add_leaf_node(messages=[
            Message.from_text(text="Hi", role=Role.AI),
        ])
        compression_timestep = tree.add_compression_node(
            compressed_timesteps=[leaf_timestep1],
            summary="Compressed"
        )
        goal_timestep2 = tree.add_goal_node(goal="Goal 2")
        
        # Serialize
        state_dict = tree.dump_to_dict()
        
        # Deserialize
        new_tree = EpisodicNodeTree.__new__(EpisodicNodeTree)
        new_tree.load_from_dict(state_dict)
        
        # Verify structure
        assert new_tree.get_goal_node() is not None
        assert new_tree.get_goal_node().goal == "Goal 2"
        assert new_tree.get_goal_node().previous_goal_node_timestep == goal_timestep1
        
        non_goal_nodes = new_tree.get_non_goal_nodes()
        assert len(non_goal_nodes) == 2  # leaf_timestep2 and compression_timestep
        
        # Verify nodes
        leaf_node2 = new_tree.get_node(leaf_timestep2)
        assert isinstance(leaf_node2, LeafEpisodicNode)
        assert len(leaf_node2.messages) == 1
        
        compression_node = new_tree.get_node(compression_timestep)
        assert isinstance(compression_node, CompressionEpisodicNode)
        assert compression_node.compressed_node_timesteps == [leaf_timestep1]
        assert compression_node.summary.done()
        assert compression_node.summary.result() == "Compressed"
