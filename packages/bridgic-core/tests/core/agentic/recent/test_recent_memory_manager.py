"""
Unit tests for ReCentMemoryManager.
"""
import pytest
import asyncio
import time
from bridgic.core.agentic.recent._recent_memory_manager import ReCentMemoryManager, ReCentContext
from bridgic.core.agentic.recent._recent_memory_config import ReCentMemoryConfig
from bridgic.core.agentic.recent._episodic_node_tree import EpisodicNodeTree
from bridgic.core.agentic.recent._episodic_node import (
    GoalEpisodicNode,
    LeafEpisodicNode,
    CompressionEpisodicNode,
)
from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Role, Response
from bridgic.core.utils._console import printer


class MockLlm(BaseLlm):
    """Mock LLM for testing async compression and context building."""
    
    def __init__(self, response_text: str = "Mock summary"):
        self.response_text = response_text
    
    async def achat(self, messages, **kwargs) -> Response:
        """Return mock response."""
        await asyncio.sleep(0.25)
        return Response(
            message=Message.from_text(text=self.response_text, role=Role.AI),
            raw=None
        )
    
    def chat(self, messages, **kwargs) -> Response:
        """Synchronous chat (not used in tests)."""
        time.sleep(0.25)
        return Response(
            message=Message.from_text(text=self.response_text, role=Role.AI),
            raw=None
        )
    
    def stream(self, messages, **kwargs):
        """Stream (not used in tests)."""
        raise NotImplementedError
    
    async def astream(self, messages, **kwargs):
        """Async stream (not used in tests)."""
        raise NotImplementedError
    
    def dump_to_dict(self):
        return {
            "response_text": self.response_text,
        }
    
    def load_from_dict(self, state_dict):
        self.response_text = state_dict["response_text"]


@pytest.fixture
def mock_llm():
    """Provide Mock LLM instance."""
    return MockLlm(response_text="Mock compression summary")


@pytest.fixture
def memory_config(mock_llm):
    """Provide ReCentMemoryConfig instance."""
    return ReCentMemoryConfig(
        llm=mock_llm,
        max_node_size=10,
        max_token_size=1000,
    )


@pytest.fixture
def sample_messages():
    """Provide sample Message list."""
    return [
        Message.from_text(text="Hello", role=Role.USER),
        Message.from_text(text="Hi there", role=Role.AI),
        Message.from_text(text="How are you?", role=Role.USER),
    ]


class TestReCentMemoryManagerInit:
    """Test initialization of ReCentMemoryManager."""

    def test_recent_memory_manager_init(self, memory_config):
        """Test initialization, verify EpisodicNodeTree and ReCentMemoryConfig setup."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        assert isinstance(manager._episodic_node_tree, EpisodicNodeTree)
        assert manager._memory_config == memory_config
        assert manager._episodic_node_tree.get_goal_node() is None
        assert len(manager._episodic_node_tree.get_non_goal_nodes()) == 0


class TestReCentMemoryManagerPushMessages:
    """Test message pushing in ReCentMemoryManager."""

    def test_push_messages(self, memory_config, sample_messages):
        """Test pushing messages (including appending to existing node and creating new node)."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Push messages (should create new leaf node)
        timestep1 = manager.push_messages(sample_messages[:2])
        leaf_node1 = manager._episodic_node_tree.get_node(timestep1)
        assert isinstance(leaf_node1, LeafEpisodicNode)
        assert len(leaf_node1.messages) == 2
        assert leaf_node1.message_appendable is True
        
        # Push more messages (should append to existing node)
        timestep2 = manager.push_messages(sample_messages[-1:])
        assert timestep2 == timestep1  # Same node
        assert len(leaf_node1.messages) == 3
        
        # Close the node and push again (should create new node)
        manager.create_leaf()  # This closes the previous leaf
        timestep3 = manager.push_messages(sample_messages[:1])
        assert timestep3 != timestep1


class TestReCentMemoryManagerCompression:
    """Test compression node creation in ReCentMemoryManager."""

    def test_create_compression(self, memory_config, sample_messages):
        """Test creating compression node synchronously (using mock LLM, including with/without goal node)."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Create some leaf nodes
        manager.push_messages(sample_messages)
        
        # Create compression (without goal node)
        compression_timestep = manager.create_compression()
        compression_node = manager._episodic_node_tree.get_node(compression_timestep)
        assert isinstance(compression_node, CompressionEpisodicNode)
        assert compression_node.summary.done()
        assert compression_node.summary.result() == "Mock compression summary"

    @pytest.mark.asyncio
    async def test_acreate_compression(self, memory_config, sample_messages):
        """Test creating compression node asynchronously (using mock LLM, including with/without goal node)."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Create some leaf nodes
        manager.push_messages(sample_messages)
        
        # Create compression (without goal node)
        compression_timestep = await manager.acreate_compression()
        compression_node = manager._episodic_node_tree.get_node(compression_timestep)
        assert isinstance(compression_node, CompressionEpisodicNode)
        assert compression_node.summary.done()
        assert compression_node.summary.result() == "Mock compression summary"

    def test_create_compression_raises_when_no_nodes(self, memory_config):
        """Test that creating compression synchronously raises ValueError when there are no nodes to compress."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Try to compress with no nodes
        with pytest.raises(ValueError, match="There is no node to could be compressed"):
            manager.create_compression()
        
        # Add only goal node (no non-goal nodes)
        manager.create_goal(goal="Goal")
        with pytest.raises(ValueError, match="There is no node to could be compressed"):
            manager.create_compression()

    @pytest.mark.asyncio
    async def test_acreate_compression_raises_when_no_nodes(self, memory_config):
        """Test that creating compression asynchronously raises ValueError when there are no nodes to compress."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Try to compress with no nodes
        with pytest.raises(ValueError, match="There is no node to could be compressed"):
            await manager.acreate_compression()
        
        # Add only goal node (no non-goal nodes)
        manager.create_goal(goal="Goal")
        with pytest.raises(ValueError, match="There is no node to could be compressed"):
            await manager.acreate_compression()


class TestReCentMemoryManagerBuildContext:
    """Test context building in ReCentMemoryManager."""

    def test_build_context(self, memory_config, sample_messages):
        """Test building context synchronously (including goal node, leaf nodes, and compression nodes)."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Build context with no nodes
        context = manager.build_context()
        assert isinstance(context, dict)
        assert context["goal_content"] == ""
        assert context["goal_timestep"] == -1
        assert context["memory_messages"] == []
        
        # Add goal and leaf nodes
        manager.create_goal(goal="Test goal", guidance="Guidance")
        manager.push_messages(sample_messages[:2])
        
        context = manager.build_context()
        assert context["goal_content"] == "Test goal"
        assert context["goal_timestep"] == 0
        assert len(context["memory_messages"]) == 2
        assert context["memory_messages"][0].content == "Hello"
        
        # Add compression node
        manager.create_compression()
        context = manager.build_context()
        assert len(context["memory_messages"]) == 1
        assert "[Stage Summary]" in context["memory_messages"][0].content

    @pytest.mark.asyncio
    async def test_abuild_context(self, memory_config, sample_messages):
        """Test building context asynchronously (including goal node, leaf nodes, and compression nodes)."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Build context with no nodes
        context = await manager.abuild_context()
        assert isinstance(context, dict)
        assert context["goal_content"] == ""
        assert context["goal_timestep"] == -1
        assert context["memory_messages"] == []
        
        # Add goal and leaf nodes
        manager.create_goal(goal="Test goal", guidance="Guidance")
        manager.push_messages(sample_messages[:2])
        
        context = await manager.abuild_context()
        assert context["goal_content"] == "Test goal"
        assert context["goal_timestep"] == 0
        assert len(context["memory_messages"]) == 2
        assert context["memory_messages"][0].content == "Hello"
        
        # Add compression node
        await manager.acreate_compression()
        context = await manager.abuild_context()
        assert len(context["memory_messages"]) == 1
        assert "[Stage Summary]" in context["memory_messages"][0].content


class TestReCentMemoryConfigPromptRendering:
    """Test prompt template rendering in ReCentMemoryConfig."""

    def test_default_system_prompt_render(self, mock_llm):
        """Test default system prompt template rendering with various goal/guidance combinations."""
        config = ReCentMemoryConfig(
            llm=mock_llm,
            max_node_size=10,
            max_token_size=1000,
        )
        
        # Test rendering with both goal and guidance
        message = config.system_template.format_message(
            role=Role.SYSTEM,
            goal="Complete the task",
            guidance="Follow the steps carefully",
        )
        assert message.role == Role.SYSTEM
        assert len(message.content) > 0
        # printer.print("\n" + message.content)

    def test_default_instruction_prompt_render(self, mock_llm):
        """Test default instruction prompt template rendering."""
        config = ReCentMemoryConfig(
            llm=mock_llm,
            max_node_size=10,
            max_token_size=1000,
        )
        
        # Test rendering instruction prompt
        message = config.instruction_template.format_message(role=Role.USER)
        assert message.role == Role.USER
        assert len(message.content) > 0
        # printer.print("\n" + message.content)


class TestReCentMemoryManagerSerialization:
    """Test serialization of ReCentMemoryManager."""

    def test_serialization_roundtrip(self, memory_config, sample_messages):
        """Test serialization-deserialization roundtrip."""
        manager = ReCentMemoryManager(compression_config=memory_config)
        
        # Build some state
        manager.create_goal(goal="Test goal", guidance="Guidance")
        manager.push_messages(sample_messages[:2])
        manager.create_leaf()
        
        # Serialize
        state_dict = manager.dump_to_dict()
        
        # Deserialize
        new_manager = ReCentMemoryManager.__new__(ReCentMemoryManager)
        new_manager.load_from_dict(state_dict)
        
        # Verify state
        goal_node = new_manager._episodic_node_tree.get_goal_node()
        assert goal_node is not None
        assert goal_node.goal == "Test goal"
        assert goal_node.guidance == "Guidance"
        
        non_goal_nodes = new_manager._episodic_node_tree.get_non_goal_nodes()
        assert len(non_goal_nodes) == 2  # Two leaf nodes
