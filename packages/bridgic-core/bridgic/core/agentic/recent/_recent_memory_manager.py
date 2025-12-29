from typing import List, Dict, Any, Optional, TypedDict
from typing_extensions import override

from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Role
from bridgic.core.types._serialization import Serializable
from bridgic.core.agentic.recent._episodic_node import (
    BaseEpisodicNode,
    LeafEpisodicNode,
    CompressionEpisodicNode,
)
from bridgic.core.agentic.recent._episodic_node_tree import EpisodicNodeTree
from bridgic.core.agentic.recent._recent_memory_config import ReCentMemoryConfig


class ReCentContext(TypedDict):
    """
    Context dictionary returned by `ReCentMemoryManager.build_context()` method.
    """

    goal_content: str
    """Goal content."""

    goal_timestep: Optional[int]
    """Goal node timestep, or -1 if no goal node exists."""

    memory_messages: List[Message]
    """List of memory messages."""


class ReCentMemoryManager(Serializable):
    """
    Memory manager that uses Recursive Compressed Episodic Node Tree Algorithm to manage memory.

    This class manages episodic memory nodes through an `EpisodicNodeTree` instance and provides 
    high-level interfaces for creating goals, leaves, compressions, and building context.
    """

    _episodic_node_tree: EpisodicNodeTree
    """The episodic node tree instance."""

    _memory_config: ReCentMemoryConfig
    """The memory configuration."""

    def __init__(self, compression_config: ReCentMemoryConfig):
        self._episodic_node_tree = EpisodicNodeTree()
        self._memory_config = compression_config

    def create_goal(self, goal: str, guidance: Optional[str] = None) -> int:
        """
        Create a new goal node.

        If an old goal exists, it will be associated automatically 
        (handled by EpisodicNodeTree).

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
        return self._episodic_node_tree.add_goal_node(goal, guidance)

    def create_leaf(self) -> int:
        """
        Create a new leaf node at the tail for appending messages.
        
        If the last node is an appendable leaf (messages can be added), it will be closed first.

        Returns
        -------
        int
            The timestep of the new leaf node.
        """
        return self._episodic_node_tree.add_leaf_node(messages=[])

    async def create_compression(self) -> int:
        """
        Create a new compression node at the tail for compressing memory.

        If the last node is an appendable leaf (messages can be added), it will be closed first.
        The nodes pointed to by `_non_goal_node_timesteps` will be compressed and their timesteps 
        will be removed from `_non_goal_node_timesteps`, but recorded in the new compression node.

        This method is thread-safe and handles async dependencies: if a compression node depends on
        other compression nodes whose summaries are still being generated, it will await them.

        Returns
        -------
        int
            The timestep of the new compression node.

        Raises
        ------
        ValueError
            If there are no nodes to compress.
        """
        # Step 1: Acquire lock to determine nodes to compress and create placeholder node
        with self._episodic_node_tree._lock:
            goal_node = self._episodic_node_tree.get_goal_node()
            non_goal_nodes = self._episodic_node_tree.get_non_goal_nodes()

            if not non_goal_nodes:
                raise ValueError("No nodes to compress")

            # Get the timesteps of nodes that will be compressed (compressed nodes).
            compressed_timesteps = [node.timestep for node in non_goal_nodes]

            # Create compression node without summary content (summary content will be set later).
            compression_timestep = self._episodic_node_tree.add_compression_node(
                compressed_timesteps=compressed_timesteps,
                summary=None
            )

            # Get reference to the created compression node for later summary update
            compression_node = self._episodic_node_tree.get_node(compression_timestep)
            assert isinstance(compression_node, CompressionEpisodicNode)

        # Step 2: Release lock, then prompt the LLM to generate the summary.
        try:
            compression_messages = []

            # Construct the prompt messages to compress the conversations.
            if goal_node:
                prompt_text = (
                    "You are a helpful assistant. Your responsibility is to summarize the messages up to the "
                    "present and organize useful information to better make the next step plan and complete "
                    "the task. When the task goal and guidance exist, it is better to refer to them before "
                    "making your summary."
                )
                if goal_node.goal:
                    prompt_text += f"\n\nTask Goal: {goal_node.goal}"
                if goal_node.guidance:
                    prompt_text += f"\n\nTask Guidance: {goal_node.guidance}"

                system_message = Message.from_text(
                    text=prompt_text,
                    role=Role.SYSTEM,
                )
                compression_messages.append(system_message)

            # Collect messages from nodes that are to be compressed (compressed nodes).
            for node in non_goal_nodes:
                if isinstance(node, LeafEpisodicNode):
                    # For leaf nodes, use their original messages.
                    compression_messages.extend(node.messages)
                elif isinstance(node, CompressionEpisodicNode):
                    # For compression nodes (that compress other nodes), await their summary content.
                    summary_text = await node.summary
                    compression_messages.append(
                        Message.from_text(
                            text=f"[Compressed Memory] {summary_text}",
                            role=Role.AI,
                        )
                    )

            # Append the instruction message.
            instruction_message = Message.from_text(
                text=(
                    "Please summarize all the conversation so far, highlighting the most important and helpful "
                    "points, facts, details, and conclusions. Organize the key content to support further "
                    "understanding and progress on the task. You may structure the summary logically and "
                    "emphasize the crucial information."
                ),
                role=Role.USER,
            )
            compression_messages.append(instruction_message)

            # Call the LLM to generate summary.
            response = await self._memory_config.llm.achat(messages=compression_messages)
            summary = response.message.content

            # Set the result of the summary future.
            compression_node.summary.set_result(summary)
        except Exception as e:
            # If summary generation fails, set exception on the summary future.
            compression_node.summary.set_result(str(e))

        return compression_timestep

    def push_messages(self, messages: List[Message]) -> int:
        """
        Append messages to the latest appendable leaf node, or create one if needed.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to append.

        Returns
        -------
        int
            The timestep of the leaf node.
        """
        # Try to get the tail appendable leaf node
        tail_leaf_node = self._episodic_node_tree.get_tail_appendable_leaf_node()
        
        if tail_leaf_node is not None:
            # If an appendable leaf node exists, append messages
            tail_leaf_node.messages.extend(messages)
            return tail_leaf_node.timestep
        else:
            # If not, create a new appendable leaf node
            return self._episodic_node_tree.add_leaf_node(messages)

    async def build_context(self) -> ReCentContext:
        """
        Build context, returning goal information and message list.

        The context is built by:
        1. Getting goal node information
        2. Getting directly accessible nodes based on _non_goal_node_timesteps
        3. For CompressionEpisodicNode, awaiting its summary Future
        4. For LeafEpisodicNode, using original messages
        5. Organizing messages in timestep order

        Returns
        -------
        ContextDict
            Dictionary containing goal content, goal timestep, and memory messages.
        """
        # 1. Get goal node.
        goal_node = self._episodic_node_tree.get_goal_node()
        goal_content = goal_node.goal if goal_node else ""
        goal_timestep = goal_node.timestep if goal_node else -1

        # 2. Get directly accessible non-goal nodes (sorted by timestep).
        non_goal_nodes = self._episodic_node_tree.get_non_goal_nodes()

        # 3. Build message list.
        memory_messages = []
        for node in non_goal_nodes:
            if isinstance(node, LeafEpisodicNode):
                # Leaf node: use original messages.
                memory_messages.extend(node.messages)
            elif isinstance(node, CompressionEpisodicNode):
                # Compression node: await its summary future.
                msg = Message.from_text(
                    text=f"[Stage Summary] {await node.summary}",
                    role=Role.AI,
                )
                memory_messages.append(msg)

        return {
            "goal_content": goal_content,
            "goal_timestep": goal_timestep,
            "memory_messages": memory_messages,
        }

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "episodic_node_tree": self._episodic_node_tree,
            "memory_config": self._memory_config,
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self._episodic_node_tree = state_dict["episodic_node_tree"]
        self._memory_config = state_dict["memory_config"]
