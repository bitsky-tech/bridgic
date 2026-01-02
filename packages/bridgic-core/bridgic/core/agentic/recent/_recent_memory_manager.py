import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from typing_extensions import override

from bridgic.core.model import BaseLlm
from bridgic.core.model.types import Message, Role, TextBlock, ToolCallBlock, ToolResultBlock
from bridgic.core.types._serialization import Serializable
from bridgic.core.agentic.recent._episodic_node import (
    BaseEpisodicNode,
    GoalEpisodicNode,
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

    goal_guidance: str
    """Goal guidance."""

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

    @property
    def memory_config(self) -> ReCentMemoryConfig:
        """Get the memory configuration."""
        return self._memory_config

    def get_specified_memory_node(self, timestep: int) -> BaseEpisodicNode:
        """
        Get the specified memory node by timestep.

        Parameters
        ----------
        timestep : int
            The timestep of the memory node.

        Returns
        -------
        BaseEpisodicNode
            The specified memory node or None if not found.
        """
        return self._episodic_node_tree.get_node(timestep)

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

    def _create_compression_node(self) -> Tuple[int, CompressionEpisodicNode, List[BaseEpisodicNode], Optional[BaseEpisodicNode]]:
        """
        Helper method to create a compression node placeholder (Step 1).
        
        This method acquires the lock, determines nodes to compress, creates a placeholder
        compression node, and releases the lock. This keeps the lock holding time minimal.

        Returns
        -------
        Tuple[int, CompressionEpisodicNode, List[BaseEpisodicNode], Optional[BaseEpisodicNode]]
            A tuple containing:
            - The timestep of the new compression node
            - The compression node instance
            - List of non-goal nodes to be compressed
            - The goal node (if exists)

        Raises
        ------
        ValueError
            If there are no nodes to compress.
        """
        with self._episodic_node_tree._lock:
            goal_node = self._episodic_node_tree.get_goal_node()
            non_goal_nodes = self._episodic_node_tree.get_non_goal_nodes()

            if not non_goal_nodes:
                raise ValueError("There is no node to could be compressed")

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

        # Return the necessary context to do the summarization.
        return compression_timestep, compression_node, non_goal_nodes, goal_node

    def create_compression(self) -> int:
        """
        Create a new compression node at the tail for compressing memory (synchronous version).

        If the last node is an appendable leaf (messages can be added), it will be closed first.
        The nodes pointed to by `_non_goal_node_timesteps` will be compressed and their timesteps 
        will be removed from `_non_goal_node_timesteps`, but recorded in the new compression node.

        This method is thread-safe and handles cross-thread dependencies: if a compression node depends on
        other compression nodes whose summaries are still being generated, it will wait for them using
        concurrent.futures.Future.

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
        compression_timestep, compression_node, non_goal_nodes, goal_node = self._create_compression_node()

        # Step 2: Release lock, then prompt the LLM to generate the summary.
        try:
            compression_messages = []

            # Construct the prompt messages to compress the conversations.
            if goal_node:
                system_message = self._memory_config.system_prompt_template.format_message(
                    role=Role.SYSTEM,
                    goal=goal_node.goal if goal_node.goal else "",
                    guidance=goal_node.guidance if goal_node.guidance else "",
                )
                compression_messages.append(system_message)

            # Collect messages from nodes that are to be compressed (compressed nodes).
            for node in non_goal_nodes:
                if isinstance(node, LeafEpisodicNode):
                    # For leaf nodes, use their original messages.
                    compression_messages.extend(node.messages)
                elif isinstance(node, CompressionEpisodicNode):
                    # For compression nodes (that compress other nodes), wait for their summary content.
                    summary_text = node.summary.result()
                    compression_messages.append(
                        Message.from_text(
                            text=f"[Compressed Memory] {summary_text}",
                            role=Role.AI,
                        )
                    )

            # Append the instruction message.
            instruction_message = self._memory_config.instruction_prompt_template.format_message(role=Role.USER)
            compression_messages.append(instruction_message)

            # Step 3: Call the LLM to generate summary (synchronous version).
            response = self._memory_config.llm.chat(messages=compression_messages)
            summary = response.message.content

            # Set the result of the summary future.
            compression_node.summary.set_result(summary)
        except Exception as e:
            # If summary generation fails, set exception description as result on the summary future.
            compression_node.summary.set_result(str(e))

        return compression_timestep

    async def acreate_compression(self) -> int:
        """
        Create a new compression node at the tail for compressing memory asynchronously.

        If the last node is an appendable leaf (messages can be added), it will be closed first.
        The nodes pointed to by `_non_goal_node_timesteps` will be compressed and their timesteps 
        will be removed from `_non_goal_node_timesteps`, but recorded in the new compression node.

        This method is thread-safe and handles cross-thread/event-loop dependencies: if a compression node depends on
        other compression nodes whose summaries are still being generated, it will non-blockingly await them using
        asyncio.wrap_future() to convert concurrent.futures.Future to asyncio.Future.

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
        compression_timestep, compression_node, non_goal_nodes, goal_node = self._create_compression_node()

        # Step 2: Release lock, then prompt the LLM to generate the summary.
        try:
            compression_messages = []

            # Construct the prompt messages to compress the conversations.
            if goal_node:
                system_message = self._memory_config.system_prompt_template.format_message(
                    role=Role.SYSTEM,
                    goal=goal_node.goal if goal_node.goal else "",
                    guidance=goal_node.guidance if goal_node.guidance else "",
                )
                compression_messages.append(system_message)

            # Collect messages from nodes that are to be compressed (compressed nodes).
            for node in non_goal_nodes:
                if isinstance(node, LeafEpisodicNode):
                    # For leaf nodes, use their original messages.
                    compression_messages.extend(node.messages)
                elif isinstance(node, CompressionEpisodicNode):
                    # For compression nodes (that compress other nodes), wait for their summary content.
                    # Convert concurrent.futures.Future to asyncio.Future for non-blocking await
                    asyncio_future = asyncio.wrap_future(node.summary)
                    summary_text = await asyncio_future
                    compression_messages.append(
                        Message.from_text(
                            text=f"[Compressed Memory] {summary_text}",
                            role=Role.AI,
                        )
                    )

            # Append the instruction message.
            instruction_message = self._memory_config.instruction_prompt_template.format_message(role=Role.USER)
            compression_messages.append(instruction_message)

            # Step 3: Call the LLM to generate summary (asynchronous version).
            response = await self._memory_config.llm.achat(messages=compression_messages)
            summary = response.message.content

            # Set the result of the summary future.
            compression_node.summary.set_result(summary)
        except Exception as e:
            # If summary generation fails, set exception description as result on the summary future.
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

    def should_trigger_compression(self) -> bool:
        """
        Determine whether memory compression should be triggered based on the compression configuration.

        This method checks two conditions:
        1. Node count: whether the total number of nodes (non-goal nodes + goal node if exists) exceeds the threshold
        2. Token count: whether the token count (calculated using `token_count_callback` of configuration) exceeds the threshold

        The text size calculation includes:
        - Goal node:
          - GoalEpisodicNode: the text content of goal and guidance
        - Non-goal nodes:
          - LeafEpisodicNode: the text content of all messages
          - CompressionEpisodicNode: the text content of the summary (if available)

        Returns
        -------
        bool
            True if compression should be triggered, False otherwise.
        """
        goal_node = self._episodic_node_tree.get_goal_node()
        non_goal_nodes = self._episodic_node_tree.get_non_goal_nodes()

        node_list = []
        if goal_node is not None:
            node_list.append(goal_node)
        if non_goal_nodes:
            node_list.extend(non_goal_nodes)

        # Check node count threshold.
        if len(node_list) >= self._memory_config.max_node_size:
            return True

        # Collect text parts.
        text_parts = []
        for node in node_list:
            if isinstance(node, GoalEpisodicNode):
                if node.goal:
                    text_parts.append(node.goal)
                if node.guidance:
                    text_parts.append(node.guidance)
            elif isinstance(node, LeafEpisodicNode):
                for message in node.messages:
                    for block in message.blocks:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolCallBlock):
                            text_parts.append("{"+ f"\"id\": \"{block.id}\", \"name\": \"{block.name}\", \"arguments\": {block.arguments}" + "}")
                        elif isinstance(block, ToolResultBlock):
                            text_parts.append(block.content)
                        else:
                            raise ValueError(f"Not supported block type: {type(block)}")
            elif isinstance(node, CompressionEpisodicNode):
                text_parts.append(node.summary.result())

        # Calculate token count using callback function.
        total_text = "\n\n".join(text_parts)
        estimated_token_count = self._memory_config.token_count_callback(total_text)

        # Check token count threshold.
        if estimated_token_count >= self._memory_config.max_token_size:
            return True

        return False

    def build_context(self) -> ReCentContext:
        """
        Build context, returning goal information and message list.

        The context is built by:
        1. Getting goal node information
        2. Getting directly accessible nodes based on _non_goal_node_timesteps
          - For CompressionEpisodicNode, waiting for its summary future
          - For LeafEpisodicNode, using original messages
        3. Organizing messages in timestep order

        Returns
        -------
        ContextDict
            Dictionary containing goal content, goal timestep, and memory messages.
        """
        # 1. Get goal node.
        goal_node = self._episodic_node_tree.get_goal_node()
        goal_content = goal_node.goal if goal_node else ""
        goal_guidance = goal_node.guidance if goal_node else ""
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
                    text=f"[Stage Summary] {node.summary.result()}",
                    role=Role.AI,
                )
                memory_messages.append(msg)

        return {
            "goal_content": goal_content,
            "goal_guidance": goal_guidance,
            "goal_timestep": goal_timestep,
            "memory_messages": memory_messages,
        }

    async def abuild_context(self) -> ReCentContext:
        """
        Build context asynchronously, returning goal information and message list.

        The context is built by:
        1. Getting goal node information
        2. Getting directly accessible nodes based on _non_goal_node_timesteps
          - For CompressionEpisodicNode, waiting for its summary future
          - For LeafEpisodicNode, using original messages
        3. Organizing messages in timestep order

        Returns
        -------
        ContextDict
            Dictionary containing goal content, goal timestep, and memory messages.
        """
        # 1. Get goal node.
        goal_node = self._episodic_node_tree.get_goal_node()
        goal_content = goal_node.goal if goal_node else ""
        goal_guidance = goal_node.guidance if goal_node else ""
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
                # Compression node: wait for its summary future (non-blocking).
                asyncio_future = asyncio.wrap_future(node.summary)
                summary_text = await asyncio_future
                msg = Message.from_text(
                    text=f"[Stage Summary] {summary_text}",
                    role=Role.AI,
                )
                memory_messages.append(msg)

        return {
            "goal_content": goal_content,
            "goal_guidance": goal_guidance,
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
