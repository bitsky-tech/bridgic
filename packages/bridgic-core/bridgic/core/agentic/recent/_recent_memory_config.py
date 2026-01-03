from typing import Dict, Any, Optional, Callable
from typing_extensions import override
from bridgic.core.model import BaseLlm
from bridgic.core.types._serialization import Serializable
from bridgic.core.prompt import EjinjaPromptTemplate
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func


DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Your responsibility is to summarize the messages up to the "
    "present and organize useful information to better make the next step plan and complete "
    "the task. When the task goal and guidance exist, it is better to refer to them before "
    "making your summary."
    "{%- if goal %}\n\nTask Goal: {{ goal }}{% endif %}"
    "{%- if guidance %}\n\nTask Guidance: {{ guidance }}{% endif %}"
)

DEFAULT_INSTRUCTION_PROMPT_TEMPLATE = (
    "Please summarize all the conversation so far, highlighting the most important and helpful "
    "points, facts, details, and conclusions. Organize the key content to support further "
    "understanding and progress on the task. You may structure the summary logically and "
    "emphasize the crucial information."
)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text using a simple approximation rule.
    
    This function uses a character-based approximation: approximately 1 token ≈ 4 characters
    for English text. This is a common heuristic used when exact tokenization is not available.
    
    Parameters
    ----------
    text : str
        The text content to estimate token count for.
    
    Returns
    -------
    int
        The estimated token count.
    """
    return len(text) // 4


class ReCentMemoryConfig(Serializable):
    """
    This configuration class defines the memory management strategy that will compress
    the conversation history when certain conditions are met.

    Attributes
    ----------
    llm : BaseLlm
        The LLM instance used for memory compression operations.
    max_node_size : int
        Maximum number of memory nodes before triggering compression.
        Defaults to 10.
    max_token_size : int
        Maximum number of tokens before triggering compression.
        Defaults to 8192 (1024 * 8).
    system_prompt_template : str
        Jinja2 prompt template for the system prompt used in memory compression, which accepts 
        parameters: `goal` and `guidance`.
    instruction_prompt_template : str
        Jinja2 prompt template for the instruction prompt used in memory compression.
    token_count_callback : Optional[Callable[[str], int]]
        Optional callback function to calculate token count from text.
        If None, defaults to `estimate_token_count` which uses a simple approximation
        (character_count / 4). The callback should accept a text string and return the token count.
    """

    llm: BaseLlm
    """The LLM used for memory compression."""

    max_node_size: int
    """Threshold for the number of memory nodes to trigger memory compression."""

    max_token_size: int
    """Threshold for the number of tokens to trigger memory compression."""

    system_prompt_template: EjinjaPromptTemplate
    """Template for system prompt used in memory compression."""

    instruction_prompt_template: EjinjaPromptTemplate
    """Instruction prompt template used in memory compression."""

    token_count_callback: Callable[[str], int]
    """Callback function to calculate token count from text. Defaults to estimate_token_count."""

    def __init__(
        self,
        llm: BaseLlm,
        max_node_size: int = 20,
        max_token_size: int = 1024 * 16,
        system_prompt_template: Optional[str] = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        instruction_prompt_template: Optional[str] = DEFAULT_INSTRUCTION_PROMPT_TEMPLATE,
        token_count_callback: Optional[Callable[[str], int]] = estimate_token_count,
    ):
        self.llm = llm
        self.max_node_size = max_node_size
        self.max_token_size = max_token_size
        self.token_count_callback = token_count_callback if token_count_callback is not None else estimate_token_count
        
        # Convert string templates to EjinjaPromptTemplate instances
        # If None is explicitly passed, use default templates
        if system_prompt_template is None:
            system_prompt_template = DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.system_prompt_template = EjinjaPromptTemplate(system_prompt_template)
        
        if instruction_prompt_template is None:
            instruction_prompt_template = DEFAULT_INSTRUCTION_PROMPT_TEMPLATE
        self.instruction_prompt_template = EjinjaPromptTemplate(instruction_prompt_template)

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["llm"] = self.llm
        state_dict["max_node_size"] = self.max_node_size
        state_dict["max_token_size"] = self.max_token_size
        state_dict["token_count_callback"] = self.token_count_callback.__module__ + "." + self.token_count_callback.__qualname__
        state_dict["system_prompt_template"] = self.system_prompt_template
        state_dict["instruction_prompt_template"] = self.instruction_prompt_template
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.llm = state_dict["llm"]
        self.max_node_size = state_dict["max_node_size"]
        self.max_token_size = state_dict["max_token_size"]
        self.token_count_callback = load_qualified_class_or_func(state_dict["token_count_callback"])
        self.system_prompt_template = state_dict["system_prompt_template"]
        self.instruction_prompt_template = state_dict["instruction_prompt_template"]