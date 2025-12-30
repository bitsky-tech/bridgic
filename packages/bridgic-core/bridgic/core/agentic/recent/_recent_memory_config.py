from typing import Dict, Any, Optional
from typing_extensions import override
from bridgic.core.model import BaseLlm
from bridgic.core.types._serialization import Serializable
from bridgic.core.prompt import EjinjaPromptTemplate


# Default prompt templates
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


class ReCentMemoryConfig(Serializable):
    """Configuration for ReCent memory management."""

    llm: BaseLlm
    """LLM model used for memory compression."""

    max_node_size: int
    """Threshold for the number of memory nodes to trigger memory compression."""

    max_token_size: int
    """Threshold for the number of tokens to trigger memory compression."""

    system_prompt_template: EjinjaPromptTemplate
    """Template for system prompt used in memory compression."""

    instruction_prompt_template: EjinjaPromptTemplate
    """Instruction prompt template used in memory compression."""

    def __init__(
        self,
        llm: BaseLlm,
        max_node_size: int,
        max_token_size: int,
        system_prompt_template: Optional[str] = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        instruction_prompt_template: Optional[str] = DEFAULT_INSTRUCTION_PROMPT_TEMPLATE,
    ):
        self.llm = llm
        self.max_node_size = max_node_size
        self.max_token_size = max_token_size
        
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
        state_dict["system_prompt_template"] = self.system_prompt_template
        state_dict["instruction_prompt_template"] = self.instruction_prompt_template
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.llm = state_dict["llm"]
        self.max_node_size = state_dict["max_node_size"]
        self.max_token_size = state_dict["max_token_size"]
        self.system_prompt_template = state_dict["system_prompt_template"]
        self.instruction_prompt_template = state_dict["instruction_prompt_template"]