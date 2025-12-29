from typing import Dict, Any
from typing_extensions import override
from bridgic.core.model import BaseLlm
from bridgic.core.types._serialization import Serializable


class CompressionConfig(Serializable):
    """Memory compression configuration."""

    llm: BaseLlm
    """LLM model used for memory compression."""

    max_node_size: int
    """Threshold for the number of memory nodes to trigger memory compression."""

    max_token_size: int
    """Threshold for the number of tokens to trigger memory compression."""

    def __init__(
        self,
        llm: BaseLlm,
        max_node_size: int,
        max_token_size: int
    ):
        self.llm = llm
        self.max_node_size = max_node_size
        self.max_token_size = max_token_size

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["llm"] = self.llm
        state_dict["max_node_size"] = self.max_node_size
        state_dict["max_token_size"] = self.max_token_size
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.llm = state_dict["llm"]
        self.max_node_size = state_dict["max_node_size"]
        self.max_token_size = state_dict["max_token_size"]