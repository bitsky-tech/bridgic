from typing import Optional, Dict, Any, Union
from typing_extensions import override
from bridgic.core.model import BaseLlm
from bridgic.core.types._serialization import Serializable
from bridgic.core.prompt import EjinjaPromptTemplate


class LlmTaskConfig(Serializable):
    """
    Configuration for a single LLM task in an agentic system.

    This class provides a generic abstraction for configuring LLM tasks with:
    - A dedicated LLM instance for the task
    - Optional system prompt template
    - Optional instruction prompt template

    This class serves as a configuration holder and the actual behavior of the system are 
    determined by the concrete implementations utilizing this configuration.

    Attributes
    ----------
    llm : BaseLlm
        The LLM instance to use for this task.
    system_template : Optional[EjinjaPromptTemplate]
        Optional system prompt template. If None, no system message will be added.
    instruction_template : Optional[EjinjaPromptTemplate]
        Optional instruction prompt template. If None, no instruction message will be added.
    """

    llm: BaseLlm
    """The LLM instance to use for this task."""

    system_template: Optional[EjinjaPromptTemplate]
    """Optional system prompt template for this task."""

    instruction_template: Optional[EjinjaPromptTemplate]
    """Optional instruction prompt template for this task."""

    def __init__(
        self,
        llm: BaseLlm,
        system_template: Optional[Union[str, EjinjaPromptTemplate]] = None,
        instruction_template: Optional[Union[str, EjinjaPromptTemplate]] = None,
    ):
        """
        Initialize LLM task configuration.

        Parameters
        ----------
        llm : BaseLlm
            The LLM instance to use for this task.
        system_template : Optional[Union[str, EjinjaPromptTemplate]]
            System prompt template. Can be a string (will be converted to EjinjaPromptTemplate)
            or an EjinjaPromptTemplate instance. If None, no system message will be added.
        instruction_template : Optional[Union[str, EjinjaPromptTemplate]]
            Instruction prompt template. Can be a string (will be converted to EjinjaPromptTemplate)
            or an EjinjaPromptTemplate instance. If None, no instruction message will be added.
        """
        self.llm = llm

        if system_template is None:
            self.system_template = None
        elif isinstance(system_template, str):
            self.system_template = EjinjaPromptTemplate(system_template)
        elif isinstance(system_template, EjinjaPromptTemplate):
            self.system_template = system_template
        else:
            raise TypeError(
                f"system_template must be str, EjinjaPromptTemplate, or None, "
                f"got {type(system_template)}"
            )

        if instruction_template is None:
            self.instruction_template = None
        elif isinstance(instruction_template, str):
            self.instruction_template = EjinjaPromptTemplate(instruction_template)
        elif isinstance(instruction_template, EjinjaPromptTemplate):
            self.instruction_template = instruction_template
        else:
            raise TypeError(
                f"instruction_template must be str, EjinjaPromptTemplate, or None, "
                f"got {type(instruction_template)}"
            )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["llm"] = self.llm
        state_dict["system_template"] = self.system_template
        state_dict["instruction_template"] = self.instruction_template
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.llm = state_dict["llm"]
        self.system_template = state_dict["system_template"]
        self.instruction_template = state_dict["instruction_template"]

