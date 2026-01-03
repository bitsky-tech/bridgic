from typing import Optional, Union
from bridgic.core.model import BaseLlm
from bridgic.core.agentic.types._llm_task_config import LlmTaskConfig
from bridgic.core.prompt import EjinjaPromptTemplate


DEFAULT_OBSERVATION_SYSTEM_TEMPLATE = (
    "You are an AI assistant that evaluates whether a task goal has been achieved."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
)

DEFAULT_OBSERVATION_INSTRUCTION_TEMPLATE = (
    "Based on the task goal and observation history above, give a brief thinking about the "
    "current status and judge whether the goal has been achieved. Please express concisely."
)

class ObservationTaskConfig:
    """
    Configuration for the observation task in ReCentAutoma.

    This class allows configuring the LLM and prompt templates for the observation task.
    When system_template or instruction_template is None, the default template will be used.

    Attributes
    ----------
    llm : BaseLlm
        The LLM instance to use for this task.
    system_template : Optional[Union[str, EjinjaPromptTemplate]]
        System prompt template. If None, uses DEFAULT_OBSERVE_SYSTEM_TEMPLATE.
    instruction_template : Optional[Union[str, EjinjaPromptTemplate]]
        Instruction prompt template. If None, uses DEFAULT_OBSERVE_INSTRUCTION_TEMPLATE.
    """

    def __init__(
        self,
        llm: BaseLlm,
        system_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_OBSERVATION_SYSTEM_TEMPLATE,
        instruction_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_OBSERVATION_INSTRUCTION_TEMPLATE,
    ):
        self.llm = llm
        self.system_template = system_template or DEFAULT_OBSERVATION_SYSTEM_TEMPLATE
        self.instruction_template = instruction_template or DEFAULT_OBSERVATION_INSTRUCTION_TEMPLATE

    def to_llm_task_config(self) -> LlmTaskConfig:
        return LlmTaskConfig(
            llm=self.llm,
            system_template=self.system_template,
            instruction_template=self.instruction_template,
        )


DEFAULT_TOOL_SELECTION_SYSTEM_TEMPLATE = (
    "You are an AI assistant. You are able to select appropriate tool(s) to help complete "
    "the next step of the task (if needed) in order to ultimately achieve the goal."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
)

DEFAULT_TOOL_SELECTION_INSTRUCTION_TEMPLATE = (
    "Based on the task goal and observation history, select appropriate tool(s) to execute if "
    "there exists any tool that can be used to help achieve the goal."
)

class ToolTaskConfig:
    """
    Configuration for the tool selection task in ReCentAutoma.

    This class allows configuring the LLM and prompt templates for the tool selection task.
    When system_template or instruction_template is None, the default template will be used.

    Attributes
    ----------
    llm : BaseLlm
        The LLM instance to use for this task.
    system_template : Optional[Union[str, EjinjaPromptTemplate]]
        System prompt template. If None, uses DEFAULT_TOOL_SELECTION_SYSTEM_TEMPLATE.
    instruction_template : Optional[Union[str, EjinjaPromptTemplate]]
        Instruction prompt template. If None, uses DEFAULT_TOOL_SELECTION_INSTRUCTION_TEMPLATE.
    """

    def __init__(
        self,
        llm: BaseLlm,
        system_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_TOOL_SELECTION_SYSTEM_TEMPLATE,
        instruction_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_TOOL_SELECTION_INSTRUCTION_TEMPLATE,
    ):
        self.llm = llm
        self.system_template = system_template or DEFAULT_TOOL_SELECTION_SYSTEM_TEMPLATE
        self.instruction_template = instruction_template or DEFAULT_TOOL_SELECTION_INSTRUCTION_TEMPLATE

    def to_llm_task_config(self) -> LlmTaskConfig:
        return LlmTaskConfig(
            llm=self.llm,
            system_template=self.system_template,
            instruction_template=self.instruction_template,
        )


DEFAULT_ANSWER_SYSTEM_TEMPLATE = (
    "You are a helpful assistant. Based on the task goal and the entire conversation history, "
    "provide a clear, complete, and well-structured final answer that addresses the goal."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
)

DEFAULT_ANSWER_INSTRUCTION_TEMPLATE = (
    "Based on the task goal and observation history above, please provide a comprehensive "
    "and well-structured final answer that addresses the goal."
)



class AnswerTaskConfig:
    """
    Configuration for the answer generation task in ReCentAutoma.

    This class allows configuring the LLM and prompt templates for the answer generation task.
    When system_template or instruction_template is None, the default template will be used.

    Attributes
    ----------
    llm : BaseLlm
        The LLM instance to use for this task.
    system_template : Optional[Union[str, EjinjaPromptTemplate]]
        System prompt template. If None, uses DEFAULT_ANSWER_SYSTEM_TEMPLATE.
    instruction_template : Optional[Union[str, EjinjaPromptTemplate]]
        Instruction prompt template. If None, uses DEFAULT_ANSWER_INSTRUCTION_TEMPLATE.
    """

    def __init__(
        self,
        llm: BaseLlm,
        system_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_ANSWER_SYSTEM_TEMPLATE,
        instruction_template: Optional[Union[str, EjinjaPromptTemplate]] = DEFAULT_ANSWER_INSTRUCTION_TEMPLATE,
    ):
        self.llm = llm
        self.system_template = system_template or DEFAULT_ANSWER_SYSTEM_TEMPLATE
        self.instruction_template = instruction_template or DEFAULT_ANSWER_INSTRUCTION_TEMPLATE

    def to_llm_task_config(self) -> LlmTaskConfig:
        return LlmTaskConfig(
            llm=self.llm,
            system_template=self.system_template,
            instruction_template=self.instruction_template,
        )

