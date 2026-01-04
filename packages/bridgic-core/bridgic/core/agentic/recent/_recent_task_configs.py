from typing import Optional, Union
from bridgic.core.model import BaseLlm
from bridgic.core.agentic.types._llm_task_config import LlmTaskConfig
from bridgic.core.prompt import EjinjaPromptTemplate


DEFAULT_OBSERVATION_SYSTEM_TEMPLATE = (
    "You are an AI assistant responsible for evaluating task progress and determining whether "
    "the goal has been achieved. Your need to analyze the current state by examining the conversation "
    "history (which may include compressed summaries of previous stages) and compare it against the "
    "task goal to make an informed judgment."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
    "\n\n## Your Evaluation Process:\n\n"
    "1. Review the conversation history carefully, including any [Stage Summary] entries that represent "
    "compressed memory of previous stages.\n"
    "2. Identify what has been accomplished so far and what information has been gathered.\n"
    "3. Compare the current state against the task goal to identify any remaining gaps.\n"
    "4. Determine if the goal has been fully achieved or if further actions are needed."
)

DEFAULT_OBSERVATION_INSTRUCTION_TEMPLATE = (
    "Based on the task goal and the complete observation history above (including any stage summaries), "
    "provide a brief, focused assessment with \"brief_thinking\" and \"goal_achieved\" fields. "
    "You should focus on: the following aspects:\n"
    "1. What has been accomplished so far and what information has been gathered.\n"
    "2. Are there any remaining gaps between the current state and the goal?\n"
    "3. Has the goal been fully achieved?"
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
    "You are an AI assistant responsible for strategic tool selection to advance task completion. "
    "Your role is to analyze the current progress (including any compressed stage summaries), "
    "identify what needs to be done next, and select the most appropriate tool(s) to take the "
    "next step toward achieving the goal."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
    "\n\n## Your Tool Selection Strategy:\n"
    "1. Review the conversation history, including [Stage Summary] entries that capture previous progress.\n"
    "2. Identify the current gap between what has been accomplished and what the goal requires.\n"
    "3. Determine the most logical next action that will bring you closer to the goal.\n"
    "4. Select tool(s) that are best suited to execute this next action.\n"
    "5. Consider the sequence of actions - some tools may need to be used before others.\n"
    "6. If multiple tools are needed, prioritize based on dependencies and logical flow."
)

DEFAULT_TOOL_SELECTION_INSTRUCTION_TEMPLATE = None
(
    "Based on the task goal and the complete observation history above (including any stage summaries), "
    "analyze what has been accomplished and what remains to be done, then select the appropriate tool(s) to proceed:\n\n"
    "1. What is the current state of progress toward the goal?\n"
    "2. What is the next logical step needed to advance toward the goal?\n"
    "3. Which tool(s) can best execute this next step?\n"
    "4. Are there any dependencies or prerequisites that need to be considered?\n\n"
    "Select one or more tools that will most effectively help you take the next step toward achieving the goal. "
    "If no suitable tool exists or the goal cannot be advanced further with available tools, you may choose not to select any tools."
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
    "You are a helpful assistant responsible for synthesizing all information gathered during the task execution "
    "into a comprehensive final answer. Your role is to review the complete conversation history (including any "
    "compressed stage summaries), extract all relevant information, and present it in a clear, well-organized manner "
    "that fully addresses the task goal."
    "{%- if goal_content %}\n\n## Task Goal:\n\n{{ goal_content }}{% endif %}"
    "{%- if goal_guidance %}\n\n## Task Guidance:\n\n{{ goal_guidance }}{% endif %}"
)

DEFAULT_ANSWER_INSTRUCTION_TEMPLATE = (
    "Based on the task goal and the complete observation history above (including any stage summaries), "
    "synthesize all the information gathered during task execution and provide a comprehensive, "
    "well-structured final answer."
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

