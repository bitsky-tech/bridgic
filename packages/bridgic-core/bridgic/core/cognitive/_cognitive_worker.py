"""
CognitiveWorker - Core component of the cognitive architecture.

A CognitiveWorker represents one thinking cycle of an Agent and is the minimal
programmable unit of "observe-think-act".

Design:
1. Two thinking modes
   - FAST: Merges thinking and tool selection (1 LLM call, single-step only)
   - DEFAULT: Separates thinking and tool selection (2+ calls, single/multi-step)

2. Works directly with CognitiveContext
   - CognitiveWorker is the concrete implementation of GraphAutoma
   - CognitiveContext is the concrete implementation of Context
"""

import json
import time
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

from bridgic.core.model import BaseLlm
from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import ToolCall, Message
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.args import ArgsMappingRule, InOrder
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.utils._console import printer
from bridgic.core.cognitive._context import Step, CognitiveContext


#############################################################################
# Thinking modes
#############################################################################

class ThinkingMode(str, Enum):
    """
    Thinking mode: how thinking and tool selection are orchestrated.

    Attributes
    ----------
    FAST : str
        Fast mode: merge thinking and tool selection into 1 LLM call;
        single-step planning only; suitable for React-style agents.
    DEFAULT : str
        Default mode: separate thinking and tool selection;
        supports single- or multi-step planning; suitable for Plan-style agents.
    """
    FAST = "fast"
    DEFAULT = "default"


#############################################################################
# Data structures
#############################################################################

class DetailRequest(BaseModel):
    """
    Request for detailed information about a specific item in a LayeredExposure field.

    Used in layered disclosure: LLM can request to see details of a specific
    item (e.g., a step in history, a skill) before making a decision.

    Attributes
    ----------
    field : str
        Name of the LayeredExposure field (e.g., "cognitive_history", "skills").
    index : int
        0-based index of the item to get details for.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["field", "index"],
            "additionalProperties": False,
        }
    )
    field: str = Field(description="Name of the field to get details from (e.g., 'cognitive_history', 'skills')")
    index: int = Field(description="0-based index of the item to get details for")


class ToolArgument(BaseModel):
    """
    A single tool argument as name-value pair.

    Attributes
    ----------
    name : str
        Parameter name.
    value : str
        Parameter value (as string, will be converted to appropriate type).
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["name", "value"],
            "additionalProperties": False,
        }
    )
    name: str = Field(description="Parameter name")
    value: str = Field(description="Parameter value as string")


class StepToolCall(BaseModel):
    """
    A single tool call specification.

    Attributes
    ----------
    tool : str
        Name of the tool to call.
    tool_arguments : List[ToolArgument]
        Arguments to pass to the tool as list of name-value pairs.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool", "tool_arguments"],
            "additionalProperties": False,
        }
    )
    tool: str = Field(description="Name of the tool to call")
    tool_arguments: List[ToolArgument] = Field(
        description="Arguments as list of name-value pairs, e.g., [{name: 'city', value: 'Beijing'}]"
    )


class FastThinkResult(BaseModel):
    """
    Think result for fast mode: "what to do" and "which tools" in one output.

    Used when planning a single step; thinking and tool selection are merged
    into one LLM call. Supports layered disclosure via details_needed.

    Attributes
    ----------
    finish : bool
        Whether the task is complete. Set to True if the goal is achieved.
    step_content : str
        Description of what to do in this step.
    calls : List[StepToolCall]
        Tool calls to execute for this step.
    reasoning : Optional[str]
        Brief reasoning for this decision (optional).
    details_needed : List[DetailRequest]
        Request for more details before deciding. If not empty, the system will
        fetch the details and call the LLM again.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["finish", "step_content", "calls", "reasoning", "details_needed"],
            "additionalProperties": False,
        }
    )

    finish: bool = Field(
        default=False,
        description="Whether the task is complete. Set to true if the goal is achieved."
    )
    step_content: str = Field(
        default="",
        description="Description of what to do in this step (can be empty if requesting details)"
    )
    calls: List[StepToolCall] = Field(
        default_factory=list,
        description="Tool calls to execute for this step (can be empty if requesting details)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for this decision (optional)"
    )
    details_needed: List[DetailRequest] = Field(
        default_factory=list,
        description="Request details before deciding. Example: [{field: 'cognitive_history', index: 0}]"
    )


class DefaultThinkResult(BaseModel):
    """
    Think result for default mode: "what to do" with optional detail requests.

    Describes steps without selecting tools. Supports single- or multi-step planning.
    Supports layered disclosure via details_needed (same as FAST mode).

    Attributes
    ----------
    finish : bool
        Whether the task is complete. Set to True if the goal is achieved.
    steps : List[str]
        Step descriptions (one or more). Tool selection happens per step afterward.
    reasoning : Optional[str]
        Overall planning rationale (optional).
    details_needed : List[DetailRequest]
        Request for more details before finalizing steps.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["finish", "steps", "reasoning", "details_needed"],
            "additionalProperties": False,
        }
    )
    finish: bool = Field(
        default=False,
        description="Whether the task is complete. Set to true if the goal is achieved."
    )
    steps: List[str] = Field(
        default_factory=list,
        description=(
            "Steps to execute (can be empty if requesting details first).\n"
            "Examples:\n"
            "- Single step: ['Search for flights from Beijing to Shanghai']\n"
            "- Multiple steps: ['Search flights', 'Book the cheapest flight', 'Search hotels']"
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Overall planning rationale (optional)"
    )
    details_needed: List[DetailRequest] = Field(
        default_factory=list,
        description="Request details before deciding. Example: [{field: 'cognitive_history', index: 0}]"
    )


class FastThinkDecision(BaseModel):
    """
    Decision model for fast mode final round: no details_needed field.

    Used on the last round of thinking to enforce decision output.
    Identical to FastThinkResult but without the details_needed field.

    Attributes
    ----------
    finish : bool
        Whether the task is complete. Set to True if the goal is achieved.
    step_content : str
        Description of what to do in this step.
    calls : List[StepToolCall]
        Tool calls to execute for this step.
    reasoning : Optional[str]
        Brief reasoning for this decision (optional).
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["finish", "step_content", "calls", "reasoning"],
            "additionalProperties": False,
        }
    )

    finish: bool = Field(
        default=False,
        description="Whether the task is complete. Set to true if the goal is achieved."
    )
    step_content: str = Field(
        default="",
        description="Description of what to do in this step"
    )
    calls: List[StepToolCall] = Field(
        default_factory=list,
        description="Tool calls to execute for this step"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief reasoning for this decision (optional)"
    )


class DefaultThinkDecision(BaseModel):
    """
    Decision model for default mode final round: no details_needed field.

    Used on the last round of thinking to enforce decision output.
    Identical to DefaultThinkResult but without the details_needed field.

    Attributes
    ----------
    finish : bool
        Whether the task is complete. Set to True if the goal is achieved.
    steps : List[str]
        Step descriptions (one or more). Tool selection happens per step afterward.
    reasoning : Optional[str]
        Overall planning rationale (optional).
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["finish", "steps", "reasoning"],
            "additionalProperties": False,
        }
    )
    finish: bool = Field(
        default=False,
        description="Whether the task is complete. Set to true if the goal is achieved."
    )
    steps: List[str] = Field(
        default_factory=list,
        description=(
            "Steps to execute.\n"
            "Examples:\n"
            "- Single step: ['Search for flights from Beijing to Shanghai']\n"
            "- Multiple steps: ['Search flights', 'Book the cheapest flight', 'Search hotels']"
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Overall planning rationale (optional)"
    )


class ActionStepResult(BaseModel):
    """
    Result of executing one tool in the action phase.

    Attributes
    ----------
    tool_id : str
        ID of the tool call.
    tool_name : str
        Name of the tool.
    tool_arguments : Dict[str, Any]
        Arguments passed to the tool.
    tool_result : Any
        Raw result returned by the tool.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["tool_id", "tool_name", "tool_arguments", "tool_result"],
            "additionalProperties": False,
        }
    )
    tool_id: str
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any


class ActionResult(BaseModel):
    """
    Overall result of the action phase (one or more tool executions).

    Attributes
    ----------
    status : bool
        True if all tools succeeded, False otherwise.
    results : List[ActionStepResult]
        Per-tool results in order.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["status", "results"],
            "additionalProperties": False,
        }
    )
    status: bool
    results: List[ActionStepResult]


#############################################################################
# CognitiveWorker
#############################################################################

class CognitiveWorker(GraphAutoma):
    """
    Cognitive worker: one thinking cycle of an Agent (observe-think-act).

    One CognitiveWorker represents one full "observe-think-act" cycle.
    Subclass and override template methods to customize behavior.

    Parameters
    ----------
    llm : BaseLlm
        LLM used for thinking.
    mode : ThinkingMode, optional
        Thinking mode: FAST or DEFAULT. Default is DEFAULT.
        - FAST: Single call (step + tools + details check in one)
        - DEFAULT: Separate thinking and tool selection phases
    max_detail_rounds : int, optional
        Maximum rounds of progressive disclosure queries before proceeding.
        Default is 1. Set higher to allow more detail fetching iterations.
    verbose : bool, optional
        Enable logging of thinking process and tool execution. Default is False.
    verbose_prompt : bool, optional
        Enable logging of full prompts sent to LLM. Default is False.
        Separate from verbose to avoid cluttering output with long prompts.

    Template Methods (override in subclasses)
    -----------------------------------------
    thinking() -> str
        Return the thinking prompt (how to decide next steps). Must be implemented.

    build_thinking_prompt(think_prompt, tools_description, output_instructions, context_info)
        Assemble the final prompts for the thinking phase. Override to customize
        prompt structure (reorder, add extra instructions, etc.).
        Default: concatenates components as system_prompt, context_info as user_prompt.

    observation(latest_info) -> str
        Transform the last step result into a form suitable for thinking.
        Default: return as string.

    select_tools(step_content, context) -> List[ToolCall]
        Select tools for a step. Default: LLM-based selection.
        Only used in DEFAULT mode (FAST mode selects tools during thinking).

    verify_tools(matched_list, context) -> List[Tuple[ToolCall, ToolSpec]]
        Verify/adjust matched tools before execution. Default: no change.

    consequence(action_results) -> Any
        Process tool execution results (format or summarize). Default: return raw.

    Examples
    --------
    >>> class ReactWorker(CognitiveWorker):
    ...     def __init__(self, llm):
    ...         # FAST mode with 2 detail rounds
    ...         super().__init__(llm, mode=ThinkingMode.FAST, max_detail_rounds=2)
    ...
    ...     async def thinking(self):
    ...         return "Plan ONE immediate next step with appropriate tools."
    ...
    >>> class PlanWorker(CognitiveWorker):
    ...     def __init__(self, llm):
    ...         # DEFAULT mode
    ...         super().__init__(llm)
    ...
    ...     async def thinking(self):
    ...         return "Create a step-by-step plan to achieve the goal."
    """

    def __init__(
        self,
        llm: BaseLlm,
        mode: ThinkingMode = ThinkingMode.DEFAULT,
        max_detail_rounds: int = 1,
        verbose: bool = False,
        verbose_prompt: bool = False,
    ):
        super().__init__()
        self._llm = llm
        self.mode = mode
        self.max_detail_rounds = max_detail_rounds

        # Logging runtime
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Usage stats
        self.spend_tokens = 0
        self.spend_time = 0

    ############################################################################
    # Worker methods (GraphAutoma execution flow)
    ############################################################################

    @worker(is_start=True)
    async def _observation(self, context: CognitiveContext) -> Optional[str]:
        """
        Observation phase: prepare context for thinking.

        Calls the user-overridable observation() method which can return
        additional context to include in the thinking phase.
        """
        if not isinstance(context, CognitiveContext):
            raise TypeError(
                f"Expected CognitiveContext, got {type(context).__name__}. "
                "CognitiveWorker requires CognitiveContext or its subclass."
            )

        # Call user-overridable observation hook
        observation = await self.observation(context)
        if observation:
            self._log("Observe", "Custom observation", observation, color="cyan")

        return observation

    @worker(dependencies=["_observation"])
    async def _thinking(self, observation: Optional[str], context: CognitiveContext) -> Any:
        """
        Thinking phase: decide what to do next.

        Dispatches to fast (thinking + tool selection in one call) or
        default (thinking only; tool selection per step later) based on thinking mode.
        """
        if self.mode == ThinkingMode.FAST:
            await self._fast_thinking(observation, context)
        else:
            await self._default_thinking(observation, context)

    async def _fast_thinking(self, observation: Optional[str], context: CognitiveContext):
        """Fast mode thinking with layered disclosure support."""
        self._log("Think", "Mode: FAST (thinking + tool selection)", color="blue")

        think_prompt = await self.thinking()
        think_result: Union[FastThinkResult, FastThinkDecision] = None

        for round_idx in range(self.max_detail_rounds + 1):
            is_last_round = (round_idx == self.max_detail_rounds)

            if is_last_round:
                # Last round: use Decision schema (no details_needed)
                system_prompt, user_prompt = await self._build_fast_prompts(
                    think_prompt=think_prompt,
                    context=context,
                    observation=observation,
                    is_final_round=True
                )
                think_result = await self._llm.astructured_output(
                    messages=[
                        Message.from_text(text=system_prompt, role="system"),
                        Message.from_text(text=user_prompt, role="user")
                    ],
                    constraint=PydanticModel(model=FastThinkDecision)
                )
                self._log_prompt(f"Think round {round_idx + 1} (final)", system_prompt, user_prompt)
                break  # Always exit on last round
            else:
                # Non-last round: use ThinkResult schema (allows details_needed)
                system_prompt, user_prompt = await self._build_fast_prompts(
                    think_prompt=think_prompt,
                    context=context,
                    observation=observation,
                    is_final_round=False
                )
                think_result = await self._llm.astructured_output(
                    messages=[
                        Message.from_text(text=system_prompt, role="system"),
                        Message.from_text(text=user_prompt, role="user")
                    ],
                    constraint=PydanticModel(model=FastThinkResult)
                )
                self._log_prompt(f"Think round {round_idx + 1}", system_prompt, user_prompt)

                # Check if LLM needs more details
                if think_result.details_needed:
                    reqs = [f"{r.field}[{r.index}]" for r in think_result.details_needed]
                    self._log("Think", f"Requesting details (round {round_idx + 1}): {reqs}", color="blue")
                    for req in think_result.details_needed:
                        context.get_details(req.field, req.index)
                    continue

                # LLM gave a decision, exit loop
                break

        # Ferry to next worker and log result
        if think_result.finish:
            self._log("Think", "Result: Task completed (finish=True)", color="blue")
            self.ferry_to("_finish", context=context)
        else:
            tools = [c.tool for c in think_result.calls]
            self._log("Think", f"Result: step=\"{think_result.step_content}\" tools={tools}, reasoning=\"{think_result.reasoning}\"", color="blue")
            self.ferry_to("_action_fast", think_result=think_result, context=context)

    async def _default_thinking(self, observation: Optional[str], context: CognitiveContext):
        """Default mode thinking with layered disclosure."""
        self._log("Think", "Mode: DEFAULT (thinking only)", color="blue")

        think_prompt = await self.thinking()
        think_result: Union[DefaultThinkResult, DefaultThinkDecision] = None

        for round_idx in range(self.max_detail_rounds + 1):
            is_last_round = (round_idx == self.max_detail_rounds)

            if is_last_round:
                # Last round: use Decision schema (no details_needed)
                system_prompt, user_prompt = await self._build_default_prompts(
                    think_prompt=think_prompt,
                    context=context,
                    observation=observation,
                    is_final_round=True
                )
                think_result = await self._llm.astructured_output(
                    messages=[
                        Message.from_text(text=system_prompt, role="system"),
                        Message.from_text(text=user_prompt, role="user")
                    ],
                    constraint=PydanticModel(model=DefaultThinkDecision)
                )
                self._log_prompt(f"Think round {round_idx + 1} (final)", system_prompt, user_prompt)
                break  # Always exit on last round
            else:
                # Non-last round: use ThinkResult schema (allows details_needed)
                system_prompt, user_prompt = await self._build_default_prompts(
                    think_prompt=think_prompt,
                    context=context,
                    observation=observation,
                    is_final_round=False
                )
                think_result = await self._llm.astructured_output(
                    messages=[
                        Message.from_text(text=system_prompt, role="system"),
                        Message.from_text(text=user_prompt, role="user")
                    ],
                    constraint=PydanticModel(model=DefaultThinkResult)
                )
                self._log_prompt(f"Think round {round_idx + 1}", system_prompt, user_prompt)

                # Check if LLM needs more details
                if think_result.details_needed:
                    reqs = [f"{r.field}[{r.index}]" for r in think_result.details_needed]
                    self._log("Think", f"Requesting details (round {round_idx + 1}): {reqs}", color="blue")
                    for req in think_result.details_needed:
                        context.get_details(req.field, req.index)
                    continue

                # LLM gave a decision, exit loop
                break

        # Ferry to next worker and log result
        if think_result.finish:
            self._log("Think", "Result: Task completed (finish=True)", color="blue")
            self.ferry_to("_finish", context=context)
        else:
            self._log("Think", f"Result: steps={think_result.steps}, reasoning={think_result.reasoning}", color="blue")
            self.ferry_to(
                "_action_default",
                think_result=think_result,
                context=context
            )

    @worker()
    async def _action_fast(self, think_result: Union[FastThinkResult, FastThinkDecision], context: CognitiveContext) -> CognitiveContext:
        tool_calls = self._create_tool_calls_from_fast(think_result, context)
        await self._execute_step(think_result.step_content, tool_calls, context)
        return context

    @worker()
    async def _action_default(self, think_result: Union[DefaultThinkResult, DefaultThinkDecision], context: CognitiveContext) -> CognitiveContext:
        """Action phase for DEFAULT mode: select tools and execute each step."""
        for i, step_content in enumerate(think_result.steps, 1):
            self._log("Action", f"Step {i}/{len(think_result.steps)}: {step_content}", color="green")
            tool_calls = await self.select_tools(step_content, context)
            tools = [tc.name for tc in tool_calls]
            self._log("Action", f"Tools selected: {tools}", color="green")
            await self._execute_step(step_content, tool_calls, context)

        return context

    @worker(is_output=True)
    async def _finish(self, context: CognitiveContext) -> CognitiveContext:
        context.set_finish()
        self._log("Finish", "Task completed successfully", color="lime")
        return context

    ############################################################################
    # Internal helpers
    ############################################################################

    def _log(self, stage: str, message: str, data: Any = None, color: str = "white"):
        """Log formatted message if verbose mode is enabled."""
        if not self._verbose:
            return

        prefix = f"[{stage}]"
        if data is not None:
            data_str = str(data)
            printer.print(f"{prefix} {message}", color=color)
            printer.print(f"{data_str}", color="gray")
        else:
            printer.print(f"{prefix} {message}", color=color)

    def _log_prompt(self, stage: str, system_prompt: str, user_prompt: str):
        """Log prompts if verbose_prompt mode is enabled."""
        if not self._verbose_prompt:
            return

        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        printer.print(f"[{stage}] System Prompt ({system_tokens} tokens):", color="cyan")
        printer.print(system_prompt, color="gray")
        printer.print(f"[{stage}] User Prompt ({user_tokens} tokens):", color="cyan")
        printer.print(user_prompt, color="gray")
        printer.print(f"[{stage}] Total: {total_tokens} tokens (cumulative: {self.spend_tokens} tokens)", color="yellow")

    def _log_prompt_with_tools(self, stage: str, system_prompt: str, user_prompt: str, tools_tokens: int):
        """Log prompts with tool schema tokens if verbose_prompt mode is enabled."""
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens + tools_tokens

        # Always update spend_tokens
        self.spend_tokens += total_tokens

        if not self._verbose_prompt:
            return

        printer.print(f"[{stage}] System Prompt ({system_tokens} tokens):", color="cyan")
        printer.print(system_prompt, color="gray")
        printer.print(f"[{stage}] User Prompt ({user_tokens} tokens):", color="cyan")
        printer.print(user_prompt, color="gray")
        printer.print(f"[{stage}] Tool JSON Schema ({tools_tokens} tokens):", color="cyan")
        printer.print(f"  (tool definitions are sent to LLM for function calling)", color="gray")
        printer.print(f"[{stage}] Total: {total_tokens} tokens (cumulative: {self.spend_tokens} tokens)", color="yellow")

    async def _build_fast_prompts(
        self,
        think_prompt: str,
        context: CognitiveContext,
        observation: Optional[str] = None,
        is_final_round: bool = False
    ) -> Tuple[str, str]:
        """Build prompts for FAST mode (thinking + tool selection in one call).

        Parameters
        ----------
        think_prompt : str
            The thinking prompt from the thinking() method.
        context : CognitiveContext
            The cognitive context.
        observation : Optional[str]
            Custom observation from user-overridden observation() method.
        is_final_round : bool
            If True, output instructions will not include details_needed field,
            forcing the LLM to make a decision.
        """
        # Build context info (exclude tools and skills, they go in system prompt)
        context_info = context.format_summary(exclude=['tools', 'skills'])
        if observation:
            context_info += f"\n\nObservation:\n{observation}"

        # For FAST mode, we need detailed tool info (with parameters)
        # Build tool details directly from ToolSpec list
        capabilities_parts = []
        _, tool_specs = context.get_field('tools')

        def _format_tools_details(tool_specs: List[ToolSpec]) -> str:
            """Format tool specs into detailed description string (for FAST mode prompts)."""
            lines = []
            for tool in tool_specs:
                tool_lines = [f"• {tool.tool_name}: {tool.tool_description}"]

                if tool.tool_parameters:
                    props = tool.tool_parameters.get('properties', {})
                    required = tool.tool_parameters.get('required', [])

                    if props:
                        for name, info in props.items():
                            param_type = info.get('type', 'any')
                            param_desc = info.get('description', '')
                            is_required = name in required

                            req_mark = " [required]" if is_required else " [optional]"
                            param_line = f"  - {name} ({param_type}){req_mark}"
                            if param_desc:
                                param_line += f": {param_desc}"
                            tool_lines.append(param_line)

                lines.extend(tool_lines)

            return "\n".join(lines)

        if tool_specs:
            tools_details = _format_tools_details(tool_specs)
            capabilities_parts.append(f"# Available Tools (with parameters):\n{tools_details}")

        # Add skills to system_prompt (capabilities)
        skills_summary = context.summary().get('skills')
        if skills_summary:
            capabilities_parts.append(f"# {skills_summary}")

        capabilities_description = "\n\n".join(capabilities_parts)

        # Prepare output instructions based on whether this is the final round
        if is_final_round:
            # Final round: no details_needed, must make a decision
            output_instructions = (
                "# Output Format:\n"
                "- **finish**: true only when goal is fully achieved\n"
                "- **step_content**: Description of what to do in this step\n"
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name, value}]}]\n"
                "\n"
                "# Execution Guidelines:\n"
                "- Focus on ONE step at a time\n"
                "- Tool arguments format: [{args_name: 'name', args_value: 'value'}]"
            )
        else:
            # Non-final round: include details_needed option
            output_instructions = (
                "# Output Format:\n"
                "- **finish**: true only when goal is fully achieved\n"
                "- **step_content**: Description of what to do in this step (empty if requesting details)\n"
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name, value}]}]\n"
                "- **details_needed**: Request details via [{field: 'xxx', index: N}]\n"
                "\n"
                "# BEFORE Taking Action - Check If You Need Details:\n"
                "You MUST check the following before calling any tools:\n"
                "\n"
                "1. **skills** (CHECK FIRST):\n"
                "   - Look at the Available Skills list above\n"
                "   - Is there a skill that matches your current task?\n"
                "   - If YES and you haven't seen its details yet: Request its details FIRST via details_needed\n"
                "   - If YES and you already have the skill details: IMMEDIATELY start executing the workflow - do NOT request details again\n"
                "   - If NO: Proceed directly with tools (don't request skills that don't match)\n"
                "\n"
                "2. **cognitive_history** (CHECK WHEN NEEDED):\n"
                "   - Do you need specific data from a previous step (e.g., flight numbers, hotel names)?\n"
                "   - If YES: Request the step's details to extract the information\n"
                "   - If NO: Don't request history details unnecessarily\n"
                "\n"
                "# Execution Guidelines:\n"
                "- Focus on ONE step at a time\n"
                "- When requesting details, leave step_content and calls empty\n"
                "- Tool arguments format: [{args_name: 'name', args_value: 'value'}]"
            )

        # Call template method to assemble final prompts
        system_prompt, user_prompt = await self.build_thinking_prompt(
            think_prompt=think_prompt.strip(),
            tools_description=capabilities_description,  # Now includes both tools and skills
            output_instructions=output_instructions,
            context_info=context_info
        )

        self.spend_tokens += self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        return system_prompt, user_prompt

    async def _build_default_prompts(
        self,
        think_prompt: str,
        context: CognitiveContext,
        observation: Optional[str] = None,
        is_final_round: bool = False
    ) -> Tuple[str, str]:
        """Build prompts for DEFAULT mode (thinking only; tool selection done separately).

        Parameters
        ----------
        think_prompt : str
            The thinking prompt from the thinking() method.
        context : CognitiveContext
            The cognitive context.
        observation : Optional[str]
            Custom observation from user-overridden observation() method.
        is_final_round : bool
            If True, output instructions will not include details_needed field,
            forcing the LLM to make a decision.
        """
        # Build context info (exclude tools and skills, they go in system prompt)
        context_info = context.format_summary(exclude=['tools', 'skills'])
        if observation:
            context_info += f"\n\nObservation:\n{observation}"

        # Build capabilities description (tools + skills) for system_prompt
        capabilities_description = context.format_summary(include=['tools', 'skills'])

        # Prepare output instructions based on whether this is the final round
        if is_final_round:
            # Final round: no details_needed, must make a decision
            output_instructions = (
                "# Output Format:\n"
                "- **finish**: true only when goal is fully achieved\n"
                "- **steps**: List of step descriptions\n"
                "\n"
                "# Planning Guidelines:\n"
                "- If a matching skill exists, follow its defined process strictly\n"
                "- Each step should map to ONE logical action\n"
                "- Only describe WHAT to do (tool selection is done separately)"
            )
        else:
            # Non-final round: include details_needed option
            output_instructions = (
                "# Output Format:\n"
                "- **finish**: true only when goal is fully achieved\n"
                "- **steps**: List of step descriptions (leave empty if requesting details first)\n"
                "- **details_needed**: Request details via [{field: 'xxx', index: N}]\n"
                "\n"
                "# BEFORE Planning - Check If You Need Details:\n"
                "You MUST check the following before planning steps:\n"
                "\n"
                "1. **skills** (CHECK FIRST - CRITICAL):\n"
                "   - Look at the Available Skills list above\n"
                "   - Is there a skill that matches your current task?\n"
                "   - If YES and you haven't seen its details: Request details FIRST via details_needed, leave steps empty\n"
                "   - If YES and you already have the skill details: Follow the skill's workflow STRICTLY. **DO NOT REQUEST SAME ITEM AGAIN**\n"
                "   - If NO matching skill: Plan based on available tools\n"
                "\n"
                "2. **cognitive_history** (CHECK WHEN NEEDED):\n"
                "   - Do you need specific data from a previous step (e.g., flight numbers, hotel names)?\n"
                "   - If YES: Request the step's details to extract the information\n"
                "   - If NO: Don't request history details unnecessarily\n"
                "\n"
                "# Planning Guidelines:\n"
                "- If a matching skill exists, follow its defined process strictly\n"
                "- Each step should map to ONE logical action\n"
                "- Only describe WHAT to do (tool selection is done separately)\n"
                "- When requesting details, leave steps empty; plan after receiving details"
            )

        # Call template method to assemble final prompts
        system_prompt, user_prompt = await self.build_thinking_prompt(
            think_prompt=think_prompt.strip(),
            tools_description=capabilities_description,  # Now includes tools and skills
            output_instructions=output_instructions,
            context_info=context_info
        )

        self.spend_tokens += self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        return system_prompt, user_prompt

    def _build_tool_selection_prompts(self, step_content: str, context: CognitiveContext) -> Tuple[str, str]:
        """Build prompts for tool selection (DEFAULT mode).

        Parameters
        ----------
        step_content : str
            Description of the step to execute.
        context : CognitiveContext
            The cognitive context (contains goal, history, disclosed details).
        """
        system_parts = [
            "You are a tool selection assistant. "
            "Select the appropriate tool(s) to execute the given step. "
            "Analyze the step content and choose the tool(s) that best match the required action. "
            "Use the provided context information (including any detailed information) to determine "
            "the correct arguments for the tools."
        ]
        system_parts.append(context.format_summary(include=['tools']))
        system_prompt = "\n\n".join(system_parts)

        user_parts = [context.format_summary(include=['disclosed_details', 'cognitive_history'])]
        user_parts.append(f"Step to execute: {step_content}")
        user_prompt = "\n\n".join(user_parts)

        # Note: token counting is done in _log_prompt_with_tools which includes tool schema tokens
        self.spend_tokens += self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        return system_prompt, user_prompt

    async def _select_tools_for_step(self, step_content: str, context: CognitiveContext) -> List[ToolCall]:
        """Select tools for a single step via LLM (DEFAULT mode).

        Parameters
        ----------
        step_content : str
            Description of the step to execute.
        context : CognitiveContext
            The cognitive context (contains goal, history, disclosed details).
        """
        system_prompt, user_prompt = self._build_tool_selection_prompts(
            step_content=step_content,
            context=context
        )

        # Get tools from context
        _, tool_specs = context.get_field('tools')
        tools = [tool_spec.to_tool() for tool_spec in tool_specs]
        tool_calls, _ = await self._llm.aselect_tool(
            messages=[
                Message.from_text(text=system_prompt, role="system"),
                Message.from_text(text=user_prompt, role="user")
            ],
            tools=tools
        )

        # Calculate tool definition tokens (tool json_schema is also sent to LLM) and log prompts with tool tokens included
        tools_json = json.dumps([t.model_dump() if hasattr(t, 'model_dump') else str(t) for t in tools], ensure_ascii=False)
        tools_tokens = self._count_tokens(tools_json)
        self._log_prompt_with_tools("Tool Selection", system_prompt, user_prompt, tools_tokens)
        self.spend_tokens += tools_tokens
        return tool_calls

    def _create_tool_calls_from_fast(self, think_result: Union[FastThinkResult, FastThinkDecision], context: CognitiveContext) -> List[ToolCall]:
        """Convert FastThinkResult into a list of ToolCall."""
        tool_calls = []

        def _find_tool_by_name(tool_specs: List[ToolSpec], name: str) -> Optional[ToolSpec]:
            """Find a tool spec by name from the list."""
            for tool in tool_specs:
                if tool.tool_name == name:
                    return tool
            return None

        def _convert_tool_arguments(tool_name: str, tool_arguments: List[ToolArgument], context: CognitiveContext) -> Dict[str, Any]:
            """Convert List[ToolArgument] to Dict[str, Any] with type conversion."""
            # Find tool spec by name from the list
            _, tool_specs = context.get_field('tools')
            tool_spec = _find_tool_by_name(tool_specs, tool_name)

            # Get parameter type info
            param_types = {}
            if tool_spec and tool_spec.tool_parameters:
                props = tool_spec.tool_parameters.get('properties', {})
                for name, info in props.items():
                    param_types[name] = info.get('type', 'string')

            # Convert arguments
            result = {}
            for arg in tool_arguments:
                value = arg.value
                param_type = param_types.get(arg.name, 'string')

                # Type conversion based on parameter type
                if param_type == 'integer':
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                elif param_type == 'number':
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        pass
                elif param_type == 'boolean':
                    value = value.lower() in ('true', '1', 'yes')
                # string type: keep as-is

                result[arg.name] = value

            return result

        for idx, call in enumerate(think_result.calls):
            # Convert List[ToolArgument] to Dict[str, Any]
            arguments = _convert_tool_arguments(call.tool, call.tool_arguments, context)
            tool_calls.append(ToolCall(
                id=f"call_{idx}",
                name=call.tool,
                arguments=arguments
            ))
        return tool_calls

    async def _execute_step(self, step_content: str, tool_calls: List[ToolCall], context: CognitiveContext):
        """Execute one step: match tools, verify, run, process result, update history."""
        # Get tools from context
        _, tool_specs = context.get_field('tools')
        matched_list = self._match_tool_calls(tool_calls, tool_specs)
        verify_list = await self.verify_tools(matched_list, context)
        if not verify_list:
            raise ValueError(f"No matching tools found for: {[tc.name for tc in tool_calls]}")

        # Log tool execution
        for tc, _ in verify_list:
            self._log("Action", f"Executing: {tc.name}({tc.arguments})", color="green")

        action_result = await self._execute_tools(verify_list)
        consequence = await self.consequence(action_result.results)

        # Log result
        status = "success" if action_result.status else "failed"
        self._log("Action", f"Result: {status}", consequence, color="green")

        info = Step(
            content=step_content,
            status=action_result.status,
            result=consequence,
            metadata={"tool_calls": [tc[0].name for tc in verify_list]}
        )
        context.add_info(info)

    async def _execute_tools(self, matched_list: List[Tuple[ToolCall, ToolSpec]]) -> ActionResult:
        """Execute a list of (ToolCall, ToolSpec) pairs and return ActionResult."""
        sandbox = ConcurrentAutoma()
        tool_calls = []

        for tool_call, tool_spec in matched_list:
            tool_calls.append(tool_call)
            tool_worker = tool_spec.create_worker()
            worker_key = f"tool_{tool_call.name}_{tool_call.id}"
            sandbox.add_worker(
                key=worker_key,
                worker=tool_worker,
                args_mapping_rule=ArgsMappingRule.UNPACK
            )

        tool_args = [tc.arguments for tc in tool_calls]

        try:
            results = await sandbox.arun(InOrder(tool_args))
            step_results = [
                ActionStepResult(
                    tool_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=tc.arguments,
                    tool_result=result
                )
                for tc, result in zip(tool_calls, results)
            ]
            return ActionResult(status=True, results=step_results)
        except Exception as e:
            return ActionResult(
                status=False,
                results=[ActionStepResult(
                    tool_id="",
                    tool_name="",
                    tool_arguments={},
                    tool_result=traceback.format_exc()
                )]
            )

    def _match_tool_calls(
        self,
        tool_calls: List[ToolCall],
        tool_specs: List[ToolSpec]
    ) -> List[Tuple[ToolCall, ToolSpec]]:
        """Match each ToolCall to its ToolSpec by name."""
        matched = []
        for tc in tool_calls:
            for spec in tool_specs:
                if tc.name == spec.tool_name:
                    if tc.arguments.get("__args__") is not None:
                        props = list(spec.tool_parameters.get('properties', {}).keys())
                        args = tc.arguments.get("__args__")
                        if isinstance(args, list):
                            tc.arguments = dict(zip(props, args))
                        else:
                            tc.arguments = {props[0]: args} if props else {}
                    matched.append((tc, spec))
                    break
        return matched

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Rough approximation: ~4 chars per token (typical for English/UTF-8)."""
        return (len(text) + 3) // 4

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    async def observation(self, context: CognitiveContext) -> Optional[str]:
        """
        Hook for custom observation logic before thinking.

        Override to add custom observation that will be included in the
        thinking context. The context's cognitive_history already contains
        layered history (working memory with full details, short-term with summaries).

        Parameters
        ----------
        context : CognitiveContext
            The cognitive context.

        Returns
        -------
        Optional[str]
            Custom observation to include in thinking context.
            Return None or empty string to skip.

        Examples
        --------
        >>> async def observation(self, context):
        ...     # Add custom observation based on latest step
        ...     if len(context.cognitive_history) > 0:
        ...         latest = context.cognitive_history[-1]
        ...         return f"Last action: {latest.content}, Result: {latest.result}"
        ...     return None
        """
        return None

    async def thinking(self) -> str:
        """
        Define how to think about the next step(s). Must be implemented.

        The returned prompt is used to guide the LLM. This is the core template method.

        Returns
        -------
        str
            Thinking prompt for the LLM.

        Examples
        --------
        >>> async def thinking(self):
        ...     return "Plan ONE immediate next step."  # React-style
        ...
        >>> async def thinking(self):
        ...     return "Create a complete step-by-step plan."  # Plan-style
        """
        raise NotImplementedError("thinking() must be implemented")

    async def build_thinking_prompt(
        self,
        think_prompt: str,
        tools_description: str,
        output_instructions: str,
        context_info: str,
    ) -> Tuple[str, str]:
        """
        Assemble the final prompts for the thinking phase.

        Override this method to customize how the prompt components are combined.
        This allows you to reorder, modify, or add to the prompt structure.

        Parameters
        ----------
        think_prompt : str
            The thinking prompt from the thinking() method.
        tools_description : str
            Formatted description of available tools.
            - FAST mode: detailed with parameters
            - DEFAULT mode: empty (tools already in context_info)
        output_instructions : str
            Instructions for the output format (finish, steps/step_content, etc.).
        context_info : str
            Context information including goal, status, tools summary, skills,
            history, and fetched details.

        Returns
        -------
        Tuple[str, str]
            (system_prompt, user_prompt) to be sent to the LLM.

        Examples
        --------
        >>> async def build_thinking_prompt(self, think_prompt, tools_description,
        ...                                  output_instructions, context_info):
        ...     # Custom: put output_instructions first for emphasis
        ...     system_prompt = f"{output_instructions}\\n\\n{think_prompt}\\n\\n{tools_description}"
        ...     return system_prompt, context_info
        ...
        >>> async def build_thinking_prompt(self, think_prompt, tools_description,
        ...                                  output_instructions, context_info):
        ...     # Custom: add extra instructions
        ...     extra = "Remember: Always plan multiple steps for complex tasks."
        ...     system_prompt = f"{think_prompt}\\n\\n{extra}\\n\\n{tools_description}\\n\\n{output_instructions}"
        ...     return system_prompt, context_info
        """
        parts = [think_prompt]
        if tools_description:
            parts.append(tools_description)
        parts.append(output_instructions)
        system_prompt = "\n\n".join(parts)

        user_prompt = context_info

        return system_prompt, user_prompt

    async def consequence(self, action_results: List[ActionStepResult]) -> Any:
        """
        Process tool execution results (format or summarize). Default: return raw list.

        Parameters
        ----------
        action_results : List[ActionStepResult]
            Per-tool results from the action phase.

        Returns
        -------
        Any
            Value stored as the step result in cognitive history.
        """
        return action_results

    async def select_tools(self, step_content: str, context: CognitiveContext) -> List[ToolCall]:
        """
        Select tools for a step. Override to customize tool selection logic.

        This method is called in DEFAULT mode to select tools for each step.
        (In FAST mode, tools are already selected during thinking.)

        Default: uses LLM-based tool selection via aselect_tool.

        Parameters
        ----------
        step_content : str
            Description of the step to execute.
        context : CognitiveContext
            Current cognitive context. Previously disclosed details are available
            via context.summary()['disclosed_details'].

        Returns
        -------
        List[ToolCall]
            Selected tool calls for this step.

        Examples
        --------
        >>> async def select_tools(self, step_content, context):
        ...     # Custom logic: always use a specific tool
        ...     return [ToolCall(id="1", name="my_tool", arguments={"arg": "value"})]
        ...
        >>> async def select_tools(self, step_content, context):
        ...     # Access disclosed details from context if needed
        ...     disclosed = context.summary().get('disclosed_details', '')
        ...     return [ToolCall(id="1", name="search", arguments={"query": step_content})]
        """
        return await self._select_tools_for_step(step_content, context)

    async def verify_tools(self, matched_list: List[Tuple[ToolCall, ToolSpec]], context: CognitiveContext) -> List[Tuple[ToolCall, ToolSpec]]:
        """
        Verify and optionally adjust matched tools before execution.

        Called after tool matching, before actual execution. Override to
        validate, filter, or modify the tools that will be executed.

        Default: return as-is (no verification).

        Parameters
        ----------
        matched_list : List[Tuple[ToolCall, ToolSpec]]
            Pairs of (tool_call, tool_spec) ready for execution.
        context : CognitiveContext
            Current cognitive context.

        Returns
        -------
        List[Tuple[ToolCall, ToolSpec]]
            Verified/adjusted list of tools to execute.

        Examples
        --------
        >>> async def verify_tools(self, matched_list, context):
        ...     # Filter out dangerous tools
        ...     return [(tc, spec) for tc, spec in matched_list
        ...             if spec.tool_name not in ["delete", "drop"]]
        """
        return matched_list

    ############################################################################
    # Entry point
    ############################################################################

    async def arun(
        self,
        *args: Tuple[Any, ...],
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs
    ) -> Any:
        """Run the observe-think-act cycle. First arg should be CognitiveContext."""
        start_time = time.time()
        result = await super().arun(*args, feedback_data=feedback_data, **kwargs)
        self.spend_time += time.time() - start_time
        return result
