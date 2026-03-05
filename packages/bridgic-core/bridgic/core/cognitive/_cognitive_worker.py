"""
CognitiveWorker - Core component of the cognitive architecture.

A CognitiveWorker represents one thinking cycle of an Agent and is the minimal
programmable unit of "observe-think-act".

Design:
1. Single thinking mode: Merges thinking and tool selection into 1 LLM call.

2. Works directly with CognitiveContext
   - CognitiveWorker is the concrete implementation of GraphAutoma
   - CognitiveContext is the concrete implementation of Context
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

from bridgic.core.model import BaseLlm
from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import ToolCall, Message
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.interaction import InteractionFeedback
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.utils._console import printer
from bridgic.core.cognitive._context import CognitiveContext


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

    @field_validator('value', mode='before')
    @classmethod
    def coerce_to_str(cls, v: Any) -> str:
        return str(v) if not isinstance(v, str) else v


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


class ThinkResult(BaseModel):
    """
    Think result: "what to do" and "which tools" in one output.

    Used when planning a single step; thinking and tool selection are merged
    into one LLM call. Supports layered disclosure via details_needed.

    Note: Worker executes one thinking cycle without deciding task completion.
    Termination is controlled by Agent via until() conditions.

    Attributes
    ----------
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
            "required": ["step_content", "calls", "reasoning", "details_needed"],
            "additionalProperties": False,
        }
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

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator('calls', 'details_needed', mode='before')
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return [] if v is None else v


class ThinkDecision(BaseModel):
    """
    Decision model for the final round: no details_needed field.

    Used on the last round of thinking to enforce decision output.
    Identical to ThinkResult but without the details_needed field.

    Note: Worker executes one thinking cycle without deciding task completion.
    Termination is controlled by Agent via until() conditions.

    Attributes
    ----------
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
            "required": ["step_content", "calls", "reasoning"],
            "additionalProperties": False,
        }
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

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator('calls', mode='before')
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return [] if v is None else v


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
# Sentinel
#############################################################################

_DELEGATE = object()  # Worker returns this to delegate observation to Agent


#############################################################################
# CognitiveWorker
#############################################################################

class CognitiveWorker(GraphAutoma):
    """
    Cognitive worker: pure thinking unit of an Agent.

    A CognitiveWorker represents one thinking cycle and is responsible for
    "what to think and how to think". Observation and action execution are
    handled by AgentAutoma as shared infrastructure.

    Subclass and override template methods to customize behavior.

    Parameters
    ----------
    llm : Optional[BaseLlm]
        LLM used for thinking. Can be None if the agent will inject one via set_llm().
    max_detail_rounds : int, optional
        Maximum rounds of progressive disclosure queries before proceeding.
        Default is 1. Set higher to allow more detail fetching iterations.
    verbose : bool, optional
        Enable logging of thinking process. Default is False.
    verbose_prompt : bool, optional
        Enable logging of full prompts sent to LLM. Default is False.

    Template Methods (override in subclasses)
    -----------------------------------------
    thinking() -> str
        Return the thinking prompt (how to decide next steps). Must be implemented.

    build_thinking_prompt(think_prompt, tools_description, output_instructions, context_info)
        Assemble the final prompts for the thinking phase.

    observation(context) -> Union[str, _DELEGATE]
        Provide custom observation. Default returns _DELEGATE (delegate to Agent).
        Override to return a string for worker-specific observation.

    verify_tools(matched_list, context) -> List[Tuple[ToolCall, ToolSpec]]
        Verify/adjust matched tools before execution. Called by agent.action().

    consequence(action_results) -> Any
        Process tool execution results. Called by agent.action().

    Examples
    --------
    >>> class ReactWorker(CognitiveWorker):
    ...     def __init__(self, llm):
    ...         super().__init__(llm, max_detail_rounds=2)
    ...
    ...     async def thinking(self):
    ...         return "Plan ONE immediate next step with appropriate tools."
    ...
    """

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        max_detail_rounds: int = 1,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
    ):
        super().__init__()
        self._llm = llm
        self.max_detail_rounds = max_detail_rounds

        # Logging runtime (None = inherit from AgentAutoma)
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Usage stats
        self.spend_tokens = 0
        self.spend_time = 0

        # Last decision from thinking phase (consumed by ThinkStepDescriptor)
        self._last_decision: Optional[ThinkDecision] = None

    def set_llm(self, llm: BaseLlm) -> None:
        """
        Set the LLM used for thinking and tool selection.

        Parameters
        ----------
        llm : BaseLlm
            LLM instance to use. Replaces any previously set LLM.
        """
        self._llm = llm

    ############################################################################
    # Worker methods (GraphAutoma execution flow)
    ############################################################################

    @worker(is_start=True)
    async def _thinking(self, context: CognitiveContext) -> Any:
        """
        Thinking phase: decide what to do next (thinking + tool selection in one call).

        Reads observation from context.observation (set by ThinkStepDescriptor before
        calling arun). Stores the decision in _last_decision for ThinkStepDescriptor
        to pick up and pass to agent.action().
        """
        if not isinstance(context, CognitiveContext):
            raise TypeError(
                f"Expected CognitiveContext, got {type(context).__name__}. "
                "CognitiveWorker requires CognitiveContext or its subclass."
            )
        if self._llm is None:
            raise RuntimeError(
                "CognitiveWorker has no LLM set. Either pass llm= in __init__ "
                "or use set_llm() before running."
            )

        observation = context.observation
        await self._run_thinking(observation, context)

    async def _run_thinking(self, observation: Optional[str], context: CognitiveContext):
        """Thinking phase with layered disclosure support."""
        self._log("Think", "Thinking with tool selection", color="blue")

        think_prompt = await self.thinking()
        think_result: Union[ThinkResult, ThinkDecision] = None

        for round_idx in range(self.max_detail_rounds + 1):
            is_last_round = (round_idx == self.max_detail_rounds)

            if is_last_round:
                # Last round: use Decision schema (no details_needed)
                system_prompt, user_prompt = await self._build_prompts(
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
                    constraint=PydanticModel(model=ThinkDecision)
                )
                self._log_prompt(f"Think round {round_idx + 1} (final)", system_prompt, user_prompt)
                break  # Always exit on last round
            else:
                # Non-last round: use ThinkResult schema (allows details_needed)
                system_prompt, user_prompt = await self._build_prompts(
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
                    constraint=PydanticModel(model=ThinkResult)
                )
                self._log_prompt(f"Think round {round_idx + 1}", system_prompt, user_prompt)

                # Check if LLM needs more details
                if think_result.details_needed:
                    # Filter out already-disclosed items
                    new_requests = self._filter_disclosed_requests(
                        think_result.details_needed, context
                    )
                    if new_requests:
                        reqs = [f"{r.field}[{r.index}]" for r in new_requests]
                        self._log("Think", f"Requesting details (round {round_idx + 1}): {reqs}", color="blue")
                        for req in new_requests:
                            context.get_details(req.field, req.index)
                        continue
                    else:
                        self._log("Think", f"All requested details already disclosed (round {round_idx + 1}), skipping", color="yellow")
                        continue  # Go to final round for forced decision

                # LLM gave a decision, exit loop
                break

        # Store decision for AgentAutoma to execute (Worker doesn't decide task completion)
        tools = [c.tool for c in think_result.calls]
        self._log("Think", f"Result: step=\"{think_result.step_content}\" tools={tools}, reasoning=\"{think_result.reasoning}\"", color="blue")
        self._last_decision = think_result

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

    def _filter_disclosed_requests(
        self,
        details_needed: List[DetailRequest],
        context: CognitiveContext
    ) -> List[DetailRequest]:
        """Filter out detail requests for items already disclosed in context."""
        try:
            disclosed = object.__getattribute__(context, '_disclosed_details')
        except AttributeError:
            return details_needed
        return [
            req for req in details_needed
            if not any(d[0] == req.field and d[1] == req.index for d in disclosed)
        ]

    def _append_disclosed_info(self, output_instructions: str, context: CognitiveContext) -> str:
        """Append already-disclosed items info to output instructions."""
        try:
            disclosed = object.__getattribute__(context, '_disclosed_details')
        except AttributeError:
            return output_instructions
        if disclosed:
            disclosed_items = [f"{field}[{idx}]" for field, idx, _ in disclosed]
            output_instructions += (
                "\n\n# Already Disclosed (DO NOT request again):\n"
                f"These items are already loaded: {', '.join(disclosed_items)}.\n"
                "Their content is in 'Previously Disclosed Details' below. "
                "Use them directly — do NOT add them to details_needed."
            )
        return output_instructions

    async def _build_prompts(
        self,
        think_prompt: str,
        context: CognitiveContext,
        observation: Optional[str] = None,
        is_final_round: bool = False
    ) -> Tuple[str, str]:
        """Build prompts for the thinking phase (thinking + tool selection in one call).

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

        # Build tool details directly from ToolSpec list
        capabilities_parts = []
        _, tool_specs = context.get_field('tools')

        def _format_tools_details(tool_specs: List[ToolSpec]) -> str:
            """Format tool specs into detailed description string."""
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
        # When is_final_round=True (i.e. max_detail_rounds=0), skip the details_needed hint
        # because the LLM cannot request details in this round.
        if len(context.skills) > 0:
            if is_final_round:
                lines = ["# Available Skills:"]
                for i, skill_summary in enumerate(context.skills.summary()):
                    lines.append(f"  [{i}] {skill_summary}")
                capabilities_parts.append("\n".join(lines))
            else:
                skills_summary = context.summary().get('skills')
                if skills_summary:
                    capabilities_parts.append(f"# {skills_summary}")

        capabilities_description = "\n\n".join(capabilities_parts)

        # Prepare output instructions based on whether this is the final round
        if is_final_round:
            # Final round: no details_needed, must make a decision
            output_instructions = (
                "# Output Format:\n"
                "- **step_content**: Description of what to do in this step\n"
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name: 'param_name', value: 'param_value'}]}]\n"
                "\n"
                "# Execution Guidelines:\n"
                "- Focus on ONE step at a time"
            )
        else:
            # Non-final round: include details_needed option
            output_instructions = (
                "# Output Format:\n"
                "- **step_content**: Description of what to do in this step (empty if requesting details)\n"
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name: 'param_name', value: 'param_value'}]}]\n"
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
                "- When requesting details, leave step_content and calls empty"
            )

            # Append already-disclosed info to prevent redundant detail requests
            output_instructions = self._append_disclosed_info(output_instructions, context)

        # Call template method to assemble final prompts
        system_prompt, user_prompt = await self.build_thinking_prompt(
            think_prompt=think_prompt.strip(),
            tools_description=capabilities_description,  # Now includes both tools and skills
            output_instructions=output_instructions,
            context_info=context_info
        )

        self.spend_tokens += self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        return system_prompt, user_prompt

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Rough approximation: ~4 chars per token (typical for English/UTF-8)."""
        return (len(text) + 3) // 4

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    async def observation(self, context: CognitiveContext) -> Any:
        """
        Provide custom observation before thinking.

        Default returns _DELEGATE, signaling that ThinkStepDescriptor should
        fall back to the agent's observation() method instead.

        Override to return a string for worker-specific observation that takes
        precedence over the agent-level default.

        Parameters
        ----------
        context : CognitiveContext
            The cognitive context.

        Returns
        -------
        Any
            _DELEGATE (default) to delegate to agent.observation().
            A string to use as the observation directly.

        Examples
        --------
        >>> async def observation(self, context):
        ...     # Worker-specific observation, overrides agent default
        ...     if len(context.cognitive_history) > 0:
        ...         latest = context.cognitive_history[-1]
        ...         return f"Last action: {latest.content}, Result: {latest.result}"
        ...     return _DELEGATE  # fall back to agent for everything else
        """
        return _DELEGATE

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
            Formatted description of available tools (with parameters).
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
