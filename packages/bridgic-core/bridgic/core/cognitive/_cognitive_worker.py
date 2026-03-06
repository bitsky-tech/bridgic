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
from typing import Annotated, Any, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, create_model
from pydantic.functional_validators import BeforeValidator

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


class ThinkDecision(BaseModel):
    """
    Decision model for the final round: no policy signals, must decide.

    Used on the last round of thinking to enforce decision output.
    No policy fields - this is the final, actionable decision.

    Note: Worker executes one thinking cycle without deciding task completion.
    Termination is controlled by Agent via until() conditions.

    Attributes
    ----------
    step_content : str
        Description of what to do in this step.
    calls : List[StepToolCall]
        Tool calls to execute for this step.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["step_content", "calls"],
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


def _coerce_none_to_list(v: Any) -> list:
    """Coerce None to empty list for field validation."""
    return [] if v is None else v


class _ThinkResultBase(BaseModel):
    """
    Internal base class for dynamically-generated ThinkResult models.

    Contains common fields (step_content, calls) with their validators.
    Policy-specific fields (details_needed, rehearsal, reflection) are added
    dynamically by _create_think_result_model() based on enabled policies.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["step_content", "calls"],
            "additionalProperties": False,
        }
    )

    step_content: str = Field(
        default="",
        description="Description of what to do in this step (can be empty if using policies)"
    )
    calls: List[StepToolCall] = Field(
        default_factory=list,
        description="Tool calls to execute for this step (can be empty if using policies)"
    )

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator('calls', mode='before')
    @classmethod
    def coerce_calls(cls, v: Any) -> list:
        return [] if v is None else v


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
    enable_rehearsal : bool, optional
        Enable rehearsal policy (predict tool execution outcomes). Default is False.
    enable_reflection : bool, optional
        Enable reflection policy (assess information quality). Default is False.
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

    observation(context, default_observation) -> Union[str, _DELEGATE]
        Enhance or customize observation. Default returns default_observation as-is.
        Return _DELEGATE for legacy behavior (delegate to Agent).

    before_action(matched_tools, context) -> List[Tuple[ToolCall, ToolSpec]]
        Verify/adjust matched tools before execution. Called by agent.action().

    after_action(action_results, context) -> Any
        Process tool execution results. Called by agent.action().

    Examples
    --------
    >>> class ReactWorker(CognitiveWorker):
    ...     def __init__(self, llm):
    ...         super().__init__(llm, enable_rehearsal=True)
    ...
    ...     async def thinking(self):
    ...         return "Plan ONE immediate next step with appropriate tools."
    ...
    """

    # Class-level cache: (enable_rehearsal, enable_reflection) → model class
    _model_cache: Dict[Tuple[bool, bool], Type[BaseModel]] = {}

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        enable_rehearsal: bool = False,
        enable_reflection: bool = False,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
    ):
        super().__init__()
        self._llm = llm

        self.enable_rehearsal = enable_rehearsal
        self.enable_reflection = enable_reflection

        # Policy round: one round before decision if any policy is enabled
        self._policy_rounds = 1 if (enable_rehearsal or enable_reflection) else 0

        # Logging runtime (None = inherit from AgentAutoma)
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Usage stats
        self.spend_tokens = 0
        self.spend_time = 0

        # Last decision from thinking phase (consumed by ThinkStepDescriptor)
        self._last_decision: Optional[Any] = None

        # Dynamic ThinkResult model based on enabled policies
        self._ThinkResultModel = self._create_think_result_model()

    def set_llm(self, llm: BaseLlm) -> None:
        """
        Set the LLM used for thinking and tool selection.

        Parameters
        ----------
        llm : BaseLlm
            LLM instance to use. Replaces any previously set LLM.
        """
        self._llm = llm

    def _create_think_result_model(self) -> Type[BaseModel]:
        """
        Dynamically create ThinkResult model based on enabled policies.

        Returns a cached model class matching the current policy configuration.
        Always includes details_needed (disclosure is built-in). Policy-specific
        fields (rehearsal, reflection) are added based on enabled policies.
        """
        cache_key = (
            self.enable_rehearsal,
            self.enable_reflection,
        )
        if cache_key in CognitiveWorker._model_cache:
            return CognitiveWorker._model_cache[cache_key]

        extra_fields: Dict[str, Any] = {}

        # details_needed is always included (disclosure is built-in behavior)
        extra_fields['details_needed'] = (
            Annotated[List[DetailRequest], BeforeValidator(_coerce_none_to_list)],
            Field(
                default_factory=list,
                description="Request details before deciding. Example: [{field: 'cognitive_history', index: 0}]"
            )
        )

        if self.enable_rehearsal:
            extra_fields['rehearsal'] = (
                Optional[str],
                Field(
                    default=None,
                    description="Rehearsal: predict what will happen if you execute the planned tools. What results will they return? Any potential issues?"
                )
            )

        if self.enable_reflection:
            extra_fields['reflection'] = (
                Optional[str],
                Field(
                    default=None,
                    description="Reflection: assess information quality. Is the information sufficient? Any contradictions or gaps?"
                )
            )

        model = create_model('ThinkResult', __base__=_ThinkResultBase, **extra_fields)
        CognitiveWorker._model_cache[cache_key] = model
        return model

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
        """Thinking phase with cognitive policies support."""
        self._log("Think", "Thinking with tool selection", color="blue")

        think_prompt = await self.thinking()
        think_result = None

        # Policy accumulation context (passed to subsequent rounds)
        policy_context = []

        # After one disclosure batch, force decision on next round
        disclosure_done = False

        # Policy phase: exactly one round if any policy is enabled, then move on
        policy_phase_done = (self._policy_rounds == 0)

        round_num = 0

        while True:
            round_num += 1
            is_policy_round = not policy_phase_done

            # After disclosure, use ThinkDecision (no details_needed) to hard-block further requests
            model = ThinkDecision if disclosure_done else self._ThinkResultModel

            system_prompt, user_prompt = await self._build_prompts(
                think_prompt=think_prompt,
                context=context,
                observation=observation,
                policy_context=policy_context,
                is_policy_round=is_policy_round,
                disclosure_done=disclosure_done,
            )
            think_result = await self._llm.astructured_output(
                messages=[
                    Message.from_text(text=system_prompt, role="system"),
                    Message.from_text(text=user_prompt, role="user")
                ],
                constraint=PydanticModel(model=model)
            )
            self._log_prompt(
                f"Think round {round_num}" + (" (policy)" if is_policy_round else ""),
                system_prompt, user_prompt
            )

            # Process policy round
            if is_policy_round:
                if self.enable_rehearsal and hasattr(think_result, 'rehearsal') and think_result.rehearsal:
                    policy_context.append(f"Rehearsal: {think_result.rehearsal}")
                    self._log("Think", f"Rehearsal output: {think_result.rehearsal}", color="blue")

                if self.enable_reflection and hasattr(think_result, 'reflection') and think_result.reflection:
                    policy_context.append(f"Reflection: {think_result.reflection}")
                    self._log("Think", f"Reflection output: {think_result.reflection}", color="blue")

                policy_phase_done = True
                continue

            # Check for disclosure requests (single batch: all requests processed at once)
            if not disclosure_done and hasattr(think_result, 'details_needed') and think_result.details_needed:
                reqs = think_result.details_needed
                req_labels = [f"{r.field}[{r.index}]" for r in reqs]
                self._log("Think", f"Requesting details (batch): {req_labels}", color="blue")
                for req in reqs:
                    context.get_details(req.field, req.index)
                disclosure_done = True  # Force decision on next round
                continue

            # LLM gave a decision (no disclosure requests)
            break

        # Store decision
        tools = [c.tool for c in think_result.calls]
        self._log("Think", f"Result: step=\"{think_result.step_content}\" tools={tools}", color="blue")
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

    def _build_output_instructions(
        self,
        is_policy_round: bool,
        disclosure_done: bool,
        context: CognitiveContext,
    ) -> str:
        """
        Build output instructions dynamically based on the current round type.

        Mirrors _create_think_result_model: the same flags that add model fields
        also add the corresponding prompt guidance, keeping the two in sync.

          details_needed   → always present (disclosure is built-in)
          enable_rehearsal → rehearsal field + guidance (policy round only)
          enable_reflection→ reflection field + guidance (policy round only)

        disclosure_done=True means a batch disclosure has already been done;
        we use ThinkDecision (no details_needed) and instruct the LLM to decide.
        """
        # --- Field lines ---
        if is_policy_round:
            role_note = (
                "This is a policy analysis round — analyze the situation using "
                "the policy fields below. Do NOT output step_content or calls yet.\n"
            )
            field_lines = [
                role_note,
                "- **step_content**: Leave empty in policy round",
                "- **calls**: Leave empty in policy round",
            ]
            if self.enable_rehearsal:
                field_lines.append(
                    "- **rehearsal**: Predict what will happen if you execute the tools. "
                    "What results will they return? Any potential issues or edge cases?"
                )
            if self.enable_reflection:
                field_lines.append(
                    "- **reflection**: Reflect on the information quality. "
                    "Is it sufficient? Any contradictions or gaps?"
                )
        elif disclosure_done:
            # Post-disclosure: ThinkDecision model is used, no details_needed field
            field_lines = [
                "- **step_content**: Description of what to do in this step",
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name: 'param_name', value: 'param_value'}]}]",
            ]
        else:
            # Normal round: details_needed available for batch requests
            field_lines = [
                "- **step_content**: Description of what to do in this step (empty if requesting details)",
                "- **calls**: Tool calls as [{tool, tool_arguments: [{name: 'param_name', value: 'param_value'}]}]",
                "- **details_needed**: Request details via [{field: 'xxx', index: N}]",
            ]

        output_instructions = "# Output Format:\n" + "\n".join(field_lines)

        # --- Guidance section ---
        if is_policy_round:
            policy_bullets = []
            if self.enable_rehearsal:
                policy_bullets.append("- Rehearse: mentally simulate tool execution")
            if self.enable_reflection:
                policy_bullets.append("- Reflect: assess information quality")
            output_instructions += (
                "\n\n# Policy Round:\n"
                "This is a policy round. Do NOT make decisions yet.\n"
                "Use the policy fields to think through the situation:\n"
                + "\n".join(policy_bullets)
            )
        elif disclosure_done:
            output_instructions += (
                "\n\n# Execution Guidelines:\n"
                "- Focus on ONE step at a time\n"
                "- Make a concrete decision now"
            )
        else:
            output_instructions += (
                "\n\n# BEFORE Taking Action - Check If You Need Details:\n"
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
                "- You can request multiple items at once in a single details_needed batch"
            )

        return output_instructions

    async def _build_prompts(
        self,
        think_prompt: str,
        context: CognitiveContext,
        observation: Optional[str] = None,
        policy_context: List[str] = None,
        is_policy_round: bool = False,
        disclosure_done: bool = False,
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
        policy_context : List[str]
            Policy outputs from previous rounds (rehearsal, reflection).
        is_policy_round : bool
            If True, output instructions will include policy fields (rehearsal, reflection).
        disclosure_done : bool
            If True, a batch disclosure has already been done; use ThinkDecision model
            and remove details_needed guidance to force a final decision.
        """
        # Build context info (exclude tools and skills, they go in system prompt)
        context_info = context.format_summary(exclude=['tools', 'skills'])
        if observation:
            context_info += f"\n\nObservation:\n{observation}"

        # Add policy context (from previous round's rehearsal/reflection)
        if policy_context:
            context_info += "\n\n# Policy Context (from previous round):\n"
            context_info += "\n".join(policy_context)

        # Append already-disclosed items to user prompt (not output_instructions)
        try:
            disclosed = object.__getattribute__(context, '_disclosed_details')
            if disclosed:
                disclosed_items = [f"{field}[{idx}]" for field, idx, _ in disclosed]
                context_info += (
                    "\n\n# Already Disclosed (DO NOT request again):\n"
                    f"These items are already loaded: {', '.join(disclosed_items)}.\n"
                    "Their content is in 'Previously Disclosed Details' below. "
                    "Use them directly — do NOT add them to details_needed."
                )
        except AttributeError:
            pass

        # Task restatement at the start of user prompt
        user_prompt_context = f"Based on the context below, decide your next action.\n\n{context_info}"

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

        # Add skills to system prompt (always show summary since details_needed is always available).
        if len(context.skills) > 0:
            skills_summary = context.summary().get('skills')
            if skills_summary:
                capabilities_parts.append(f"# {skills_summary}")

        capabilities_description = "\n\n".join(capabilities_parts)

        # In policy round: override think_prompt with a role-limiting prefix
        effective_think_prompt = think_prompt
        if is_policy_round:
            effective_think_prompt = (
                f"Before deciding, use the policy fields to analyze the situation.\n\n{think_prompt}"
            )

        # Build output instructions dynamically based on round type
        output_instructions = self._build_output_instructions(
            is_policy_round=is_policy_round,
            disclosure_done=disclosure_done,
            context=context,
        )

        # Call template method to assemble final prompts
        system_prompt, user_prompt = await self.build_thinking_prompt(
            think_prompt=effective_think_prompt.strip(),
            tools_description=capabilities_description,  # Now includes both tools and skills
            output_instructions=output_instructions,
            context_info=user_prompt_context
        )

        self.spend_tokens += self._count_tokens(system_prompt) + self._count_tokens(user_prompt)
        return system_prompt, user_prompt

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Rough approximation: ~4 chars per token (typical for English/UTF-8)."""
        return (len(text) + 3) // 4

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    async def observation(
        self,
        context: CognitiveContext,
        default_observation: Optional[str] = None
    ) -> Any:
        """
        Enhance or customize the observation before thinking.

        Parameters
        ----------
        context : CognitiveContext
            The cognitive context.
        default_observation : Optional[str]
            The default observation from Agent. If None, this is a legacy call
            (for backward compatibility).

        Returns
        -------
        Any
            _DELEGATE (legacy mode) to delegate to agent.observation().
            A string to use as the observation (can be enhanced from default).

        Examples
        --------
        >>> async def observation(self, context, default_observation=None):
        ...     # Legacy mode: delegate
        ...     if default_observation is None:
        ...         return _DELEGATE
        ...
        ...     # Enhancement mode: add context
        ...     if len(context.cognitive_history) > 0:
        ...         latest = context.cognitive_history[-1]
        ...         return f"{default_observation}\\n\\nLast step: {latest.content}"
        ...     return default_observation
        """
        # Backward compatibility: if no default_observation, return _DELEGATE
        if default_observation is None:
            return _DELEGATE

        # New semantics: default is to return as-is (no modification)
        return default_observation

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

    async def before_action(
        self,
        matched_tools: List[Tuple[ToolCall, ToolSpec]],
        context: CognitiveContext
    ) -> List[Tuple[ToolCall, ToolSpec]]:
        """
        Verify and optionally adjust matched tools before execution.

        Called after tool matching, before actual execution. Override to
        validate, filter, or modify the tools that will be executed.

        Default: return as-is (no verification).

        Parameters
        ----------
        matched_tools : List[Tuple[ToolCall, ToolSpec]]
            Pairs of (tool_call, tool_spec) ready for execution.
        context : CognitiveContext
            Current cognitive context.

        Returns
        -------
        List[Tuple[ToolCall, ToolSpec]]
            Verified/adjusted list of tools to execute.

        Examples
        --------
        >>> async def before_action(self, matched_tools, context):
        ...     # Filter out dangerous tools
        ...     return [(tc, spec) for tc, spec in matched_tools
        ...             if spec.tool_name not in ["delete", "drop"]]
        """
        return matched_tools

    async def after_action(
        self,
        action_results: List[ActionStepResult],
        context: CognitiveContext
    ) -> Any:
        """
        Process tool execution results (format or summarize).

        Called after all tools have been executed. Override to transform
        the raw results into a more useful format.

        Default: return raw list.

        Parameters
        ----------
        action_results : List[ActionStepResult]
            Per-tool results from the action phase.
        context : CognitiveContext
            Current cognitive context.

        Returns
        -------
        Any
            Value stored as the step result in cognitive history.

        Examples
        --------
        >>> async def after_action(self, action_results, context):
        ...     # Format results as summary
        ...     return {"results": [r.tool_result for r in action_results]}
        """
        return action_results

    # Backward compatibility aliases
    async def verify_tools(self, matched_list: List[Tuple[ToolCall, ToolSpec]], context: CognitiveContext) -> List[Tuple[ToolCall, ToolSpec]]:
        """Deprecated: Use before_action() instead."""
        return await self.before_action(matched_list, context)

    async def consequence(self, action_results: List[ActionStepResult]) -> Any:
        """Deprecated: Use after_action() instead."""
        # Old signature didn't have context parameter
        # Try to call new method if subclass overrode it
        try:
            # Check if subclass overrode after_action
            if type(self).after_action is not CognitiveWorker.after_action:
                # Subclass overrode it, but we don't have context here
                # Just return raw results (best effort)
                return action_results
        except:
            pass
        return action_results

    ############################################################################
    # Entry point
    ############################################################################

    @classmethod
    def from_prompt(
        cls,
        thinking_prompt: str,
        llm: Optional[BaseLlm] = None,
        enable_rehearsal: bool = False,
        enable_reflection: bool = False,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
    ) -> "CognitiveWorker":
        """
        Create a simple CognitiveWorker from a thinking prompt string.

        Convenience factory for cases where you only need to customize the
        thinking prompt without overriding other methods.

        Parameters
        ----------
        thinking_prompt : str
            The thinking prompt to use.
        llm : Optional[BaseLlm]
            LLM instance to use.
        enable_rehearsal : bool
            Enable rehearsal policy.
        enable_reflection : bool
            Enable reflection policy.
        verbose : Optional[bool]
            Enable verbose logging.
        verbose_prompt : Optional[bool]
            Enable prompt logging.

        Returns
        -------
        CognitiveWorker
            A worker instance with the specified thinking prompt.

        Examples
        --------
        >>> worker = CognitiveWorker.from_prompt(
        ...     "Plan ONE immediate next step",
        ...     llm=llm,
        ...     enable_rehearsal=True
        ... )
        """
        class _PromptWorker(cls):
            async def thinking(self):
                return thinking_prompt

        return _PromptWorker(
            llm=llm,
            enable_rehearsal=enable_rehearsal,
            enable_reflection=enable_reflection,
            verbose=verbose,
            verbose_prompt=verbose_prompt,
        )

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
