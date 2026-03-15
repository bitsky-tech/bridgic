"""
CognitiveWorker - Core thinking unit of the cognitive architecture.

A CognitiveWorker represents the "think" phase of one observe-think-act cycle.
Observation (before) and action (after) are orchestrated by AgentAutoma.

Design:
1. Think-only unit: Each arun() call performs exactly the thinking phase.
   Observation is injected via context.observation (set by AgentAutoma._run_once
   before calling arun). Action is executed by AgentAutoma.action() after arun
   returns.

2. Multi-round thinking loop: The thinking phase calls the LLM at least once.
   Cognitive policies (acquiring, rehearsal, reflection) may trigger additional
   rounds within the same arun() call. Each policy fires at most once per call,
   then permanently closes for that round.

3. Works directly with CognitiveContext:
   - CognitiveWorker extends GraphAutoma (single-node graph, _thinking as start+output)
   - CognitiveContext extends Context (Exposure-based field management)
"""

import time
import json
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional, Tuple, Type, Union

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
if TYPE_CHECKING:
    from bridgic.core.cognitive._agent_automa import ActionStepResult


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


class _ThinkBase(BaseModel):
    """
    Unified base for all dynamically-generated ThinkModel variants.

    Factory (_create_think_model) adds: output, details, rehearsal,
    reflection — all optional and conditional on configuration.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["step_content"],
            "additionalProperties": False,
        }
    )

    step_content: str = Field(
        default="",
        description="Description of what to do in this step, or your analysis/reasoning"
    )
    finish: bool = Field(
        default=False,
        description="Set True when your current sub-task is FULLY complete and no more steps are needed."
    )

    @field_validator('step_content', mode='before')
    @classmethod
    def coerce_step_content(cls, v: Any) -> str:
        return "" if v is None else str(v)


def _coerce_none_to_list(v: Any) -> list:
    """Coerce None to empty list for field validation."""
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

    Class Attributes
    ----------------
    output_schema : Optional[Type[BaseModel]]
        If set, the worker produces a typed Pydantic instance directly using the
        output_schema as the LLM constraint. The agent's action() phase is skipped
        entirely. ``await step`` returns the typed instance.
        Policy rounds (rehearsal/reflection) still run if enabled.

    Template Methods (override in subclasses)
    -----------------------------------------
    thinking() -> str
        Return the thinking prompt (how to decide next steps). Must be implemented.

    build_messages(think_prompt, tools_description, output_instructions, context_info)
        Assemble the final messages for the thinking phase. Returns List[Message].

    observation(context, default_observation) -> Union[str, _DELEGATE]
        Enhance or customize observation. Default returns default_observation as-is.
        Return _DELEGATE for legacy behavior (delegate to Agent).

    before_action(matched_tools, context) -> List[Tuple[ToolCall, ToolSpec]]
        Verify/adjust matched tools before execution. Called by agent.action().

    Examples
    --------
    >>> class ReactWorker(CognitiveWorker):
    ...     def __init__(self, llm):
    ...         super().__init__(llm, enable_rehearsal=True)
    ...
    ...     async def thinking(self):
    ...         return "Plan ONE immediate next step with appropriate tools."
    ...
    >>> class PlannerWorker(CognitiveWorker):
    ...     output_schema = PlanningResult  # skip tool loop, return typed instance
    ...
    ...     async def thinking(self):
    ...         return "Analyze the goal and produce a phased execution plan."
    """

    # Class-level cache: (enable_rehearsal, enable_reflection, enable_acquiring, output_schema) → model
    _think_model_cache: Dict[Tuple, Type[BaseModel]] = {}

    # Subclasses set this to a Pydantic model to produce typed output directly
    output_schema: Optional[Type[BaseModel]] = None

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        enable_rehearsal: bool = False,
        enable_reflection: bool = False,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        super().__init__()
        self._llm = llm

        # Policy flags
        self.enable_rehearsal = enable_rehearsal
        self.enable_reflection = enable_reflection

        # Instance-level output_schema overrides the class attribute when provided
        if output_schema is not None:
            self.output_schema = output_schema

        # Logging runtime (None = inherit from AgentAutoma)
        self._verbose = verbose
        self._verbose_prompt = verbose_prompt

        # Usage stats
        self.spend_tokens = 0
        self.spend_time = 0

    def set_llm(self, llm: BaseLlm) -> None:
        """
        Set the LLM used for thinking and tool selection.

        Parameters
        ----------
        llm : BaseLlm
            LLM instance to use. Replaces any previously set LLM.
        """
        self._llm = llm

    @classmethod
    def _create_think_model(
        cls,
        enable_rehearsal: bool = False,
        enable_reflection: bool = False,
        enable_acquiring: bool = True,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Type[BaseModel]:
        """
        Unified factory for all ThinkModel variants.

        Builds and caches a Pydantic model with _ThinkBase as its base, adding
        fields dynamically:
        - output: List[StepToolCall] when output_schema is None, else Optional[output_schema]
        - details: only when enable_acquiring=True
        - rehearsal: only when enable_rehearsal=True
        - reflection: only when enable_reflection=True

        All variants are cached by (enable_rehearsal, enable_reflection,
        enable_acquiring, output_schema).
        """
        key = (enable_rehearsal, enable_reflection, enable_acquiring, output_schema)
        if key in cls._think_model_cache:
            return cls._think_model_cache[key]

        extra_fields: Dict[str, Any] = {}

        # output field — type depends on output_schema
        if output_schema is None:
            extra_fields['output'] = (
                Annotated[List[StepToolCall], BeforeValidator(_coerce_none_to_list)],
                Field(
                    default_factory=list,
                    description="Tool calls to execute as [{tool, tool_arguments: [{name: 'param_name', value: 'param_value'}]}]"
                )
            )
        else:
            extra_fields['output'] = (
                Optional[output_schema],
                Field(default=None, description="Structured result matching the required output schema.")
            )

        # Information that can be requested for progressive disclosure to assist in decision-making.
        if enable_acquiring:
            extra_fields['details'] = (
                Annotated[List[DetailRequest], BeforeValidator(_coerce_none_to_list)],
                Field(
                    default_factory=list,
                    description="Request details before deciding. Example: [{field: 'cognitive_history', index: 0}]"
                )
            )

        # Can predict what will happen after the operation and assist in decision-making.
        if enable_rehearsal:
            extra_fields['rehearsal'] = (
                Optional[str],
                Field(
                    default=None,
                    description="(Optional) Predict what will happen if you execute the planned tools. What results will they return? Any potential issues?"
                )
            )

        # Can reflect on the current situation to assist in decision-making.
        if enable_reflection:
            extra_fields['reflection'] = (
                Optional[str],
                Field(
                    default=None,
                    description="(Optional) Assess information quality. Is it sufficient? Any contradictions or gaps?"
                )
            )

        model = create_model('ThinkModel', __base__=_ThinkBase, **extra_fields)
        cls._think_model_cache[key] = model
        return model

    @property
    def _ThinkResultModel(self) -> Type[BaseModel]:
        """Backward-compat: model for normal rounds (acquiring open, all enabled policies)."""
        return self._create_think_model(
            enable_rehearsal=self.enable_rehearsal,
            enable_reflection=self.enable_reflection,
            enable_acquiring=True,
            output_schema=None,
        )

    @property
    def _ThinkDecisionModel(self) -> Type[BaseModel]:
        """Backward-compat: model for forced-decision rounds (all operators closed)."""
        return self._create_think_model(
            enable_rehearsal=False,
            enable_reflection=False,
            enable_acquiring=False,
            output_schema=self.output_schema,
        )


    ############################################################################
    # Worker methods (GraphAutoma execution flow)
    ############################################################################

    @worker(is_start=True, is_output=True)
    async def _thinking(self, context: CognitiveContext) -> Any:
        """
        Thinking phase: decide what to do next (thinking + tool selection in one call).

        Reads observation from context.observation (set by AgentAutoma.run() before
        calling arun). Returns the decision directly; AgentAutoma.run() reads the
        arun() return value (no side-channel via _last_decision).
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
        return await self._run_thinking(observation, context)

    async def _run_thinking(self, observation: Optional[str], context: CognitiveContext) -> Any:
        """Thinking phase with cognitive policies support. Returns the final decision."""

        ###############################
        # Call the LLM to get the decision
        ###############################

        # Get the custom thinking prompt from the worker's thinking() method.
        think_prompt = await self.thinking()

        # Accumulated policy outputs injected into subsequent rounds' prompts
        policy_context = []

        # Per-operator open flags — each fires at most once then closes
        # acquiring is a built-in framework capability, disabled when output_schema is set
        acquiring_open = (self.output_schema is None)
        rehearsal_open = self.enable_rehearsal
        reflection_open = self.enable_reflection

        while True:
            # Build messages
            messages = await self._build_messages(
                think_prompt=think_prompt,
                context=context,
                observation=observation,
                policy_context=policy_context,
                acquiring_open=acquiring_open,
                rehearsal_open=rehearsal_open,
                reflection_open=reflection_open,
            )
            # Build model from current per-operator state
            model = self._create_think_model(
                enable_rehearsal=rehearsal_open,
                enable_reflection=reflection_open,
                enable_acquiring=acquiring_open,
                output_schema=self.output_schema,
            )
            # LLM call
            think_result = await self._llm.astructured_output(
                messages=messages,
                constraint=PydanticModel(model=model)
            )
            self._log_prompt("Think", messages)

            ###############################
            # Analyze the LLM response and decide what to do next
            ###############################

            re_think = False  # Whether any operator was activated this round

            # Operator 1: acquiring — fetch external details (one-shot, only when non-empty)
            if acquiring_open:
                reqs = getattr(think_result, 'details', None) or []
                if reqs:
                    for req in reqs:
                        context.get_details(req.field, req.index)
                    acquiring_open = False  # close — never fires again
                    re_think = True

            # Operator 2: rehearsal — inject prediction into next round (one-shot, only when filled)
            if rehearsal_open:
                rehearsal_val = getattr(think_result, 'rehearsal', None)
                if rehearsal_val is not None:
                    policy_context.append(f"## Mental Simulation (Rehearsal):\n{rehearsal_val}")
                    rehearsal_open = False  # close
                    re_think = True

            # Operator 3: reflection — inject quality assessment into next round (one-shot, only when filled)
            if reflection_open:
                reflection_val = getattr(think_result, 'reflection', None)
                if reflection_val is not None:
                    policy_context.append(f"## Information Assessment (Reflection):\n{reflection_val}")
                    reflection_open = False  # close
                    re_think = True

            if re_think:
                continue  # Re-think with narrowed model + accumulated context

            break  # No operator activated — LLM gave a direct decision

        return think_result

    ############################################################################
    # Internal helpers
    ############################################################################

    def _log_prompt(self, stage: str, messages: List[Message]):
        """Log prompts with timestamp and caller location if verbose_prompt is enabled."""
        if not self._verbose_prompt:
            return
        import inspect
        from datetime import datetime
        from os.path import basename

        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                filename = basename(caller.f_code.co_filename)
                lineno = caller.f_lineno
            else:
                filename, lineno = "?", 0
        finally:
            del frame

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        total_tokens = sum(self._count_tokens(m.content) for m in messages)
        for i, msg in enumerate(messages):
            tokens = self._count_tokens(msg.content)
            printer.print(f"[{ts}] [{stage}] ({filename}:{lineno}) Message {i+1} ({msg.role}, {tokens} tokens):", color="cyan")
            printer.print(msg.content, color="gray")
        printer.print(f"[{ts}] [{stage}] ({filename}:{lineno}) Total: {total_tokens} tokens (cumulative: {self.spend_tokens})", color="yellow")

    @staticmethod
    def _generate_schema_example(schema: dict, defs: dict = None) -> Any:
        """
        Recursively build a compact example value from a JSON Schema node.

        Handles $ref, anyOf (Optional), object, array, scalar types, enum, and const.
        Arrays are represented as a single-element list to keep examples concise.
        """
        if defs is None:
            defs = schema.get('$defs', {})

        # Resolve $ref
        if '$ref' in schema:
            ref_name = schema['$ref'].split('/')[-1]
            return CognitiveWorker._generate_schema_example(defs.get(ref_name, {}), defs)

        # anyOf / oneOf — covers Optional[X] (anyOf: [{type: X}, {type: null}])
        if 'anyOf' in schema:
            non_null = [s for s in schema['anyOf'] if s.get('type') != 'null']
            if non_null:
                return CognitiveWorker._generate_schema_example(non_null[0], defs)
            return None

        # allOf — usually single-item wrapper
        if 'allOf' in schema:
            if len(schema['allOf']) == 1:
                return CognitiveWorker._generate_schema_example(schema['allOf'][0], defs)
            return {}

        # Enum — use first value
        if 'enum' in schema:
            return schema['enum'][0]

        # Const
        if 'const' in schema:
            return schema['const']

        schema_type = schema.get('type')

        if schema_type == 'object':
            props = schema.get('properties', {})
            result = {}
            for name, prop_schema in props.items():
                if 'default' in prop_schema:
                    result[name] = prop_schema['default']
                else:
                    result[name] = CognitiveWorker._generate_schema_example(prop_schema, defs)
            return result

        if schema_type == 'array':
            items = schema.get('items', {})
            item_example = CognitiveWorker._generate_schema_example(items, defs)
            return [item_example]

        if schema_type == 'string':
            return "..."
        if schema_type == 'integer':
            return schema.get('default', 0)
        if schema_type == 'number':
            return schema.get('default', 0.0)
        if schema_type == 'boolean':
            return schema.get('default', False)

        return None

    @staticmethod
    def _build_schema_example_prompt(model: Type[BaseModel]) -> str:
        """
        Build a compact inline example from a Pydantic model.

        Replaces the old full JSON Schema dump with a concise representative example,
        which is more token-efficient and equally effective for guiding the LLM.

        Example output for ``PlanningResult``:
            {"phases": [{"sub_goal": "...", "skill_name": "...", "max_steps": 20}]}
        """
        schema = model.model_json_schema()
        example = CognitiveWorker._generate_schema_example(schema)
        return json.dumps(example, ensure_ascii=False)

    def _output_fields_prompt(self, acquiring_open: bool, rehearsal_open: bool, reflection_open: bool) -> str:
        """Base fields: step_content, output, finish. Always present."""
        parts = []

        # Base fields
        parts.append(
            "# Output Fields\n"
            "- **step_content**: Your analysis and reasoning for this step\n"
            "- **finish**: Set True when the sub-task is fully complete (default: False)"
        )

        # Cognitive operators
        if acquiring_open:
            parts.append(
                "- **details**: Available fields: **skills**, **cognitive_history**. "
                "example: [{field: 'skills', index: 0}, ...]"
            )
        if rehearsal_open:
            parts.append("- **rehearsal**: string describing your simulation and predictions")
        if reflection_open:
            parts.append("- **reflection**: string describing your assessment conclusions")

        # Output schema
        if self.output_schema is not None:
            parts.append(
                "- **output**: Structured result — example: "
                f"{self._build_schema_example_prompt(self.output_schema)}"
            )
        else:
            parts.append(
                "- **output**: Tool calls to execute: "
                "[{tool, tool_arguments: [{name: 'param', value: 'value'}]}]\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _acquiring_prompt() -> str:
        """Acquiring operator: when/why to use + details field format."""
        return (
            "# Context Acquiring\n"
            "If the context contains progressively disclosed information (e.g. skills, history steps) "
            "and you want to inspect the details, use the **details** field to request them. "
            "The framework will expand these items in the next round. "
            "Batch all requests in a single output. "
            "When using this field, leave step_content and output empty.\n\n"
            "## Field format:\n"
            "- **details**: [{field: \"skills\", index: 0}, ...]\n"
            "  Available fields: **skills** (view a skill's full workflow), "
            "**cognitive_history** (view the full result of a previous step)"
        )

    @staticmethod
    def _rehearsal_prompt() -> str:
        """Rehearsal operator: when/why to use + rehearsal field format."""
        return (
            "# Pre-Action Rehearsal (optional)\n"
            "To ensure the next action is accurate, you may mentally simulate it first: "
            "which tools you plan to call, what they are expected to return, and whether any issues may arise. "
            "When using this field, leave step_content and output empty; "
            "the framework will ask for an actual decision in the next round.\n\n"
            "## Field format:\n"
            "- **rehearsal**: \"(string describing your simulation and predictions)\""
        )

    @staticmethod
    def _reflection_prompt() -> str:
        """Reflection operator: when/why to use + reflection field format."""
        return (
            "# Pre-Action Reflection (optional)\n"
            "Before committing to a decision, evaluate whether the current information is "
            "sufficient and self-consistent. "
            "If you have doubts, fill the **reflection** field and leave step_content and output empty; "
            "the framework will ask for an actual decision in the next round.\n\n"
            "## Field format:\n"
            "- **reflection**: \"(string describing your assessment conclusions)\""
        )

    def _build_output_instructions(
        self,
        acquiring_open: bool,
        rehearsal_open: bool,
        reflection_open: bool,
    ) -> str:
        parts = []
        if acquiring_open:
            parts.append(self._acquiring_prompt())
        if rehearsal_open:
            parts.append(self._rehearsal_prompt())
        if reflection_open:
            parts.append(self._reflection_prompt())
        parts.append(self._output_fields_prompt(acquiring_open, rehearsal_open, reflection_open))
        return "\n\n".join(parts)

    async def _build_messages(
        self,
        think_prompt: str,
        context: CognitiveContext,
        observation: Optional[str] = None,
        policy_context: List[str] = None,
        acquiring_open: bool = True,
        rehearsal_open: bool = False,
        reflection_open: bool = False,
    ) -> List[Message]:
        """Build messages for the thinking phase (thinking + tool selection in one call).

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
        acquiring_open : bool
            Whether the acquiring operator can still fire this round.
        rehearsal_open : bool
            Whether the rehearsal operator can still fire this round.
        reflection_open : bool
            Whether the reflection operator can still fire this round.
        """
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

        # 1. Build tool details and skills summary
        capabilities_parts = []
        _, tool_specs = context.get_field('tools')
        if tool_specs:
            tools_details = _format_tools_details(tool_specs)
            capabilities_parts.append(f"# Available Tools (with parameters):\n{tools_details}")
        if len(context.skills) > 0:
            skills_summary = context.summary().get('skills')
            if skills_summary:
                capabilities_parts.append(f"# {skills_summary}")
        capabilities_description = "\n\n".join(capabilities_parts)

        # 2. Build context info about current status
        context_info = context.format_summary(exclude=['tools', 'skills'])
        if observation is not None:
            context_info += f"\n\nObservation:\n{observation}"
        user_prompt_context = f"Based on the context below, decide your next action.\n\n{context_info}"

        # Inject phase-level goal if set by sequential()/loop()
        phase_goal = getattr(context, '_phase_goal', None)
        if phase_goal:
            user_prompt_context += f"\n\n## Current Phase Goal\n{phase_goal}\nWhen this phase goal is achieved, set finish=True."

        # 3. Build output instructions dynamically based on per-operator state
        output_instructions = self._build_output_instructions(
            acquiring_open=acquiring_open,
            rehearsal_open=rehearsal_open,
            reflection_open=reflection_open,
        )

        # Call template method to assemble final messages
        messages = await self.build_messages(
            think_prompt=think_prompt.strip(),
            tools_description=capabilities_description,
            output_instructions=output_instructions,
            context_info=user_prompt_context
        )

        self.spend_tokens += sum(self._count_tokens(m.content) for m in messages)
        return messages

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Rough approximation: ~4 chars per token (typical for English/UTF-8)."""
        return (len(text) + 3) // 4

    ############################################################################
    # Template methods (override by user to customize the behavior)
    ############################################################################

    async def observation(self, context: CognitiveContext) -> Any:
        """
        Enhance or customize the observation before thinking.

        Parameters
        ----------
        context : CognitiveContext
            The cognitive context.

        Returns
        -------
        Any
            _DELEGATE (legacy mode) to delegate to agent.observation().
            A string to use as the observation (can be enhanced from default).

        Examples
        --------
        >>> async def observation(self, context):
        ...     return _DELEGATE
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

    async def build_messages(
        self,
        think_prompt: str,
        tools_description: str,
        output_instructions: str,
        context_info: str,
    ) -> List[Message]:
        """
        Assemble the final messages for the thinking phase.

        Override this method to customize how the prompt components are structured
        across messages. This allows you to reorder, modify, or add to the message list.

        Parameters
        ----------
        think_prompt : str
            The thinking prompt from the thinking() method.
        tools_description : str
            Formatted description of available tools and skills.
        output_instructions : str
            Instructions for the output format (finish, steps/step_content, etc.).
        context_info : str
            Context information including goal, status, history, and fetched details.

        Returns
        -------
        List[Message]
            Messages to be sent to the LLM. Default structure:
            - Message 1 (system): think_prompt + tools_description (if non-empty) + output_instructions
            - Message 2 (user): context_info

        Examples
        --------
        >>> async def build_messages(self, think_prompt, tools_description,
        ...                          output_instructions, context_info):
        ...     # Custom: merge everything into a single system + user pair
        ...     extra = "EXTRA_INSTRUCTION: Always prefer cheapest option."
        ...     system = f"{think_prompt}\\n\\n{extra}\\n\\n{tools_description}\\n\\n{output_instructions}"
        ...     return [
        ...         Message.from_text(text=system, role="system"),
        ...         Message.from_text(text=context_info, role="user"),
        ...     ]
        """
        parts = [think_prompt]
        if tools_description:
            parts.append(tools_description)
        parts.append(output_instructions)
        system_content = "\n\n".join(parts)

        return [
            Message.from_text(text=system_content, role="system"),
            Message.from_text(text=context_info, role="user"),
        ]

    async def before_action(
        self,
        decision_result: Any,
        context: CognitiveContext
    ) -> Any:
        """
        Verify and optionally adjust the output before execution.

        Parameters
        ----------
        decision_result : Any
            The result of the decision.
        context : CognitiveContext
            Current cognitive context.

        Returns
        -------
        Any
            Verified/adjusted decision result.

        Examples
        --------
        >>> async def before_action(self, decision_result, context):
        ...     # Filter out dangerous tools
        ...     return decision_result.filter(lambda x: x.tool_name not in ["delete", "drop"])
        """
        return decision_result

    ############################################################################
    # Entry point
    ############################################################################

    @classmethod
    def inline(
        cls,
        thinking_prompt: str,
        llm: Optional[BaseLlm] = None,
        enable_rehearsal: bool = False,
        enable_reflection: bool = False,
        verbose: Optional[bool] = None,
        verbose_prompt: Optional[bool] = None,
        output_schema: Optional[Type[BaseModel]] = None,
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
        output_schema : Optional[Type[BaseModel]]
            If set, the worker produces a typed instance directly instead of
            going through the standard tool-call loop.

        Returns
        -------
        CognitiveWorker
            A worker instance with the specified thinking prompt.

        Examples
        --------
        >>> worker = CognitiveWorker.inline(
        ...     "Plan ONE immediate next step",
        ...     llm=llm,
        ...     enable_rehearsal=True
        ... )
        >>> planner = CognitiveWorker.inline(
        ...     "Analyze the goal and produce a phased plan.",
        ...     output_schema=PlanningResult,
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
            output_schema=output_schema,
        )

    # Alias for inline() — preferred for readability in some contexts
    from_prompt = inline

    async def arun(
        self,
        *args: Tuple[Any, ...],
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs
    ) -> Any:
        """Execute the thinking phase. Observation must be pre-set in context.observation."""
        start_time = time.time()
        result = await super().arun(*args, feedback_data=feedback_data, **kwargs)
        self.spend_time += time.time() - start_time
        return result


# Module-level default decision model (no policies, no output_schema).
# Used by tests and as a convenience for simple tool-call decisions.
ThinkDecision = CognitiveWorker._create_think_model(
    enable_rehearsal=False,
    enable_reflection=False,
    enable_acquiring=False,
    output_schema=None,
)
