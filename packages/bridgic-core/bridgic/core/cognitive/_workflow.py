"""
Workflow Module — data models, builder, and replay engine for agent execution traces.

This module contains:
- Data models: TraceStep, RunConfig, SequentialBlock, LoopBlock, Workflow, etc.
- WorkflowBuilder: stack-based builder that converts phase-annotated trace steps
  into structured Workflow blocks.
- Replay: Workflow.arun() replays a recorded workflow deterministically,
  falling back to agent mode on divergence.
- Utilities: observation fingerprinting, loop hint extraction, worker resolution.

Key Types
---------
- TraceStep: One observe-think-act cycle record
- RunConfig: Captured run() parameters for replay/fallback
- SequentialBlock: Linear execution block (steps run once in order)
- LoopBlock: Iterative block with a code slot for loop control
- Workflow: Complete workflow definition with replay capability
- WorkflowBuilder: Converts raw trace data into a Workflow
"""

from __future__ import annotations
import hashlib
import importlib
import json
import re
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from bridgic.core.model.protocols import PydanticModel
from bridgic.core.model.types import ToolCall
from bridgic.core.model.types._message import Message
from bridgic.core.cognitive._cognitive_worker import _DELEGATE, CognitiveWorker
from bridgic.core.cognitive._context import CognitiveContext, Step
if TYPE_CHECKING:
    from bridgic.core.cognitive._agent_automa import AgentAutoma
    


################################################################################################################
# Trace-level data structures
################################################################################################################

class StepOutputType(str, Enum):
    """Discriminator for the kind of output a trace step produced."""
    TOOL_CALLS = "tool_calls"
    STRUCTURED = "structured"
    CONTENT_ONLY = "content_only"


class RecordedToolCall(BaseModel):
    """A complete record of one tool invocation."""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_result: Any


class TraceStep(BaseModel):
    """Record of one observe-think-act cycle."""
    model_config = ConfigDict(extra="forbid")

    name: str                                       # self.run(name=...) 的 name
    step_content: str                               # LLM reasoning content
    tool_calls: List[RecordedToolCall] = Field(default_factory=list)
    finished: bool = False                          # worker signalled finish?
    observation_hash: Optional[str] = None          # structural fingerprint for divergence detection
    observation_text: Optional[str] = None          # raw observation text for code generation
    output_type: StepOutputType = StepOutputType.TOOL_CALLS  # backward-compatible default
    structured_output: Optional[Dict[str, Any]] = None       # model_dump() serialization
    structured_output_class: Optional[str] = None            # FQN for deserialization


################################################################################################################
# Run configuration (captured from self.run())
################################################################################################################

class RunConfig(BaseModel):
    """Parameters captured from self.run(), used for replay and fallback."""
    model_config = ConfigDict(extra="forbid")

    worker_class: str                               # fully qualified class name
    worker_thinking_prompt: Optional[str] = None    # inline worker prompt
    tools: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    max_attempts: int = 1
    on_error: str = "raise"                         # ErrorStrategy.value


################################################################################################################
# Loop control code slot
################################################################################################################

class LoopCodeSlot(BaseModel):
    """Fillable code slot for loop iteration/termination control."""
    model_config = ConfigDict(extra="forbid")

    slot_id: str                                    # unique identifier
    description: str                                # human-readable description
    iterator_hint: Optional[str] = None             # e.g. "each order on the page"
    termination_hint: Optional[str] = None          # e.g. "all orders tracked"
    generated_code: Optional[str] = None            # filled later by model
    example_iterations: Optional[List[List[Dict[str, Any]]]] = None  # 2-round samples for code generation


################################################################################################################
# Workflow blocks
################################################################################################################

class WorkflowBlock(BaseModel):
    """Base class for workflow structural blocks."""
    model_config = ConfigDict(extra="forbid")

    block_type: Literal["sequential", "loop"]
    goal: Optional[str] = None
    run_config: RunConfig


class SequentialBlock(WorkflowBlock):
    """Sequential execution block — steps run once in recorded order."""
    block_type: Literal["sequential"] = "sequential"
    steps: List[TraceStep] = Field(default_factory=list)


class LoopBlock(WorkflowBlock):
    """Loop execution block — body_steps are one iteration template."""
    block_type: Literal["loop"] = "loop"
    body_steps: List[TraceStep] = Field(default_factory=list)
    observed_iterations: int = 1
    code_slot: LoopCodeSlot


################################################################################################################
# Errors
################################################################################################################

class WorkflowDivergenceError(Exception):
    """Raised during replay when the environment diverges from the recorded trace."""

    def __init__(self, block_index: int, step_index: int, reason: str):
        self.block_index = block_index
        self.step_index = step_index
        self.reason = reason
        super().__init__(
            f"Workflow divergence at block[{block_index}] step[{step_index}]: {reason}"
        )


################################################################################################################
# Utilities
################################################################################################################

def observation_fingerprint(obs: Any) -> Optional[str]:
    """Compute a stable hash fingerprint of an observation value.

    Used for divergence detection during replay.  Returns None for
    None observations.
    """
    if obs is None:
        return None
    try:
        serialized = json.dumps(obs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(obs)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _detect_iteration_boundary(
    steps: List[dict],
) -> Tuple[List[dict], int]:
    """Detect repeating pattern in a flat step list and extract the first iteration.

    Uses tool-name signature sequences with KMP failure function to find the
    minimal repeating period.  Returns ``(first_iteration_steps, iteration_count)``.
    If no clean repetition is found, returns ``(steps, 1)``.
    """
    if not steps:
        return steps, 0

    # Build signature sequence from tool names
    signatures: List[Tuple[str, ...]] = []
    for s in steps:
        tool_calls = s.get("tool_calls", [])
        if tool_calls:
            sig = tuple(tc["tool_name"] for tc in tool_calls)
        else:
            sig = ("__content__",)
        signatures.append(sig)

    # KMP failure function
    n = len(signatures)
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and signatures[i] != signatures[j]:
            j = fail[j - 1]
        if signatures[i] == signatures[j]:
            j += 1
        fail[i] = j

    period = n - fail[-1]
    if period < n and n % period == 0:
        return steps[:period], n // period

    # No clean repetition detected
    return steps, 1


################################################################################################################
# Workflow Builder
################################################################################################################

class _PhaseAccumulator:
    """Accumulates trace steps for a single structural phase (sequential or loop)."""

    __slots__ = ("phase_type", "goal", "steps", "run_config")

    def __init__(self, phase_type: str, goal: Optional[str] = None):
        self.phase_type: str = phase_type  # "sequential" | "loop"
        self.goal: Optional[str] = goal
        self.steps: List[dict] = []
        self.run_config: Optional[dict] = None  # captured from self.run()


class WorkflowBuilder:
    """Stack-based builder that converts phase-annotated trace steps into structured Workflow blocks.

    ``begin_phase()`` / ``end_phase()`` bracket a structural phase.
    ``record_step()`` routes step data to the active phase (or orphan list).
    ``set_run_config()`` attaches run() parameters to the current phase.
    ``build()`` converts completed phases into the appropriate block types.
    """

    def __init__(self):
        self._stack: List[_PhaseAccumulator] = []
        self._completed: List[_PhaseAccumulator] = []
        self._orphan_steps: List[dict] = []
        self._suppress_depth: int = 0

    def begin_phase(self, phase_type: str, goal: Optional[str] = None) -> None:
        self._stack.append(_PhaseAccumulator(phase_type, goal=goal))

    def end_phase(self) -> None:
        if self._stack:
            self._completed.append(self._stack.pop())

    def record_step(self, step_data: dict) -> None:
        """Route a trace step to the active phase or the orphan list.

        Steps are silently dropped when inside a ``suppress()`` context.
        """
        if self._suppress_depth > 0:
            return
        if self._stack:
            self._stack[-1].steps.append(step_data)
        else:
            self._orphan_steps.append(step_data)

    def set_run_config(self, config: dict) -> None:
        """Attach run() parameters to the current (topmost) phase."""
        if self._stack:
            self._stack[-1].run_config = config

    def is_loop_pattern_confirmed(self, min_iterations: int = 2) -> bool:
        """Check if the current loop phase has a confirmed repeating pattern.

        Returns True when the topmost phase is a loop and KMP detects
        >= min_iterations complete repetitions in the accumulated steps.
        """
        if not self._stack:
            return False
        current = self._stack[-1]
        if current.phase_type != "loop":
            return False
        if len(current.steps) < 2:
            return False
        _, iterations = _detect_iteration_boundary(current.steps)
        return iterations >= min_iterations

    @contextmanager
    def suppress(self):
        """Suppress trace step recording while inside this context.

        Supports nesting — recording resumes only when the outermost
        ``suppress()`` exits.
        """
        self._suppress_depth += 1
        try:
            yield
        finally:
            self._suppress_depth -= 1

    @contextmanager
    def phase(self, phase_type: str, goal: Optional[str] = None):
        """Context manager that brackets a structural phase."""
        self.begin_phase(phase_type, goal=goal)
        try:
            yield
        finally:
            self.end_phase()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, metadata: Optional[Dict[str, Any]] = None) -> Workflow:
        """Convert accumulated phases into a ``Workflow``."""
        blocks = []
        for idx, phase in enumerate(self._completed):
            run_config = RunConfig(**(phase.run_config or {
                "worker_class": "unknown",
            }))

            if phase.phase_type == "sequential":
                steps = [
                    TraceStep(
                        name=s["name"],
                        step_content=s.get("step_content", ""),
                        tool_calls=[
                            RecordedToolCall(**tc) for tc in s.get("tool_calls", [])
                        ],
                        finished=s.get("finished", False),
                        observation_hash=s.get("observation_hash"),
                        observation_text=s.get("observation_text"),
                        output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                        structured_output=s.get("structured_output"),
                        structured_output_class=s.get("structured_output_class"),
                    )
                    for s in phase.steps
                ]
                blocks.append(SequentialBlock(
                    goal=phase.goal,
                    run_config=run_config,
                    steps=steps,
                ))

            elif phase.phase_type == "loop":
                body_steps_raw, iterations = _detect_iteration_boundary(phase.steps)
                period = len(body_steps_raw)
                trace_steps = [
                    TraceStep(
                        name=s["name"],
                        step_content=s.get("step_content", ""),
                        tool_calls=[
                            RecordedToolCall(**tc) for tc in s.get("tool_calls", [])
                        ],
                        finished=s.get("finished", False),
                        observation_hash=s.get("observation_hash"),
                        observation_text=s.get("observation_text"),
                        output_type=StepOutputType(s.get("output_type", StepOutputType.TOOL_CALLS)),
                        structured_output=s.get("structured_output"),
                        structured_output_class=s.get("structured_output_class"),
                    )
                    for s in body_steps_raw
                ]

                # Extract up to 2 iteration samples for code generation
                example_iterations: Optional[List[List[Dict[str, Any]]]] = None
                if iterations >= 2 and period > 0:
                    max_obs_len = 8000
                    example_iterations = []
                    for iter_idx in range(min(2, iterations)):
                        start = iter_idx * period
                        end = start + period
                        iter_sample = []
                        for si, raw_step in enumerate(phase.steps[start:end]):
                            obs_text = raw_step.get("observation_text")
                            if obs_text and len(obs_text) > max_obs_len:
                                obs_text = obs_text[:max_obs_len]
                            tcs = raw_step.get("tool_calls", [])
                            iter_sample.append({
                                "step_idx": si,
                                "tool_name": tcs[0]["tool_name"] if tcs else None,
                                "observation_text": obs_text,
                                "tool_arguments": tcs[0].get("tool_arguments", {}) if tcs else {},
                                "step_content": raw_step.get("step_content", ""),
                            })
                        example_iterations.append(iter_sample)

                blocks.append(LoopBlock(
                    goal=phase.goal,
                    run_config=run_config,
                    body_steps=trace_steps,
                    observed_iterations=iterations,
                    code_slot=LoopCodeSlot(
                        slot_id=str(uuid.uuid4()),
                        description=phase.goal or "",
                        example_iterations=example_iterations,
                    ),
                ))

        return Workflow(blocks=blocks, metadata=metadata or {})


################################################################################################################
# Loop continue/stop decision model
################################################################################################################

class _LoopContinueDecision(BaseModel):
    """Lightweight structured output for loop termination check."""
    should_continue: bool
    reason: str = ""


class _LoopCodeGenResult(BaseModel):
    """Structured output for loop code generation."""
    code: str


################################################################################################################
# Workflow — data model + replay engine
################################################################################################################

class Workflow(BaseModel):
    """Complete workflow definition built from an agent execution trace.

    Beyond being a serializable data model, Workflow is also an active object
    that can replay itself through an AgentAutoma, with stage-level fallback
    to agent mode when the environment diverges from the recorded trace.
    """
    model_config = ConfigDict(extra="forbid")

    blocks: List[Union[SequentialBlock, LoopBlock]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Replay entry point
    # ------------------------------------------------------------------

    async def arun(
        self,
        agent: AgentAutoma,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
    ) -> CognitiveContext:
        """Replay this workflow using the given agent for execution.

        Iterates through blocks in order. On each block, replays recorded
        tool calls (skipping LLM). If the environment diverges from the
        recorded trace, falls back to agent mode for the remaining blocks.

        Parameters
        ----------
        agent : AgentAutoma
            The agent providing execution primitives (observation, tool
            execution, fallback to full agent mode).
        context : CognitiveContext
            The context to replay into.
        fill_llm : Any, optional
            LLM to use for adaptive fill when exact replay fails.
            Defaults to ``agent._llm``.
        """
        effective_fill_llm = fill_llm or agent._llm
        agent._log("Workflow", f"Replay starting: {len(self.blocks)} block(s)", color="green")

        for block_idx, block in enumerate(self.blocks):
            block_type = type(block).__name__
            block_goal = getattr(block, 'goal', None) or ''
            goal_preview = block_goal

            if isinstance(block, SequentialBlock):
                agent._log(
                    "Workflow",
                    f"Block {block_idx}: {block_type} "
                    f"({len(block.steps)} steps) | goal={goal_preview}"
                )
            elif isinstance(block, LoopBlock):
                agent._log(
                    "Workflow",
                    f"Block {block_idx}: LoopBlock "
                    f"({len(block.body_steps)} steps/iter, "
                    f"{block.observed_iterations} recorded iters) | goal={goal_preview}"
                )

            try:
                if isinstance(block, SequentialBlock):
                    await self._replay_sequential(
                        agent, block, block_idx, context,
                        fill_llm=effective_fill_llm,
                    )
                elif isinstance(block, LoopBlock):
                    await self._replay_loop(
                        agent, block, block_idx, context,
                        fill_llm=effective_fill_llm,
                    )
            except WorkflowDivergenceError as e:
                agent._log(
                    "Workflow",
                    f"Divergence at block {block_idx}: {e.reason}. "
                    f"Falling back to agent mode.",
                    color="red",
                )
                await self._fallback_from(agent, block_idx, context)
                break

        agent._log("Workflow", "Replay finished.", color="green")
        return context

    # ------------------------------------------------------------------
    # Block replay
    # ------------------------------------------------------------------

    async def _replay_sequential(
        self,
        agent: AgentAutoma,
        block: SequentialBlock,
        block_idx: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
    ) -> None:
        """Replay a sequential block: observe → dispatch → add_info for each step."""
        worker = self._resolve_worker(block.run_config, agent._llm)

        for step_idx, step in enumerate(block.steps):
            await self._observe(agent, worker, context)
            # Build semantic hint for adaptive fill
            hint_parts = []
            if block.goal:
                hint_parts.append(f"Phase goal: {block.goal}")
            if step.step_content:
                hint_parts.append(
                    f"Original reasoning (from capture): {step.step_content}"
                )
            step_hint = "\n".join(hint_parts)

            result_step = await self._dispatch_step(
                agent, worker, step, block_idx, step_idx,
                len(block.steps), context, fill_llm=fill_llm,
                step_hint=step_hint,
            )
            if result_step is not None:
                context.add_info(result_step)
            
            import asyncio
            await asyncio.sleep(3)

    async def _replay_loop(
        self,
        agent: AgentAutoma,
        block: LoopBlock,
        block_idx: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
    ) -> None:
        """Replay a loop block with three-level strategy:

        1. If generated_code already exists → _replay_loop_with_code()
        2. If example_iterations available → _generate_loop_code() → _replay_loop_with_code()
        3. Fallback → per-step adaptive fill with LLM-based termination
        """
        effective_fill_llm = fill_llm or agent._llm
        code_slot = block.code_slot

        # Level 1: pre-existing generated code
        if code_slot.generated_code:
            try:
                agent._log("Workflow", "Loop replay: using pre-existing generated code", color="cyan")
                await self._replay_loop_with_code(
                    agent, block, block_idx, context,
                    code_slot.generated_code,
                    fill_llm=effective_fill_llm,
                )
                return
            except Exception as exc:
                agent._log(
                    "Workflow",
                    f"Code-driven loop failed ({exc}), falling back to adaptive fill",
                    color="yellow",
                )

        # Level 2: generate code from example iterations
        if code_slot.example_iterations:
            try:
                worker = self._resolve_worker(block.run_config, agent._llm)
                await self._observe(agent, worker, context)
                live_obs = str(context.observation) if context.observation else ""

                agent._log("Workflow", "Loop replay: generating code from examples", color="cyan")
                generated = await self._generate_loop_code(
                    effective_fill_llm, block, live_obs,
                )
                code_slot.generated_code = generated
                # Clear examples to save memory
                code_slot.example_iterations = None

                await self._replay_loop_with_code(
                    agent, block, block_idx, context,
                    generated,
                    fill_llm=effective_fill_llm,
                )
                return
            except Exception as exc:
                agent._log(
                    "Workflow",
                    f"Code generation/execution failed ({exc}), falling back to adaptive fill",
                    color="yellow",
                )

        # Level 3: fallback to per-step adaptive fill + LLM termination
        agent._log("Workflow", "Loop replay: using per-step adaptive fill", color="yellow")
        await self._replay_loop_adaptive(
            agent, block, block_idx, context,
            fill_llm=effective_fill_llm,
        )

    async def _replay_loop_adaptive(
        self,
        agent: AgentAutoma,
        block: LoopBlock,
        block_idx: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
    ) -> None:
        """Fallback loop replay: per-step adaptive fill with LLM-based termination."""
        worker = self._resolve_worker(block.run_config, agent._llm)
        effective_fill_llm = fill_llm or agent._llm
        iteration = 0
        max_iter = block.run_config.max_attempts

        while iteration < max_iter:
            agent._log("Workflow", f"Loop iteration {iteration}", color="cyan")

            for step_idx, step in enumerate(block.body_steps):
                await self._observe(agent, worker, context)
                # Build semantic hint: block goal + original reasoning
                hint_parts = []
                if block.goal:
                    hint_parts.append(f"Loop goal: {block.goal}")
                if step.step_content:
                    hint_parts.append(
                        f"Original reasoning (from capture): {step.step_content}"
                    )
                step_hint = "\n".join(hint_parts)

                result_step = await self._execute_loop_step(
                    agent, worker, step, block_idx, step_idx,
                    len(block.body_steps), context,
                    fill_llm=effective_fill_llm,
                    step_hint=step_hint,
                )
                if result_step is not None:
                    context.add_info(result_step)

            iteration += 1

            # Termination check
            await self._observe(agent, worker, context)
            should_continue = await self._check_loop_continue(
                effective_fill_llm, block, context,
            )
            agent._log("Workflow", f"Loop continue={should_continue}", color="yellow")
            if not should_continue:
                break

        agent._log("Workflow", f"Loop done after {iteration} iteration(s)", color="green")

    # ------------------------------------------------------------------
    # Step dispatch & observation
    # ------------------------------------------------------------------

    async def _observe(
        self,
        agent: AgentAutoma,
        worker: Any,
        context: CognitiveContext,
    ) -> None:
        """Run observation phase: worker.observation() with _DELEGATE fallback."""
        obs = await worker.observation(context)
        if obs is _DELEGATE:
            obs = await agent.observation(context)
        context.observation = obs

    async def _dispatch_step(
        self,
        agent: AgentAutoma,
        worker: Any,
        step: TraceStep,
        block_idx: int,
        step_idx: int,
        total_steps: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
        step_hint: str = "",
    ) -> Optional[Step]:
        """Route a step to the appropriate execution strategy by output_type."""
        if step.output_type == StepOutputType.TOOL_CALLS:
            return await self._execute_tool_calls(
                agent, worker, step, block_idx, step_idx,
                total_steps, context, fill_llm=fill_llm,
                step_hint=step_hint,
            )
        elif step.output_type == StepOutputType.STRUCTURED:
            return await self._execute_structured(
                agent, worker, step, block_idx, step_idx,
                total_steps, context, fill_llm=fill_llm,
                step_hint=step_hint,
            )
        elif step.output_type == StepOutputType.CONTENT_ONLY:
            return self._execute_content_only(
                agent, step, step_idx, total_steps,
            )
        return None

    # ------------------------------------------------------------------
    # Step execution strategies
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        agent: AgentAutoma,
        worker: Any,
        step: TraceStep,
        block_idx: int,
        step_idx: int,
        total_steps: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
        step_hint: str = "",
    ) -> Optional[Step]:
        """Execute a TOOL_CALLS step with EXACT → ADAPTIVE → DIVERGENCE escalation."""
        if not step.tool_calls:
            return None

        tool_call_pairs = self._match_recorded_tools(step.tool_calls, context)
        if not tool_call_pairs:
            return None

        tool_names = [tc.tool_name for tc in step.tool_calls]
        agent._log(
            "Workflow",
            f"Replay step {step_idx}/{total_steps - 1}: "
            f"tools={tool_names}",
            color="purple",
        )

        async def exact_fn():
            pairs = await worker.before_action(tool_call_pairs, context)
            return await agent.action_tool_call(pairs, context)

        async def adaptive_fn():
            filled_pairs = await self._adaptive_fill_tool_args(
                fill_llm, step, context, step_hint=step_hint,
            )
            filled_pairs = await worker.before_action(filled_pairs, context)
            return await agent.action_tool_call(filled_pairs, context)

        action_result = await self._escalate(
            exact_fn=exact_fn,
            adaptive_fn=adaptive_fn,
            block_idx=block_idx,
            step_idx=step_idx,
            divergence_reason="Adaptive fill also failed",
            agent=agent,
            label="tool args",
        )

        return Step(
            content=step.step_content,
            result=action_result.results,
            metadata={
                "replayed": True,
                "output_type": "tool_calls",
            },
        )

    async def _execute_structured(
        self,
        agent: AgentAutoma,
        worker: Any,
        step: TraceStep,
        block_idx: int,
        step_idx: int,
        total_steps: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
        step_hint: str = "",
    ) -> Step:
        """Execute a STRUCTURED step with EXACT → ADAPTIVE → DIVERGENCE escalation."""
        agent._log(
            "Workflow",
            f"Replay step {step_idx}/{total_steps - 1}: "
            f"structured output ({step.structured_output_class})",
            color="purple",
        )

        async def exact_fn():
            output_obj = self._deserialize_structured_output(
                step.structured_output or {},
                step.structured_output_class,
            )
            return await agent.action_custom_output(output_obj, context)

        async def adaptive_fn():
            output_obj = await self._adaptive_fill_structured_output(
                fill_llm, step, context, step_hint=step_hint,
            )
            return await agent.action_custom_output(output_obj, context)

        action_result = await self._escalate(
            exact_fn=exact_fn,
            adaptive_fn=adaptive_fn,
            block_idx=block_idx,
            step_idx=step_idx,
            divergence_reason="Adaptive fill for structured output also failed",
            agent=agent,
            label="structured output",
        )

        return Step(
            content=step.step_content,
            result=action_result,
            metadata={"replayed": True, "output_type": "structured"},
        )

    def _execute_content_only(
        self,
        agent: AgentAutoma,
        step: TraceStep,
        step_idx: int,
        total_steps: int,
    ) -> Step:
        """Execute a CONTENT_ONLY step — no action, just record reasoning."""

        agent._log(
            "Workflow",
            f"Replay step {step_idx}/{total_steps - 1}: "
            f"content only (reasoning)",
            color="purple",
        )
        return Step(
            content=step.step_content,
            result=None,
            metadata={"replayed": True, "output_type": "content_only"},
        )

    async def _execute_loop_step(
        self,
        agent: AgentAutoma,
        worker: Any,
        step: TraceStep,
        block_idx: int,
        step_idx: int,
        total_steps: int,
        context: CognitiveContext,
        *,
        fill_llm: Any = None,
        step_hint: str = "",
    ) -> Optional[Step]:
        """Execute a single step inside a loop — always adaptive fill (no exact attempt)."""
        if step.output_type == StepOutputType.TOOL_CALLS and step.tool_calls:
            tool_names = [tc.tool_name for tc in step.tool_calls]
            agent._log(
                "Workflow",
                f"  Loop step {step_idx}/{total_steps - 1}: "
                f"tools={tool_names} (adaptive)",
                color="purple",
            )
            filled_pairs = await self._adaptive_fill_tool_args(
                fill_llm, step, context, step_hint=step_hint,
            )
            filled_pairs = await worker.before_action(filled_pairs, context)
            action_result = await agent.action_tool_call(filled_pairs, context)
            return Step(
                content=step.step_content,
                result=action_result.results,
                metadata={"replayed": True, "loop_adaptive": True},
            )
        elif step.output_type == StepOutputType.STRUCTURED:
            agent._log(
                "Workflow",
                f"  Loop step {step_idx}/{total_steps - 1}: "
                f"structured ({step.structured_output_class}) (adaptive)",
                color="purple",
            )
            output_obj = await self._adaptive_fill_structured_output(
                fill_llm, step, context, step_hint=step_hint,
            )
            action_result = await agent.action_custom_output(output_obj, context)
            return Step(
                content=step.step_content,
                result=action_result,
                metadata={"replayed": True, "loop_adaptive": True},
            )
        elif step.output_type == StepOutputType.CONTENT_ONLY:
            return self._execute_content_only(agent, step, step_idx, total_steps)
        return None

    async def _check_loop_continue(
        self,
        llm: Any,
        block: LoopBlock,
        context: CognitiveContext,
    ) -> bool:
        """Ask a lightweight LLM whether the loop should continue."""
        prompt = (
            f"Loop goal: {block.goal}\n\n"
            f"## Current Observation\n{context.observation}\n\n"
            f"## Recent Steps\n{context.summary()}\n\n"
            "Are there more items to process? "
            "should_continue=true if yes, false if all done."
            "Please output in JSON format."
        )
        result = await llm.astructured_output(
            messages=[Message.from_text(prompt)],
            constraint=PydanticModel(model=_LoopContinueDecision),
        )
        return result.should_continue

    # ------------------------------------------------------------------
    # Code-driven loop replay
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_code_slot(code: str) -> Dict[str, Any]:
        """Safely execute generated Python code in a restricted namespace.

        Returns a namespace dict that should contain ``get_items`` and
        ``fill_step_args`` callables.

        Raises ``RuntimeError`` if the code contains forbidden constructs
        or fails to execute.
        """
        _SAFE_BUILTINS = {
            "True": True, "False": False, "None": None,
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "filter": filter,
            "float": float, "frozenset": frozenset, "getattr": getattr,
            "hasattr": hasattr, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len,
            "list": list, "map": map, "max": max, "min": min,
            "next": next, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "type": type, "zip": zip,
        }

        # Block dangerous patterns — match standalone identifiers only
        _FORBIDDEN = [
            r'\b__import__\b', r'(?<!\w)eval\s*\(', r'(?<!\w)exec\s*\(',
            r'(?<!\.)compile\s*\(', r'(?<!\w)open\s*\(',
            r'\bimport\s+os\b', r'\bimport\s+sys\b',
        ]
        for pattern in _FORBIDDEN:
            if re.search(pattern, code):
                raise RuntimeError(f"Forbidden construct in generated code: {pattern}")

        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "re": re,
            "json": json,
        }
        exec(code, namespace)  # noqa: S102
        return namespace

    async def _generate_loop_code(
        self,
        llm: Any,
        block: LoopBlock,
        live_observation: str,
    ) -> str:
        """Call LLM once to generate Python code for loop replay.

        The generated code must define two functions:
        - ``get_items(observation: str) -> list[dict]``
        - ``fill_step_args(step_idx: int, tool_name: str, observation: str, item: dict) -> dict``

        Returns the generated code string.
        """
        code_slot = block.code_slot
        examples = code_slot.example_iterations or []

        # Build body step template
        step_templates = []
        for si, step in enumerate(block.body_steps):
            if step.tool_calls:
                tc = step.tool_calls[0]
                # Get tool schema from recorded call
                step_templates.append({
                    "step_idx": si,
                    "tool_name": tc.tool_name,
                    "argument_keys": list(tc.tool_arguments.keys()),
                })

        # Build example sections
        example_sections = []
        for iter_idx, iteration in enumerate(examples):
            lines = [f"### Iteration {iter_idx}"]
            for step_sample in iteration:
                lines.append(f"  Step {step_sample['step_idx']}: {step_sample.get('tool_name', 'N/A')}")
                obs = step_sample.get("observation_text", "")
                if obs:
                    lines.append(f"    Observation (excerpt): {obs[:2000]}")
                lines.append(f"    Arguments: {json.dumps(step_sample.get('tool_arguments', {}))}")
                lines.append(f"    Reasoning: {step_sample.get('step_content', '')}")
            example_sections.append("\n".join(lines))

        prompt = (
            "You are generating Python code to automate a loop workflow.\n\n"
            f"## Loop Goal\n{block.goal or 'Process items on the page'}\n\n"
            f"## Body Steps Template\n{json.dumps(step_templates, indent=2)}\n\n"
            f"## Example Iterations (from capture)\n"
            f"{chr(10).join(example_sections)}\n\n"
            f"## Current Live Observation\n{live_observation[:4000]}\n\n"
            "## Requirements\n"
            "Define exactly TWO functions (no imports needed, `re` and `json` are available):\n\n"
            "1. `get_items(observation: str) -> list[dict]`\n"
            "   Parse the observation text and return a list of dicts, one per item to process.\n"
            "   Each dict should contain the fields needed by fill_step_args.\n\n"
            "2. `fill_step_args(step_idx: int, tool_name: str, observation: str, item: dict) -> dict`\n"
            "   Given a step index, tool name, current observation, and an item dict,\n"
            "   return the tool arguments dict.\n\n"
            "Output ONLY the Python code. Do not use import, eval, exec, open, os, or sys."
        )

        result = await llm.astructured_output(
            messages=[Message.from_text(prompt)],
            constraint=PydanticModel(model=_LoopCodeGenResult),
        )
        return result.code

    async def _replay_loop_with_code(
        self,
        agent: AgentAutoma,
        block: LoopBlock,
        block_idx: int,
        context: CognitiveContext,
        code: str,
        *,
        fill_llm: Any = None,
    ) -> None:
        """Replay a loop block using generated code instead of per-step LLM calls."""
        worker = self._resolve_worker(block.run_config, agent._llm)
        namespace = self._execute_code_slot(code)

        get_items = namespace.get("get_items")
        fill_step_args = namespace.get("fill_step_args")
        if not callable(get_items) or not callable(fill_step_args):
            raise RuntimeError("Generated code missing get_items or fill_step_args")

        # Initial observation to extract items
        await self._observe(agent, worker, context)
        items = get_items(str(context.observation))
        agent._log("Workflow", f"Code-driven loop: {len(items)} item(s) found", color="cyan")

        for item_idx, item in enumerate(items):
            agent._log("Workflow", f"Code-driven loop item {item_idx}/{len(items) - 1}", color="cyan")
            for step_idx, step in enumerate(block.body_steps):
                await self._observe(agent, worker, context)

                if step.output_type == StepOutputType.STRUCTURED:
                    # Structured steps fall back to adaptive fill
                    output_obj = await self._adaptive_fill_structured_output(
                        fill_llm or agent._llm, step, context,
                        step_hint=f"Loop goal: {block.goal}",
                    )
                    action_result = await agent.action_custom_output(output_obj, context)
                    result_step = Step(
                        content=step.step_content,
                        result=action_result,
                        metadata={"replayed": True, "code_driven": True},
                    )
                elif step.output_type == StepOutputType.TOOL_CALLS and step.tool_calls:
                    # Use generated code to fill args
                    tc_template = step.tool_calls[0]
                    filled_args = fill_step_args(
                        step_idx, tc_template.tool_name,
                        str(context.observation), item,
                    )

                    # Match tool spec
                    _, tool_specs = context.get_field("tools")
                    matched_spec = None
                    for spec in tool_specs:
                        if tc_template.tool_name == spec.tool_name:
                            matched_spec = spec
                            break
                    if matched_spec is None:
                        continue

                    tc = ToolCall(
                        id=f"code_{item_idx}_{step_idx}",
                        name=tc_template.tool_name,
                        arguments=filled_args,
                    )
                    pairs = [(tc, matched_spec)]
                    pairs = await worker.before_action(pairs, context)
                    action_result = await agent.action_tool_call(pairs, context)
                    result_step = Step(
                        content=step.step_content,
                        result=action_result.results,
                        metadata={"replayed": True, "code_driven": True},
                    )
                elif step.output_type == StepOutputType.CONTENT_ONLY:
                    result_step = self._execute_content_only(
                        agent, step, step_idx, len(block.body_steps),
                    )
                else:
                    continue

                context.add_info(result_step)

        agent._log("Workflow", f"Code-driven loop done: {len(items)} item(s)", color="green")

    # ------------------------------------------------------------------
    # Escalation & adaptive fill
    # ------------------------------------------------------------------

    async def _escalate(
        self,
        *,
        exact_fn,
        adaptive_fn,
        block_idx: int,
        step_idx: int,
        divergence_reason: str,
        agent: AgentAutoma,
        label: str,
    ):
        """Execute with EXACT → ADAPTIVE → DIVERGENCE three-level escalation.

        Parameters
        ----------
        exact_fn : async () -> T
            Zero-overhead happy path using recorded data.
        adaptive_fn : async () -> T
            Lightweight LLM-based repair when exact fails.
        block_idx, step_idx : int
            Position for error reporting.
        divergence_reason : str
            Human-readable reason included in WorkflowDivergenceError.
        agent : AgentAutoma
            For logging.
        label : str
            Short description for log messages (e.g. "tool args", "structured output").
        """
        try:
            return await exact_fn()
        except Exception:
            agent._log(
                "Workflow",
                f"Exact replay failed at step {step_idx}, "
                f"trying adaptive fill for {label}",
                color="yellow",
            )
            try:
                return await adaptive_fn()
            except Exception:
                raise WorkflowDivergenceError(
                    block_idx, step_idx, divergence_reason,
                )

    async def _adaptive_fill_tool_args(
        self,
        llm: Any,
        step: TraceStep,
        context: CognitiveContext,
        *,
        step_hint: str = "",
    ) -> List[Tuple[Any, Any]]:
        """Use a lightweight LLM call to fill tool arguments based on current observation.

        Called when exact replay of recorded arguments fails.  Constructs a minimal
        prompt with the current observation and tool schema, then asks the LLM to
        produce correct arguments.

        Returns a list of (ToolCall, ToolSpec) pairs ready for execution.
        """
        _TYPE_MAP = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        def _build_args_model(tool_name: str, tool_parameters: dict) -> Type[BaseModel]:
            """Build a Pydantic model from a ToolSpec's tool_parameters JSON Schema."""
            properties = tool_parameters.get("properties", {})
            required = set(tool_parameters.get("required", []))
            field_definitions = {}
            for prop_name, prop_schema in properties.items():
                py_type = _TYPE_MAP.get(prop_schema.get("type", ""), str)
                if prop_name in required:
                    field_definitions[prop_name] = (py_type, ...)
                else:
                    field_definitions[prop_name] = (Optional[py_type], None)

            model_name = f"{tool_name}_Args"
            return create_model(model_name, **field_definitions)

        _, tool_specs = context.get_field("tools")
        results = []

        for idx, rc in enumerate(step.tool_calls):
            # Find matching ToolSpec
            matched_spec = None
            for spec in tool_specs:
                if rc.tool_name == spec.tool_name:
                    matched_spec = spec
                    break
            if matched_spec is None:
                continue

            # Build dynamic model from tool schema
            tool_params = matched_spec.tool_parameters or {}
            args_model = _build_args_model(rc.tool_name, tool_params)

            # Minimal prompt for argument fill
            hint_section = ""
            if step_hint:
                hint_section = f"## Step Intent\n{step_hint}\n\n"

            prompt = (
                "You are replaying a recorded workflow step. "
                "The original arguments no longer match the current page state. "
                "Use the STEP INTENT to understand WHAT this step is trying to do, "
                "then pick the correct arguments from the CURRENT OBSERVATION.\n\n"
                f"{hint_section}"
                "## Current Observation\n"
                f"{context.observation}\n\n"
                "## Tool to Call\n"
                f"Name: {rc.tool_name}\n"
                f"Parameters schema: {json.dumps(tool_params)}\n\n"
                "## Original Arguments (failed, for reference only)\n"
                f"{json.dumps(rc.tool_arguments)}\n\n"
                "Fill in the correct arguments based on the CURRENT observation."
                "Please output in JSON format."
            )

            messages = [Message.from_text(prompt)]
            fill_result = await llm.astructured_output(
                messages=messages,
                constraint=PydanticModel(model=args_model),
            )

            filled_args = fill_result.model_dump()
            tc = ToolCall(
                id=f"adaptive_{idx}",
                name=rc.tool_name,
                arguments=filled_args,
            )
            results.append((tc, matched_spec))

        return results

    async def _adaptive_fill_structured_output(
        self,
        llm: Any,
        step: TraceStep,
        context: CognitiveContext,
        *,
        step_hint: str = "",
    ) -> Any:
        """Use a lightweight LLM call to produce structured output based on current observation.

        Called when the recorded structured output is no longer valid.  Imports the
        original output class by FQN and asks the LLM to fill it from scratch.
        """
        if not step.structured_output_class:
            raise ValueError("No structured_output_class recorded for adaptive fill")

        # Import the output class
        module_path, _, class_name = step.structured_output_class.rpartition(".")
        if not module_path:
            raise ValueError(f"Cannot resolve class: {step.structured_output_class}")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        hint_section = ""
        if step_hint:
            hint_section = f"## Step Intent\n{step_hint}\n\n"

        prompt = (
            "You are replaying a recorded workflow step. "
            "The original structured output is no longer valid.\n\n"
            f"{hint_section}"
            "## Current Observation\n"
            f"{context.observation}\n\n"
            "## Original Output (for reference only)\n"
            f"{json.dumps(step.structured_output or {})}\n\n"
            "Produce the correct output based on the CURRENT observation."
            "Please output in JSON format."
        )

        messages = [Message.from_text(prompt)]
        return await llm.astructured_output(
            messages=messages,
            constraint=PydanticModel(model=cls),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_worker(run_config: RunConfig, llm: Any) -> Any:
        """Instantiate a CognitiveWorker from a RunConfig.

        If worker_thinking_prompt is set, creates an inline worker.
        Otherwise, imports and instantiates the class by fully qualified name.
        """
        if run_config.worker_thinking_prompt is not None:
            return CognitiveWorker.inline(
                run_config.worker_thinking_prompt,
                llm=llm,
            )

        module_path, _, class_name = run_config.worker_class.rpartition(".")
        if not module_path:
            raise RuntimeError(
                f"Cannot resolve worker class: {run_config.worker_class}"
            )
        module = importlib.import_module(module_path)
        worker_cls = getattr(module, class_name)
        return worker_cls(llm=llm)

    @staticmethod
    def _match_recorded_tools(
        recorded_calls: List[RecordedToolCall],
        context: CognitiveContext,
    ) -> List[Tuple[Any, Any]]:
        """Match recorded tool calls against available ToolSpecs in the context."""
        _, tool_specs = context.get_field("tools")
        matched = []
        for idx, rc in enumerate(recorded_calls):
            for spec in tool_specs:
                if rc.tool_name == spec.tool_name:
                    tc = ToolCall(
                        id=f"replay_{idx}",
                        name=rc.tool_name,
                        arguments=rc.tool_arguments,
                    )
                    matched.append((tc, spec))
                    break
        return matched

    @staticmethod
    def _deserialize_structured_output(
        data: Dict[str, Any],
        class_fqn: Optional[str],
    ) -> Any:
        """Safely reconstruct a structured output from its serialized form.

        Parameters
        ----------
        data : dict
            The dict produced by ``model_dump()`` during recording.
        class_fqn : Optional[str]
            Fully qualified class name (``module.ClassName``).  If importable
            and is a ``BaseModel`` subclass, we use ``model_validate()``.
            Otherwise, the raw dict is returned.
        """
        if class_fqn is None:
            return data
        try:
            module_path, _, class_name = class_fqn.rpartition(".")
            if not module_path:
                return data
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if isinstance(cls, type) and issubclass(cls, BaseModel):
                return cls.model_validate(data)
        except Exception:
            pass
        return data

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    async def _fallback_from(
        self,
        agent: AgentAutoma,
        block_idx: int,
        context: CognitiveContext,
    ) -> None:
        """Fall back to agent mode starting from the given block.

        Re-runs remaining blocks using their saved run_config. Each block
        starts from scratch — the agent observes current state and adapts.
        """
        remaining_blocks = self.blocks[block_idx:]
        agent._log(
            "Workflow",
            f"Fallback: running {len(remaining_blocks)} "
            f"remaining block(s) in agent mode",
            color="yellow",
        )

        for fb_idx, block in enumerate(remaining_blocks):
            agent._log(
                "Workflow",
                f"Fallback block {block_idx + fb_idx}: "
                f"{type(block).__name__} (agent mode)",
                color="yellow",
            )
            worker = self._resolve_worker(block.run_config, agent._llm)
            await agent.run(
                worker,
                max_attempts=block.run_config.max_attempts,
                tools=block.run_config.tools,
                skills=block.run_config.skills,
            )