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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from bridgic.core.cognitive._agent_automa import AgentAutoma
    from bridgic.core.cognitive._context import CognitiveContext


################################################################################################################
# Trace-level data structures
################################################################################################################

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


def resolve_worker(run_config: RunConfig, llm: Any) -> Any:
    """Instantiate a CognitiveWorker from a RunConfig.

    If worker_thinking_prompt is set, creates an inline worker.
    Otherwise, imports and instantiates the class by fully qualified name.
    """
    from bridgic.core.cognitive._cognitive_worker import CognitiveWorker

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


def _match_recorded_tools(
    recorded_calls: List[RecordedToolCall],
    context: CognitiveContext,
) -> List[Tuple[Any, Any]]:
    """Match recorded tool calls against available ToolSpecs in the context."""
    from bridgic.core.model.types import ToolCall

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


def _extract_loop_hints(goal: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Heuristically extract iterator and termination hints from a loop goal string.

    Returns (iterator_hint, termination_hint).
    """
    if not goal:
        return None, None

    iterator_hint = None
    termination_hint = None

    goal_lower = goal.lower()

    # Iterator hint: look for patterns like "for each X", "every X", "each X"
    iter_patterns = [
        r"(?:for\s+)?each\s+(.+?)(?:\s*[,，;；。.:]|\s+(?:that|which|and|check|do|process|handle))",
        r"every\s+(.+?)(?:\s*[,，;；。.:]|\s+(?:that|which|and|check|do|process|handle))",
        r"遍历\s*(.+?)(?:\s*[,，;；。.:]|\s+)",
        r"每[一个]*\s*(.+?)(?:\s*[,，;；。.:]|\s+)",
    ]
    for pat in iter_patterns:
        m = re.search(pat, goal_lower)
        if m:
            iterator_hint = m.group(1).strip()
            break

    # Termination hint: look for "until X", "when all X", "finish when"
    term_patterns = [
        r"(?:until|finish\s+when|stop\s+when|end\s+when)\s+(.+?)(?:\s*$|\s*[.。])",
        r"(?:直到|当|所有)\s*(.+?)(?:\s*$|\s*[.。])",
    ]
    for pat in term_patterns:
        m = re.search(pat, goal_lower)
        if m:
            termination_hint = m.group(1).strip()
            break

    return iterator_hint, termination_hint


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

    def begin_phase(self, phase_type: str, goal: Optional[str] = None) -> None:
        self._stack.append(_PhaseAccumulator(phase_type, goal=goal))

    def end_phase(self) -> None:
        if self._stack:
            self._completed.append(self._stack.pop())

    def record_step(self, step_data: dict) -> None:
        """Route a trace step to the active phase or the orphan list."""
        if self._stack:
            self._stack[-1].steps.append(step_data)
        else:
            self._orphan_steps.append(step_data)

    def set_run_config(self, config: dict) -> None:
        """Attach run() parameters to the current (topmost) phase."""
        if self._stack:
            self._stack[-1].run_config = config

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
                    )
                    for s in phase.steps
                ]
                blocks.append(SequentialBlock(
                    goal=phase.goal,
                    run_config=run_config,
                    steps=steps,
                ))

            elif phase.phase_type == "loop":
                # Split steps into iterations by finished=True boundaries
                iterations: List[List[dict]] = []
                current_iter: List[dict] = []
                for s in phase.steps:
                    current_iter.append(s)
                    if s.get("finished", False):
                        iterations.append(current_iter)
                        current_iter = []
                if current_iter:  # trailing incomplete iteration
                    iterations.append(current_iter)

                # First iteration as body template
                body_raw = iterations[0] if iterations else []
                body_steps = [
                    TraceStep(
                        name=s["name"],
                        step_content=s.get("step_content", ""),
                        tool_calls=[
                            RecordedToolCall(**tc) for tc in s.get("tool_calls", [])
                        ],
                        finished=s.get("finished", False),
                        observation_hash=s.get("observation_hash"),
                    )
                    for s in body_raw
                ]

                # Heuristic extraction from goal
                iterator_hint, termination_hint = _extract_loop_hints(phase.goal)

                code_slot = LoopCodeSlot(
                    slot_id=f"loop_{idx}_{uuid.uuid4().hex[:8]}",
                    description=f"Loop control for block {idx}",
                    iterator_hint=iterator_hint,
                    termination_hint=termination_hint,
                    generated_code=None,
                )

                blocks.append(LoopBlock(
                    goal=phase.goal,
                    run_config=run_config,
                    body_steps=body_steps,
                    observed_iterations=len(iterations),
                    code_slot=code_slot,
                ))

        return Workflow(blocks=blocks, metadata=metadata or {})


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

    async def arun(self, agent: AgentAutoma, context: CognitiveContext) -> CognitiveContext:
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
        """
        agent._log("Workflow", f"Replay starting: {len(self.blocks)} block(s)", color="magenta")

        for block_idx, block in enumerate(self.blocks):
            block_type = type(block).__name__
            block_goal = getattr(block, 'goal', None) or ''
            goal_preview = (block_goal[:80] + '...') if len(block_goal) > 80 else block_goal

            if isinstance(block, SequentialBlock):
                agent._log(
                    "Workflow",
                    f"Block {block_idx}: {block_type} "
                    f"({len(block.steps)} steps) | goal={goal_preview}",
                    color="magenta",
                )
            elif isinstance(block, LoopBlock):
                has_code = block.code_slot.generated_code is not None
                agent._log(
                    "Workflow",
                    f"Block {block_idx}: {block_type} "
                    f"({len(block.body_steps)} body steps, "
                    f"{block.observed_iterations} iterations, "
                    f"has_code={has_code}) | goal={goal_preview}",
                    color="magenta",
                )

            try:
                if isinstance(block, SequentialBlock):
                    await self._replay_sequential(agent, block, block_idx, context)
                elif isinstance(block, LoopBlock):
                    await self._replay_loop(agent, block, block_idx, context)
            except WorkflowDivergenceError as e:
                agent._log(
                    "Workflow",
                    f"Divergence at block {block_idx}: {e.reason}. "
                    f"Falling back to agent mode.",
                    color="red",
                )
                await self._fallback_from(agent, block_idx, context)
                break

        agent._log("Workflow", "Replay finished.", color="magenta")
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
    ) -> None:
        """Replay a sequential block by re-executing recorded tool calls."""
        from bridgic.core.cognitive._cognitive_worker import _DELEGATE
        from bridgic.core.cognitive._context import Step

        worker = resolve_worker(block.run_config, agent._llm)

        for step_idx, step in enumerate(block.steps):
            # 1. Observe current state (for context, not for divergence check)
            obs = await worker.observation(context)
            if obs is _DELEGATE:
                obs = await agent.observation(context)
            context.observation = obs

            # Sequential blocks replay tool calls unconditionally —
            # no divergence check, deterministic re-execution.

            # 2. Replay tool calls (skip LLM)
            if step.tool_calls:
                tool_call_pairs = _match_recorded_tools(step.tool_calls, context)
                if tool_call_pairs:
                    tool_names = [tc.tool_name for tc in step.tool_calls]
                    agent._log(
                        "Workflow",
                        f"Replay step {step_idx}/{len(block.steps)-1}: "
                        f"tools={tool_names}",
                        color="magenta",
                    )
                    tool_call_pairs = await worker.before_action(tool_call_pairs, context)
                    action_result = await agent.action_tool_call(tool_call_pairs, context)
                    result_step = Step(
                        content=step.step_content,
                        result=action_result.results,
                        metadata={
                            "tool_calls": [tc[0].name for tc in tool_call_pairs],
                            "tool_arguments": [tc[0].arguments for tc in tool_call_pairs],
                            "action_results": [
                                {
                                    "tool_name": r.tool_name,
                                    "tool_arguments": r.tool_arguments,
                                    "tool_result": r.tool_result,
                                }
                                for r in action_result.results
                            ],
                            "replayed": True,
                        },
                    )
                    context.add_info(result_step)

    async def _replay_loop(
        self,
        agent: AgentAutoma,
        block: LoopBlock,
        block_idx: int,
        context: CognitiveContext,
    ) -> None:
        """Replay a loop block.

        If the code_slot has generated_code, use it for iteration control.
        Otherwise, fall back to agent mode for this block.
        """
        from bridgic.core.cognitive._cognitive_worker import _DELEGATE
        from bridgic.core.cognitive._context import Step

        if block.code_slot.generated_code is None:
            # No code slot filled — run in agent mode
            agent._log(
                "Workflow",
                f"Loop block {block_idx}: no generated code, "
                f"falling back to agent mode",
                color="red",
            )
            worker = resolve_worker(block.run_config, agent._llm)
            await agent.run(
                worker,
                max_attempts=block.run_config.max_attempts,
                tools=block.run_config.tools,
                skills=block.run_config.skills,
            )
            return

        # Execute with generated loop control code
        loop_ns: Dict[str, Any] = {"context": context}
        exec(block.code_slot.generated_code, loop_ns)  # noqa: S102
        should_continue = loop_ns.get("should_continue")
        if not callable(should_continue):
            raise RuntimeError(
                f"Loop code slot '{block.code_slot.slot_id}' must define a "
                f"'should_continue(context)' function."
            )

        iteration = 0
        while should_continue(context):
            for step in block.body_steps:
                worker = resolve_worker(block.run_config, agent._llm)
                obs = await worker.observation(context)
                if obs is _DELEGATE:
                    obs = await agent.observation(context)
                context.observation = obs

                if step.tool_calls:
                    tool_call_pairs = _match_recorded_tools(step.tool_calls, context)
                    if tool_call_pairs:
                        action_result = await agent.action_tool_call(tool_call_pairs, context)
                        result_step = Step(
                            content=step.step_content,
                            result=action_result.results,
                            metadata={
                                "tool_calls": [tc[0].name for tc in tool_call_pairs],
                                "replayed": True,
                                "loop_iteration": iteration,
                            },
                        )
                        context.add_info(result_step)

            iteration += 1

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
            color="red",
        )

        for fb_idx, block in enumerate(remaining_blocks):
            agent._log(
                "Workflow",
                f"Fallback block {block_idx + fb_idx}: "
                f"{type(block).__name__} (agent mode)",
                color="red",
            )
            worker = resolve_worker(block.run_config, agent._llm)
            await agent.run(
                worker,
                max_attempts=block.run_config.max_attempts,
                tools=block.run_config.tools,
                skills=block.run_config.skills,
            )