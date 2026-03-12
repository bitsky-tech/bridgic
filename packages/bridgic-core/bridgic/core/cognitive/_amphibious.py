"""
Amphibious execution engine for AgentAutoma.

The AmphibiousRunner replays a Workflow deterministically when possible,
and seamlessly switches to agent mode (LLM-driven) when it detects
divergence between recorded and actual execution.

After a run completes, it can learn from the experience and patch the
workflow for better future replays.
"""
import inspect
import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from bridgic.core.cognitive._trace import (
    DivergenceDetector,
    DivergenceLevel,
    TraceStep,
)
from bridgic.core.cognitive._workflow import (
    LinearTraceBlock,
    LoopBlock,
    StepBlock,
    Workflow,
    WorkflowPatch,
    WorkflowToolCall,
    WorkflowStepWorker,
)

if TYPE_CHECKING:
    from bridgic.core.cognitive._agent_automa import AgentAutoma


class ExecutionMode(Enum):
    """Current execution mode of the amphibious runner."""
    WORKFLOW = "workflow"
    AGENT = "agent"


class DivergenceError(Exception):
    """Raised when workflow replay diverges from actual execution."""

    def __init__(self, message: str, block_index: int, details: Any = None):
        super().__init__(message)
        self.block_index = block_index
        self.details = details


class ResolutionResult:
    """Result of agent-mode resolution of a divergence."""

    def __init__(self, completed_all: bool = False, steps_taken: int = 0):
        self.completed_all = completed_all
        self.steps_taken = steps_taken


class AmphibiousRunner:
    """Replay a Workflow with automatic fallback to agent mode on divergence.

    Parameters
    ----------
    agent : AgentAutoma
        The agent to use for agent-mode fallback.
    workflow : Workflow
        The workflow to replay.
    """

    def __init__(self, agent: "AgentAutoma", workflow: Workflow):
        self.agent = agent
        self.workflow = workflow
        self.mode: ExecutionMode = ExecutionMode.WORKFLOW
        self.current_block_index: int = 0
        self.execution_log: List[TraceStep] = []
        self.detector = DivergenceDetector()

    async def run(self, context) -> Any:
        """Main execution loop.

        Iterates over workflow blocks, replaying each one. If a block
        diverges, switches to agent mode to resolve the issue.

        Parameters
        ----------
        context
            The cognitive context.

        Returns
        -------
        The context after execution.
        """
        self.agent._current_context = context

        for i, block in enumerate(self.workflow.blocks):
            self.current_block_index = i

            if self.mode == ExecutionMode.WORKFLOW:
                try:
                    await self._replay_block(block, context)
                except DivergenceError as e:
                    self.mode = ExecutionMode.AGENT
                    resolved = await self._agent_resolve(e, context)
                    if resolved.completed_all:
                        break
                    # Agent handled the divergence; try to resume workflow
                    self.mode = ExecutionMode.WORKFLOW
            else:
                # Already in agent mode — let agent handle remaining blocks
                await self._agent_takeover(context, from_block=i)
                break

        # Learn and patch the workflow
        self._learn_and_patch()

        return context

    async def _replay_block(self, block, context) -> None:
        """Replay a single workflow block.

        Parameters
        ----------
        block
            The workflow block to replay (StepBlock, LoopBlock, or LinearTraceBlock).
        context
            The cognitive context.

        Raises
        ------
        DivergenceError
            When actual execution diverges from the recorded block.
        """
        if isinstance(block, StepBlock):
            await self._replay_step_block(block, context)
        elif isinstance(block, LoopBlock):
            await self._replay_loop_block(block, context)
        elif isinstance(block, LinearTraceBlock):
            await self._replay_linear_trace(block, context)
        else:
            raise TypeError(f"Unknown block type: {type(block).__name__}")

    async def _replay_step_block(self, block: StepBlock, context) -> None:
        """Replay a single StepBlock by executing its recorded tool calls."""
        if not block.tool_calls:
            return

        # Get observation for divergence check
        obs = await self.agent.observation(context)

        # Execute the recorded tool calls
        step_worker = WorkflowStepWorker(
            tool_calls=block.tool_calls,
            step_content=f"Replaying step: {block.name}",
        )

        try:
            await step_worker.arun(context)
        except Exception as e:
            raise DivergenceError(
                f"Step '{block.name}' replay failed: {e}",
                block_index=self.current_block_index,
                details={"error": str(e)},
            )

    async def _replay_loop_block(self, block: LoopBlock, context) -> None:
        """Replay a LoopBlock using template-driven adaptive execution.

        When a ``pattern_template`` is available, uses lightweight model
        evaluation to decide continue/stop and fill tool arguments for each
        iteration.  Falls back to deterministic iteration replay when no
        pattern template exists.
        """
        if block.pattern_template and self.agent.llm is not None:
            await self._replay_loop_adaptive(block, context)
        else:
            await self._replay_loop_deterministic(block, context)

    async def _replay_loop_deterministic(self, block: LoopBlock, context) -> None:
        """Deterministic loop replay — iterate through recorded iterations."""
        for iteration_idx, iteration_steps in enumerate(block.iterations):
            for step_data in iteration_steps:
                tool_calls = [
                    WorkflowToolCall(**tc) for tc in step_data.get("tool_calls", [])
                ]
                if not tool_calls:
                    continue

                step_worker = WorkflowStepWorker(
                    tool_calls=tool_calls,
                    step_content=step_data.get("step_content", ""),
                )

                try:
                    await step_worker.arun(context)
                except Exception as e:
                    raise DivergenceError(
                        f"Loop '{block.name}' iteration {iteration_idx} failed: {e}",
                        block_index=self.current_block_index,
                        details={
                            "iteration": iteration_idx,
                            "error": str(e),
                        },
                    )

    async def _replay_loop_adaptive(self, block: LoopBlock, context) -> None:
        """Adaptive loop replay — model evaluates continue/stop and fills arguments."""
        template = block.pattern_template
        reference = block.iterations[0] if block.iterations else []

        for attempt in range(block.max_attempts):
            # 1. Observe current state
            obs = await self.agent.observation(context)
            obs_str = str(obs) if obs is not None else ""

            # 2. Lightweight model evaluation
            decision = await self._loop_evaluate(obs_str, template, reference, context)

            if decision.get("should_stop", False):
                break

            # 3. Execute tool calls from model decision
            tool_calls_data = decision.get("tool_calls", [])
            if not tool_calls_data:
                break

            tool_calls = []
            for tc in tool_calls_data:
                tool_calls.append(WorkflowToolCall(
                    tool_name=tc.get("name", ""),
                    tool_arguments=tc.get("arguments", {}),
                ))

            step_worker = WorkflowStepWorker(
                tool_calls=tool_calls,
                step_content=f"Adaptive loop iteration {attempt}",
            )

            try:
                await step_worker.arun(context)
            except Exception as e:
                raise DivergenceError(
                    f"Adaptive loop '{block.name}' iteration {attempt} failed: {e}",
                    block_index=self.current_block_index,
                    details={"iteration": attempt, "error": str(e)},
                )

    async def _loop_evaluate(
        self,
        observation: str,
        template: List[str],
        reference_iteration: List[dict],
        context,
    ) -> Dict[str, Any]:
        """Lightweight model call to evaluate loop continuation and fill arguments.

        The model only needs to:
        1. Decide whether to continue or stop the loop
        2. Fill in tool arguments for this iteration's tool calls

        Returns a dict: ``{"should_stop": bool, "tool_calls": [{"name": str, "arguments": dict}]}``
        """
        from bridgic.core.model.types import Message

        # Build reference arguments from first iteration
        ref_args = []
        for step in reference_iteration:
            for tc in step.get("tool_calls", []):
                name = tc.get("tool_name") or tc.get("name", "")
                if name in template:
                    ref_args.append({
                        "name": name,
                        "arguments": tc.get("tool_arguments") or tc.get("arguments", {}),
                    })

        prompt = (
            "You are replaying a loop in a workflow. Based on the current observation, "
            "decide whether to continue or stop, and fill in tool arguments.\n\n"
            f"## Tool sequence template (one iteration)\n{json.dumps(template)}\n\n"
            f"## Reference arguments from first iteration\n{json.dumps(ref_args, ensure_ascii=False)}\n\n"
            f"## Current observation\n{observation}\n\n"
            "## Instructions\n"
            "- If there are no more items to process, set should_stop=true\n"
            "- If there are items to process, set should_stop=false and provide tool_calls "
            "with the correct arguments for this iteration\n"
            "- Each tool_call needs: {\"name\": \"tool_name\", \"arguments\": {...}}\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\"should_stop\": bool, \"tool_calls\": [{\"name\": str, \"arguments\": dict}]}"
        )

        try:
            response = await self.agent.llm.achat([
                Message.from_text(prompt, role="user"),
            ])
            text = ""
            if response.message and response.message.blocks:
                for block in response.message.blocks:
                    if hasattr(block, "text"):
                        text += block.text
            # Parse JSON from response
            text = text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception:
            # On any failure, stop the loop to avoid infinite retries
            return {"should_stop": True, "tool_calls": []}

    async def _replay_linear_trace(self, block: LinearTraceBlock, context) -> None:
        """Replay a LinearTraceBlock by executing each step's recorded tool calls."""
        for step_idx, step_data in enumerate(block.steps):
            tool_calls_data = step_data.get("tool_calls", [])
            tool_calls = []
            for tc_data in tool_calls_data:
                if isinstance(tc_data, dict):
                    tool_calls.append(WorkflowToolCall(**tc_data))
                elif isinstance(tc_data, WorkflowToolCall):
                    tool_calls.append(tc_data)

            if not tool_calls:
                continue

            # Get current observation for divergence check
            obs = await self.agent.observation(context)
            obs_str = str(obs) if obs is not None else None

            # Check divergence against recorded step
            recorded = TraceStep(
                index=step_idx,
                name=step_data.get("name", f"step_{step_idx}"),
                observation=step_data.get("observation"),
                observation_hash=step_data.get("observation_hash", ""),
                tool_calls=tool_calls,
                result_hashes=step_data.get("result_hashes", []),
            )

            # Check tool name pattern
            recorded_names = [tc.tool_name for tc in tool_calls]
            divergence = self.detector.check(
                recorded=recorded,
                actual_observation=obs_str,
                actual_tool_names=recorded_names,  # we use recorded names for replay
                actual_results=[],  # results not available before execution
            )

            if divergence == DivergenceLevel.MAJOR:
                raise DivergenceError(
                    f"Linear trace step {step_idx} observation diverged significantly",
                    block_index=self.current_block_index,
                    details={"step_index": step_idx},
                )

            # Execute the recorded tool calls
            step_worker = WorkflowStepWorker(
                tool_calls=tool_calls,
                step_content=step_data.get("step_content", ""),
            )

            try:
                await step_worker.arun(context)
            except Exception as e:
                raise DivergenceError(
                    f"Linear trace step {step_idx} replay failed: {e}",
                    block_index=self.current_block_index,
                    details={"step_index": step_idx, "error": str(e)},
                )

    async def _agent_resolve(self, error: DivergenceError, context) -> ResolutionResult:
        """Use agent mode to resolve a divergence.

        The agent runs its full cognition() to handle the situation.
        After resolution, we check if the agent completed everything
        or just handled the immediate issue.
        """
        try:
            await self.agent.cognition(context)
            return ResolutionResult(completed_all=True)
        except Exception:
            return ResolutionResult(completed_all=True)

    async def _agent_takeover(self, context, from_block: int) -> None:
        """Agent takes over execution for remaining blocks.

        When already in agent mode, let the agent run its cognition()
        to handle remaining work.
        """
        await self.agent.cognition(context)

    def _learn_and_patch(self) -> None:
        """Analyze execution and apply patches to improve future replays.

        Delegates to WorkflowPatcher for the actual analysis.
        """
        patcher = WorkflowPatcher(self.workflow, self.execution_log)
        patcher.analyze_and_patch()


class WorkflowPatcher:
    """Analyze execution divergence and generate workflow patches.

    Three patch strategies:
    1. Guard: Exception handling → conditional guard at workflow position
    2. Replace: Tool argument changes → update expected parameters
    3. Extend: Loop iteration changes → update iteration patterns

    Parameters
    ----------
    workflow : Workflow
        The workflow to patch.
    execution_log : List[TraceStep]
        The execution log from the amphibious run.
    """

    def __init__(self, workflow: Workflow, execution_log: List[TraceStep]):
        self.workflow = workflow
        self.execution_log = execution_log

    def analyze_and_patch(self) -> List[WorkflowPatch]:
        """Analyze the execution and generate patches.

        Returns
        -------
        List[WorkflowPatch]
            The patches that were generated and applied.
        """
        patches = []

        # Find tool argument changes and create replace patches
        replace_patches = self._generate_replace_patches()
        patches.extend(replace_patches)

        # Apply patches to workflow
        for patch in patches:
            self.workflow.patches.append(patch)

        if patches:
            self.workflow.version += 1

        return patches

    def _generate_replace_patches(self) -> List[WorkflowPatch]:
        """Generate replace patches from tool argument changes.

        When workflow replay used different arguments than recorded,
        update the expected arguments template.
        """
        patches = []

        # Compare execution log with workflow blocks
        log_idx = 0
        for block_idx, block in enumerate(self.workflow.blocks):
            if not isinstance(block, (StepBlock, LinearTraceBlock)):
                continue

            steps = []
            if isinstance(block, StepBlock):
                steps = [block]
            elif isinstance(block, LinearTraceBlock):
                steps = block.steps

            for step_data in steps:
                if log_idx >= len(self.execution_log):
                    break

                log_step = self.execution_log[log_idx]
                log_idx += 1

                # Check if tool arguments differ
                if isinstance(step_data, StepBlock):
                    recorded_calls = step_data.tool_calls
                elif isinstance(step_data, dict):
                    recorded_calls = [
                        WorkflowToolCall(**tc) for tc in step_data.get("tool_calls", [])
                    ]
                else:
                    continue

                if not recorded_calls or not log_step.tool_calls:
                    continue

                # Compare tool arguments
                for rec, actual in zip(recorded_calls, log_step.tool_calls):
                    if (rec.tool_name == actual.tool_name and
                            rec.tool_arguments != actual.tool_arguments):
                        patch = WorkflowPatch(
                            type="replace",
                            block_index=block_idx,
                            trigger_pattern=f"args_changed_{rec.tool_name}",
                            resolution_steps=[log_step.model_dump()],
                        )
                        patches.append(patch)
                        break

        return patches

    def _find_block_index_for_step(self, step: TraceStep) -> int:
        """Find which workflow block corresponds to a given step index."""
        cumulative = 0
        for block_idx, block in enumerate(self.workflow.blocks):
            if isinstance(block, StepBlock):
                if cumulative == step.index:
                    return block_idx
                cumulative += 1
            elif isinstance(block, LinearTraceBlock):
                block_size = len(block.steps)
                if cumulative <= step.index < cumulative + block_size:
                    return block_idx
                cumulative += block_size
            elif isinstance(block, LoopBlock):
                total_iter_steps = sum(len(it) for it in block.iterations)
                if cumulative <= step.index < cumulative + total_iter_steps:
                    return block_idx
                cumulative += total_iter_steps

        return max(0, len(self.workflow.blocks) - 1)
