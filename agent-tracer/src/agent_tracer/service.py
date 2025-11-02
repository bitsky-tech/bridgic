"""Tracing service - main API for agent tracing."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, AsyncGenerator
from uuid import UUID, uuid4

from agent_tracer.schema import TracingConfig
from agent_tracer.utils import mask_sensitive_data

if TYPE_CHECKING:
    from agent_tracer.base import BaseTracer
    from agent_tracer.schema import Log

# Setup logging
logger = logging.getLogger("agent_tracer")

# Context variables for managing trace state
trace_context_var: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)
step_context_var: ContextVar[StepTraceContext | None] = ContextVar("step_trace_context", default=None)


class TraceContext:
    """Context for a trace run."""

    def __init__(
        self,
        run_id: UUID | str,
        run_name: str,
        project_name: str,
        user_id: str | None,
        session_id: str | None,
    ):
        self.run_id: UUID | str = run_id
        self.run_name: str = run_name
        self.project_name: str = project_name
        self.user_id: str | None = user_id
        self.session_id: str | None = session_id
        self.tracers: dict[str, BaseTracer] = {}
        self.all_inputs: dict[str, dict] = defaultdict(dict)
        self.all_outputs: dict[str, dict] = defaultdict(dict)

        self.traces_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.worker_task: asyncio.Task | None = None


class StepTraceContext:
    """Context for a step/component trace."""

    def __init__(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        custom_data: dict[str, Any] | None = None,
        parent_step_trace_context: "StepTraceContext" | None = None,
    ) -> None:
        self.trace_id: str = trace_id
        self.trace_name: str = trace_name
        self.trace_type: str = trace_type
        self.inputs: dict[str, Any] = inputs
        self.inputs_metadata: dict[str, Any] = metadata or {}
        self.custom_data: dict[str, Any] = custom_data or {}
        self.outputs: dict[str, Any] = defaultdict(dict)
        self.outputs_metadata: dict[str, Any] = defaultdict(dict)
        self.logs: dict[str, list[Log | dict[Any, Any]]] = defaultdict(list)
        self.parent_step_trace_context: "StepTraceContext" | None = parent_step_trace_context

class TracingService:
    """Main tracing service for agent observability.

    Usage:
        1. start_trace: Start a trace for a workflow/agent run
        2. trace_step: Context manager for tracing individual steps
           - set_outputs: Set outputs for the current step
           - add_log: Add log entries
           - get_langchain_callbacks: Get LangChain callbacks
        3. end_trace: End the trace
    """

    def __init__(self, config: TracingConfig | None = None):
        """Initialize the tracing service.

        Args:
            config: Optional configuration for tracing
        """
        self.config = config or TracingConfig()
        self.deactivated = self.config.deactivate_tracing

    async def _trace_worker(self, trace_context: TraceContext) -> None:
        """Background worker to process trace operations."""
        while trace_context.running or not trace_context.traces_queue.empty():
            trace_func, args = await trace_context.traces_queue.get()
            try:
                trace_func(*args)
            except Exception as e:
                logger.exception(f"Error processing trace_func: {e}")
            finally:
                trace_context.traces_queue.task_done()

    async def _start(self, trace_context: TraceContext) -> None:
        """Start the background worker for processing traces."""
        if trace_context.running or self.deactivated:
            return
        try:
            trace_context.running = True
            trace_context.worker_task = asyncio.create_task(self._trace_worker(trace_context))
        except Exception as e:
            logger.exception(f"Error starting tracing service: {e}")


    def _initialize_langwatch_tracer(self, trace_context: TraceContext) -> None:
        """Initialize LangWatch tracer if configured."""
        try:
            from agent_tracer.tracers.langwatch import LangWatchTracer

            if "langwatch" not in trace_context.tracers or trace_context.tracers["langwatch"].trace_id != trace_context.run_id:
                trace_context.tracers["langwatch"] = LangWatchTracer(
                    trace_name=trace_context.run_name,
                    trace_type="chain",
                    project_name=trace_context.project_name,
                    trace_id=trace_context.run_id,
                )
        except ImportError:
            logger.debug("LangWatch not available (install with: pip install agent-tracer[langwatch])")
        except Exception as e:
            logger.debug(f"Error initializing LangWatch tracer: {e}")

    def _initialize_langfuse_tracer(self, trace_context: TraceContext) -> None:
        """Initialize LangFuse tracer if configured."""
        try:
            from agent_tracer.tracers.langfuse import LangFuseTracer

            trace_context.tracers["langfuse"] = LangFuseTracer(
                trace_name=trace_context.run_name,
                trace_type="chain",
                project_name=trace_context.project_name,
                trace_id=trace_context.run_id,
                user_id=trace_context.user_id,
                session_id=trace_context.session_id,
            )
        except ImportError:
            logger.debug("LangFuse not available (install with: pip install agent-tracer[langfuse])")
        except Exception as e:
            logger.debug(f"Error initializing LangFuse tracer: {e}")


    def _initialize_opik_tracer(self, trace_context: TraceContext) -> None:
        """Initialize Opik tracer if configured."""
        try:
            from agent_tracer.tracers.opik import OpikTracer

            trace_context.tracers["opik"] = OpikTracer(
                trace_name=trace_context.run_name,
                trace_type="chain",
                project_name=trace_context.project_name,
                trace_id=trace_context.run_id,
                user_id=trace_context.user_id,
                session_id=trace_context.session_id,
            )
        except ImportError:
            logger.debug("Opik not available (install with: pip install agent-tracer[opik])")
        except Exception as e:
            logger.debug(f"Error initializing Opik tracer: {e}")


    def _initialize_console_tracer(self, trace_context: TraceContext) -> None:
        """Initialize Console tracer if enabled."""
        if not self.config.enable_console:
            return
        try:
            from agent_tracer.tracers.console import ConsoleTracer

            trace_context.tracers["console"] = ConsoleTracer(
                trace_name=trace_context.run_name,
                trace_type="chain",
                project_name=trace_context.project_name,
                trace_id=trace_context.run_id,
                user_id=trace_context.user_id,
                session_id=trace_context.session_id,
            )
        except Exception as e:
            logger.debug(f"Error initializing Console tracer: {e}")

    async def start_trace(
        self,
        trace_id: UUID | str | None = None,
        trace_name: str = "Agent Run",
        user_id: str | None = None,
        session_id: str | None = None,
        project_name: str | None = None,
    ) -> None:
        """Start a trace for an agent/workflow run.

        Args:
            trace_id: Unique identifier for the trace (auto-generated if not provided)
            trace_name: Name of the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
            project_name: Optional project name (default: from env or "AgentTracer")
        """
        if self.deactivated:
            return
        try:
            if trace_id is None:
                trace_id = uuid4()
            project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "AgentTracer")
            trace_context = TraceContext(trace_id, trace_name, project_name, user_id, session_id)
            trace_context_var.set(trace_context)
            await self._start(trace_context)

            # Initialize all available tracers
            self._initialize_console_tracer(trace_context)  # Initialize console tracer first
            self._initialize_langwatch_tracer(trace_context)
            self._initialize_langfuse_tracer(trace_context)
            self._initialize_opik_tracer(trace_context)
        except Exception as e:
            logger.debug(f"Error initializing tracers: {e}")

    async def _stop(self, trace_context: TraceContext) -> None:
        """Stop the background worker."""
        try:
            trace_context.running = False
            if not trace_context.traces_queue.empty():
                await trace_context.traces_queue.join()
            if trace_context.worker_task:
                trace_context.worker_task.cancel()
                trace_context.worker_task = None
        except Exception as e:
            logger.exception(f"Error stopping tracing service: {e}")

    def _end_all_tracers(self, trace_context: TraceContext, outputs: dict, error: Exception | None = None) -> None:
        """End all active tracers."""
        for tracer in trace_context.tracers.values():
            if tracer.ready:
                try:
                    tracer.end(
                        trace_context.all_inputs,
                        outputs=trace_context.all_outputs,
                        error=error,
                        metadata=outputs,
                    )
                except Exception as e:
                    logger.error(f"Error ending trace: {e}")

    async def end_trace(self, outputs: dict | None = None, error: Exception | None = None) -> None:
        """End the current trace.

        Args:
            outputs: Optional output data
            error: Optional error that occurred
        """
        if self.deactivated:
            return
        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("end_trace called but no trace context found")
            return
        await self._stop(trace_context)
        self._end_all_tracers(trace_context, outputs or {}, error)

    def _cleanup_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Clean up inputs by masking sensitive data."""
        if not self.config.mask_sensitive_data:
            return inputs.copy()
        return mask_sensitive_data(inputs.copy(), self.config.sensitive_keywords)

    def _start_step_traces(
        self,
        step_trace_context: StepTraceContext,
        trace_context: TraceContext,
    ) -> None:
        """Start traces for a step/component."""
        inputs = self._cleanup_inputs(step_trace_context.inputs)
        step_trace_context.inputs = inputs
        step_trace_context.inputs_metadata = step_trace_context.inputs_metadata or {}

        for tracer in trace_context.tracers.values():
            if not tracer.ready:
                continue
            try:
                tracer.add_trace(
                    step_trace_context.trace_id,
                    step_trace_context.trace_name,
                    step_trace_context.trace_type,
                    inputs,
                    step_trace_context.inputs_metadata,
                    step_trace_context.custom_data,
                    step_trace_context.parent_step_trace_context,
                )
            except Exception as e:
                logger.exception(f"Error starting trace {step_trace_context.trace_name}: {e}", exc_info=True)

    def _end_step_traces(
        self,
        step_trace_context: StepTraceContext,
        trace_context: TraceContext,
        error: Exception | None = None,
    ) -> None:
        """End traces for a step/component."""
        for tracer in trace_context.tracers.values():
            if tracer.ready:
                try:
                    tracer.end_trace(
                        trace_id=step_trace_context.trace_id,
                        trace_name=step_trace_context.trace_name,
                        outputs=trace_context.all_outputs[step_trace_context.trace_name],
                        error=error,
                        logs=step_trace_context.logs[step_trace_context.trace_name],
                    )
                except Exception as e:
                    logger.exception(f"Error ending trace {step_trace_context.trace_name}: {e}")

    @asynccontextmanager
    async def trace_step(
        self,
        step_name: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        trace_type: str = "tool",
        custom_data: dict[str, Any] | None = None,
        parent_step_trace_context: "StepTraceContext" | None = None,
    ) -> AsyncGenerator[StepTraceContext, None]:
        """Trace a step/component in the workflow.

        Args:
            step_name: Name of the step
            inputs: Input data for the step
            metadata: Optional metadata
            trace_type: Type of trace (default: "tool")
            custom_data: Optional custom data

        Yields:
            self: The tracing service instance

        Example:
            async with tracer.trace_step("data_processing", {"input": data}):
                result = process_data(data)
                tracer.set_outputs("data_processing", {"result": result})
        """
        if self.deactivated:
            yield self
            return

        trace_id = f"{step_name}-{uuid4()}"
        inputs = self._cleanup_inputs(inputs)
        step_trace_context = StepTraceContext(trace_id, step_name, trace_type, inputs, metadata, custom_data, parent_step_trace_context)
        step_context_var.set(step_trace_context)

        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("trace_step called but no trace context found")
            yield step_trace_context
            return

        trace_context.all_inputs[step_name] |= inputs or {}
        await trace_context.traces_queue.put((self._start_step_traces, (step_trace_context, trace_context)))

        try:
            yield step_trace_context
        except Exception as e:
            await trace_context.traces_queue.put((self._end_step_traces, (step_trace_context, trace_context, e)))
            raise
        else:
            await trace_context.traces_queue.put((self._end_step_traces, (step_trace_context, trace_context, None)))

    @property
    def project_name(self) -> str | None:
        """Get the current project name."""
        if self.deactivated:
            return os.getenv("LANGCHAIN_PROJECT", "AgentTracer")
        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("project_name accessed but no trace context found")
            return None
        return trace_context.project_name

    def add_log(self, step_name: str, log: Log | dict[str, Any]) -> None:
        """Add a log entry to the current step.

        Args:
            step_name: Name of the step
            log: Log entry (Log object or dict with name, message, type)
        """
        if self.deactivated:
            return
        step_context = step_context_var.get()
        if step_context is None:
            raise RuntimeError("add_log called but no step context found")
        step_context.logs[step_name].append(log)

    def set_outputs(
        self,
        step_name: str,
        outputs: dict[str, Any],
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set outputs for the current step.

        Args:
            step_name: Name of the step
            outputs: Output data
            output_metadata: Optional output metadata
        """
        if self.deactivated:
            return
        step_context = step_context_var.get()
        if step_context is None:
            raise RuntimeError("set_outputs called but no step context found")

        step_context.outputs[step_name] |= outputs or {}
        step_context.outputs_metadata[step_name] |= output_metadata or {}

        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("set_outputs called but no trace context found")
            return
        trace_context.all_outputs[step_name] |= outputs or {}

    def get_tracer(self, tracer_name: str) -> BaseTracer | None:
        """Get a specific tracer by name.

        Args:
            tracer_name: Name of the tracer (e.g., "langfuse", "langwatch")

        Returns:
            The tracer instance or None
        """
        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("get_tracer called but no trace context found")
            return None
        return trace_context.tracers.get(tracer_name)

    def get_langchain_callbacks(self) -> list[Any]:
        """Get LangChain-compatible callback handlers from all tracers.

        Returns:
            List of callback handlers
        """
        if self.deactivated:
            return []
        callbacks = []
        trace_context = trace_context_var.get()
        if trace_context is None:
            logger.warning("get_langchain_callbacks called but no trace context found")
            return []
        for tracer in trace_context.tracers.values():
            if not tracer.ready:
                continue
            langchain_callback = tracer.get_langchain_callback()
            if langchain_callback:
                callbacks.append(langchain_callback)
        return callbacks

