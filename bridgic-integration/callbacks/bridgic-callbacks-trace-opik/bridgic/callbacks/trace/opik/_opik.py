"""Opik tracing callback handler for Bridgic workers and automa."""

from contextvars import ContextVar
import time
import warnings
from typing import Any, Dict, Optional, TYPE_CHECKING

import opik.decorator.tracing_runtime_config as tracing_runtime_config
import opik.opik_context as opik_context
from opik import context_storage as opik_context_storage
from opik.api_objects import helpers, opik_client, span, trace
from opik.decorator import error_info_collector
from opik.types import ErrorInfoDict

from bridgic.core.automa.worker import WorkerCallback

from bridgic.core.utils._collection import serialize_data, merge_optional_dicts
from bridgic.core.utils._worker_context import get_worker_exec_context

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma, _GraphAdaptedWorker

span_context_var = ContextVar("span_context")

class OpikTraceCallback(WorkerCallback):
    """
    Opik tracing callback handler for Bridgic workers and automa.

    This callback handler integrates Opik tracing with Bridgic framework,
    providing step-level tracing for worker execution and automa orchestration.
    It tracks worker execution, creates spans for each worker, and manages
    trace lifecycle for top-level automa instances.

    Parameters
    ----------
    project_name : Optional[str]
        The project name for Opik tracing. If None, uses default project name.
    """

    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize the Opik callback handler.

        Parameters
        ----------
        project_name : Optional[str]
            The project name for Opik tracing. If None, uses default project name.
        """
        super().__init__()
        self._project_name = project_name
        self._opik_client = opik_client.Opik(
            _use_batching=True,
            project_name=project_name,
        )
        self._opik_trace_data: Optional[trace.TraceData] = None
        self._owns_trace_context: bool = False
        self._trace_start_time: Optional[float] = None
        self._trace_inputs: Optional[Dict[str, Any]] = None

        # Store worker execution state for tracking (using stack for nested workers)
        self._worker_state_stack: list[Dict[str, Any]] = []


    @staticmethod
    def _serialize_data(value: Any, depth: int = 5) -> Any:
        return serialize_data(value, depth)

    @staticmethod
    def _merge_metadata(
        current: Optional[Dict[str, Any]],
        updates: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return merge_optional_dicts(current, updates)

    def _get_worker_instance(
        self, key: str, parent: Optional["GraphAutoma"]
    ) -> "_GraphAdaptedWorker":
        if parent is None:
            raise ValueError("Parent automa is required to get worker instance")
        return parent._get_worker_instance(key)

    def _should_trace_as_automa(
        self, is_top_level: bool, worker: Optional["_GraphAdaptedWorker"] = None
    ) -> bool:
        if is_top_level:
            return True

        # Check if worker is decorated and wraps an automa
        if worker is not None:
            return worker.is_automa()

        return False

    def _create_trace_data(self, trace_name: Optional[str] = None) -> trace.TraceData:
        return trace.TraceData(
            name=trace_name,
            metadata={"created_from": "bridgic"},
            project_name=self._project_name,
        )

    def _start_trace(self, trace_name: Optional[str] = None) -> None:
        # Check if there's an existing trace in context
        self._owns_trace_context = False
        self._trace_inputs = None
        existing_trace_data: Optional[trace.TraceData] = opik_context_storage.get_trace_data()
        if existing_trace_data is not None:
            self._opik_trace_data = existing_trace_data
            # We don't own this trace, but we still want to measure our segment duration
            # for metadata purposes on end (without closing the trace).
            self._trace_start_time = time.time()
        else:
            self._opik_trace_data = self._create_trace_data(trace_name=trace_name)
            opik_context_storage.set_trace_data(self._opik_trace_data)
            self._owns_trace_context = True
            self._trace_start_time = time.time()

            if (
                self._opik_client.config.log_start_trace_span
                and tracing_runtime_config.is_tracing_active()
            ):
                self._opik_client.trace(
                    **self._opik_trace_data.as_start_parameters
                )

    def _end_trace(
        self,
        output: Optional[Dict[str, Any]] = None,
        error_info: Optional[ErrorInfoDict] = None,
    ) -> None:
        if self._opik_trace_data is None:
            return

        if not self._owns_trace_context:
            # We don't own the trace: update current trace fields, but do NOT end it.
            execution_duration_metadata = {}
            if self._trace_start_time is not None:
                end_time = time.time()
                execution_duration_metadata["execution_duration"] = end_time - self._trace_start_time
                execution_duration_metadata["end_time"] = end_time
            # Prefer explicit output; else try worker stored outputs
            effective_output = output
            if effective_output is None and self._worker_state_stack:
                # Get from the current worker state
                current_state = self._worker_state_stack[-1]
                if "metadata" in current_state:
                    effective_output = current_state["metadata"].get("outputs")
            if effective_output is not None:
                effective_output = serialize_data(effective_output)
            # Prefer stored root inputs if present
            effective_input = self._trace_inputs
            # Apply updates to the live trace in context
            try:
                opik_context.update_current_trace(
                    metadata=execution_duration_metadata if execution_duration_metadata else None,
                    input=effective_input,
                    output=effective_output,
                )
            except Exception:
                # Best-effort; ignore update failures
                pass
            # Clean up our local state only
            self._opik_trace_data = None
            self._trace_start_time = None
            self._trace_inputs = None
            return

        self._opik_trace_data.init_end_time()
        if self._trace_start_time is not None:
            end_time = self._opik_trace_data.end_time.timestamp()
            self._opik_trace_data.metadata = OpikTraceCallback._merge_metadata(
                self._opik_trace_data.metadata,
                {
                    "execution_duration": end_time - self._trace_start_time,
                    "end_time": end_time,
                },
            )


        root_input = None
        if self._trace_inputs is not None:
            root_input = self._trace_inputs

        if output is not None:
            self._opik_trace_data.update(output=output)
        elif self._worker_state_stack:
            current_state = self._worker_state_stack[-1]
            if "metadata" in current_state:
                result = current_state["metadata"].get("outputs")
                if result is not None:
                    self._opik_trace_data.update(output=result)

        if root_input is not None:
            self._opik_trace_data.input = root_input

        if error_info is not None:
            self._opik_trace_data.update(error_info=error_info)

        if tracing_runtime_config.is_tracing_active():
            self._opik_client.trace(**self._opik_trace_data.as_parameters)

        # Pop the trace data from the context if we own it
        opik_context_storage.pop_trace_data(
            ensure_id=self._opik_trace_data.id  # type: ignore[arg-type]
        )

        # Flush trace data to ensure it's sent to the backend
        self._flush()

        self._opik_trace_data = None
        self._owns_trace_context = False
        self._trace_start_time = None
        self._trace_inputs = None

    def _add_trace_step(
        self,
        step_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_step_trace_context: Optional[span.SpanData] = None,
    ) -> span.SpanData:
        if self._opik_trace_data is None:
            # If no trace exists, create one
            self._start_trace()

        parent_span_id = None
        if parent_step_trace_context is not None:
            parent_span_id = parent_step_trace_context.id

        project_name = helpers.resolve_child_span_project_name(
            parent_project_name=self._opik_trace_data.project_name
            if self._opik_trace_data
            else None,
            child_project_name=self._project_name,
            show_warning=True,
        )

        # Create span data
        span_data = span.SpanData(
            trace_id=self._opik_trace_data.id if self._opik_trace_data else None,
            name=step_name,
            parent_span_id=parent_span_id,
            type="tool",
            input=inputs,
            metadata=metadata,
            project_name=project_name,
        )

        # Log span start if tracing is active
        if (
            self._opik_client.config.log_start_trace_span
            and tracing_runtime_config.is_tracing_active()
        ):
            self._opik_client.span(**span_data.as_start_parameters)

        return span_data

    def _set_outputs(
        self,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Store outputs in the current worker state (top of stack)
        if self._worker_state_stack:
            current_state = self._worker_state_stack[-1]
            if "metadata" not in current_state:
                current_state["metadata"] = {}
            current_state["metadata"]["outputs"] = OpikTraceCallback._serialize_data(outputs)

    def _end_trace_step(self, step_trace_context: span.SpanData, worker_metadata: Optional[Dict[str, Any]] = None) -> None:
        if step_trace_context is None:
            return

        # Get stored outputs and metadata
        outputs = None
        if worker_metadata:
            outputs = worker_metadata.get("outputs")
            
            # Merge all worker_metadata into span metadata (including start_time, end_time, execution_duration)
            current_metadata = step_trace_context.metadata or {}
            # Update with all worker metadata fields
            for key, value in worker_metadata.items():
                # Skip 'outputs' as it goes to span.output, not metadata
                if key != "outputs":
                    current_metadata[key] = value
            step_trace_context.update(metadata=current_metadata)

        # Update span with outputs
        if outputs is not None:
            step_trace_context.update(output=outputs)

        # Initialize end time and log span
        step_trace_context.init_end_time()
        if tracing_runtime_config.is_tracing_active():
            self._opik_client.span(**step_trace_context.as_parameters)

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["GraphAutoma"] = None,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Hook invoked before worker execution.

        For top-level automa, initializes a new trace. For workers, creates
        a new span. Handles nested automa as workers by checking if the
        decorated worker is an automa instance.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level : bool
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma]
            Parent automa instance containing this worker. Will be None when is_top_level is True.
        arguments : Optional[Dict[str, Any]]
            Execution arguments with keys "args" and "kwargs".
        """
        # Get worker instance
        worker = None
        if not is_top_level and parent is not None:
            try:
                worker = self._get_worker_instance(key, parent)
            except (KeyError, ValueError) as e:
                warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
                return

        # Check if should trace as automa
        should_trace_as_automa = self._should_trace_as_automa(is_top_level, worker)
        # For top-level automa, start trace
        if is_top_level:
            trace_name = key or "top_level_automa"
            self._start_trace(trace_name=trace_name)
            # Assign trace context to the top-level automa
            span_context_var.set(self._opik_trace_data)
            if self._opik_trace_data is not None:
                serialize_arguments = OpikTraceCallback._serialize_data(arguments)
                metadata = {
                    "key": key,
                    "nesting_level": 0,
                    "start_time": self._opik_trace_data.start_time.timestamp(),
                }
                self._opik_trace_data.metadata = OpikTraceCallback._merge_metadata(
                    self._opik_trace_data.metadata,
                    metadata,
                )

                if serialize_arguments is not None:
                    self._opik_trace_data.input = serialize_arguments
                    self._trace_inputs = serialize_arguments
                else:
                    self._trace_inputs = None

            return

        # For workers, create a span
        if worker is None:
            return

        # Get execution context
        worker_context_info = get_worker_exec_context(worker)
        nested_automa_instance: Optional[GraphAutoma] = worker._decorated_worker if should_trace_as_automa else None
        key = worker_context_info.key or key

        # Prepare metadata
        serialize_arguments = OpikTraceCallback._serialize_data(arguments)

        # Get parent trace context if exists
        parent_step_trace_context = None
        if hasattr(worker, "parent") and worker.parent is not None:
            parent_step_trace_context = span_context_var.get()

        # Create trace step (span) first to get the start_time
        step_trace_context = self._add_trace_step(
            step_name=f"{key}  [{nested_automa_instance.name}]" if should_trace_as_automa and nested_automa_instance is not None else key,
            inputs=serialize_arguments,
            metadata={
                "key": key,
                "dependencies": worker_context_info.dependencies,
                "worker_context_info": worker_context_info.to_metadata_dict(),
            },
            parent_step_trace_context=parent_step_trace_context,
        )

        # Now build complete worker_metadata with start_time
        worker_metadata = {
            "key": key,
            "dependencies": worker_context_info.dependencies,
            "worker_context_info": worker_context_info.to_metadata_dict(),
            "start_time": step_trace_context.start_time.timestamp(),
        }

        # Assign trace context to worker for later use
        span_context_var.set(step_trace_context)

        # Store state for on_worker_end/on_worker_error using stack
        worker_state = {
            "key": key,
            "start_time": step_trace_context.start_time.timestamp(),
            "metadata": worker_metadata,
            "span_context": step_trace_context,
            "is_top_level": False,
        }
        self._worker_state_stack.append(worker_state)

    def _end_worker_span(
        self,
        outputs: Dict[str, Any],
        error: Optional[Exception] = None,
    ) -> None:
        # Pop worker state from stack
        if not self._worker_state_stack:
            warnings.warn("Worker state stack is empty when ending worker span")
            return
        
        worker_state = self._worker_state_stack.pop()
        
        # Calculate execution time
        execution_duration = None
        start_time = worker_state.get("start_time")
        if start_time is not None:
            end_time = time.time()
            execution_duration = end_time - start_time
            worker_state["metadata"]["end_time"] = end_time
            worker_state["metadata"]["execution_duration"] = execution_duration

        # Store outputs in metadata
        worker_state["metadata"]["outputs"] = OpikTraceCallback._serialize_data(outputs)

        # Get span context from worker state
        span_context = worker_state.get("span_context")
        if span_context is not None:
            if error is not None:
                error_info = error_info_collector.collect(error)
                if error_info is not None:
                    span_context.update(error_info=error_info)
            # End trace step
            self._end_trace_step(span_context, worker_metadata=worker_state["metadata"])
            
            # Restore parent span context if exists
            if self._worker_state_stack:
                parent_state = self._worker_state_stack[-1]
                span_context_var.set(parent_state.get("span_context"))
            else:
                span_context_var.set(None)

    def _end_top_level_trace(
        self,
        output: Dict[str, Any],
        execution_status: str = "completed",
        error_info: Optional[ErrorInfoDict] = None,
    ) -> None:
        if self._opik_trace_data is not None:
            self._opik_trace_data.metadata = OpikTraceCallback._merge_metadata(
                self._opik_trace_data.metadata,
                {"execution_status": execution_status},
            )
            self._opik_trace_data.output = output
        self._end_trace(output=output, error_info=error_info)

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["GraphAutoma"] = None,
        arguments: Optional[Dict[str, Any]] = None,
        result: Any = None,
    ) -> None:
        """
        Hook invoked after worker execution.

        For top-level automa, ends the trace. For workers, ends the span
        with execution results.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level : bool
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma]
            Parent automa instance containing this worker. Will be None when is_top_level is True.
        arguments : Optional[Dict[str, Any]]
            Execution arguments with keys "args" and "kwargs".
        result : Any
            Worker execution result.
        """
        outputs = {
            "result_type": type(result).__name__ if result is not None else None,
            "result": OpikTraceCallback._serialize_data(result),
        }
        # For top-level automa, end trace
        if is_top_level:
            self._end_top_level_trace(output=outputs, execution_status="completed")
            return
        # End worker span
        self._end_worker_span(outputs=outputs)

    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["GraphAutoma"] = None,
        arguments: Optional[Dict[str, Any]] = None,
        error: Exception = None,
    ) -> bool:
        """
        Hook invoked when worker execution raises an exception.

        For top-level automa, ends the trace with error information.
        For workers, ends the span with error information.

        Parameters
        ----------
        key : str
            Worker identifier.
        is_top_level : bool
            Whether the worker is the top-level automa. When True, parent will be None.
        parent : Optional[GraphAutoma]
            Parent automa instance containing this worker. Will be None when is_top_level is True.
        arguments : Optional[Dict[str, Any]]
            Execution arguments with keys "args" and "kwargs".
        error : Exception
            The exception raised during worker execution.

        Returns
        -------
        bool
            Always returns False, indicating the exception should not be suppressed.
        """
        # For top-level automa, end trace with error
        if is_top_level:
            error_info = error_info_collector.collect(error) if error else None
            serialize_output = {
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
            }
            self._end_top_level_trace(
                output=serialize_output,
                execution_status="failed",
                error_info=error_info,
            )
            return False

        # Get worker instance
        worker = None
        if parent is not None:
            try:
                worker = self._get_worker_instance(key, parent)
            except (KeyError, ValueError) as e:
                warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
                return False

        # Prepare error outputs
        outputs = {
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
        }

        # End worker span with error
        self._end_worker_span(outputs=outputs, error=error)

        return False

    def _flush(self) -> None:
        self._opik_client.flush()

