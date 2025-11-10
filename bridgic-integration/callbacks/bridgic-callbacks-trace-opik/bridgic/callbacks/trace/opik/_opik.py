"""Opik tracing callback handler for Bridgic workers and automa."""

from contextvars import ContextVar
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, TYPE_CHECKING

import opik.decorator.tracing_runtime_config as tracing_runtime_config
import opik.opik_context as opik_context
from opik import context_storage as opik_context_storage
from opik.api_objects import opik_client, span, trace
from opik.decorator import error_info_collector
from opik.types import ErrorInfoDict

from bridgic.core.automa.worker import WorkerCallback
from bridgic.core.types._worker_types import WorkerExecutionContext

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma
    from bridgic.core.automa.worker import Worker

LOGGER = logging.getLogger(__name__)

span_context_var = ContextVar("span_context",)

class OpikCallbackHandler(WorkerCallback):
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

    # Singleton instances keyed by project_name
    _instances: Dict[Optional[str], "OpikCallbackHandler"] = {}
    _instances_initialized: Dict[Optional[str], bool] = {}

    def __new__(cls, project_name: Optional[str] = None):
        """
        Create or return a singleton instance for the given project_name.
        """
        key = project_name
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        cls._instances_initialized[key] = False
        return instance

    def __init__(self, project_name: Optional[str] = None):
        """
        Initialize the Opik callback handler.

        Parameters
        ----------
        project_name : Optional[str]
            The project name for Opik tracing. If None, uses default project name.
        """
        # Ensure per-project singleton is only initialized once
        key = project_name
        if self.__class__._instances_initialized.get(key, False):
            return

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

        # Store worker execution state for tracking
        self._worker_start_time: Optional[float] = None
        self._worker_metadata: Optional[Dict[str, Any]] = None
        self._worker_context_info: Optional[WorkerExecutionContext] = None

        # Mark singleton initialized for this project key
        self.__class__._instances_initialized[key] = True


    @staticmethod
    def _sanitize_data(value: Any, depth: int = 5) -> Any:
        """
        Convert data into a structure that can be serialized by Opik.

        Parameters
        ----------
        value : Any
            The value to sanitize.
        depth : int
            Maximum recursive depth to avoid infinite recursion.

        Returns
        -------
        Any
            A sanitized value suitable for serialization.
        """
        if depth <= 0:
            return repr(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): OpikCallbackHandler._sanitize_data(val, depth - 1)
                for key, val in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                OpikCallbackHandler._sanitize_data(item, depth - 1)
                for item in value
            ]

        return repr(value)

    @staticmethod
    def _merge_metadata(
        current: Optional[Dict[str, Any]],
        updates: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Merge metadata dictionaries, removing None values.

        Parameters
        ----------
        current : Optional[Dict[str, Any]]
            Existing metadata.
        updates : Optional[Dict[str, Any]]
            Metadata to merge into current.

        Returns
        -------
        Optional[Dict[str, Any]]
            The merged metadata dictionary or None if both inputs are empty.
        """
        if not current and not updates:
            return None

        merged: Dict[str, Any] = dict(current or {})
        if updates:
            for key, value in updates.items():
                if value is None:
                    continue
                merged[key] = value

        return merged

    @staticmethod
    def _sanitize_data(value: Any, depth: int = 5) -> Any:
        """
        Convert data into a structure that can be serialized by Opik.

        Parameters
        ----------
        value : Any
            The value to sanitize.
        depth : int
            Maximum recursive depth to avoid infinite recursion.

        Returns
        -------
        Any
            A sanitized value suitable for serialization.
        """
        if depth <= 0:
            return repr(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Mapping):
            return {
                str(key): OpikCallbackHandler._sanitize_data(val, depth - 1)
                for key, val in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                OpikCallbackHandler._sanitize_data(item, depth - 1)
                for item in value
            ]

        return repr(value)

    def _get_worker_instance(
        self, key: str, parent: Optional["GraphAutoma"]
    ) -> "Worker":
        """
        Get the worker instance by key from parent automa.

        Parameters
        ----------
        key : str
            Worker key identifier.
        parent : Optional[GraphAutoma]
            Parent automa instance containing the worker.

        Returns
        -------
        Worker
            The worker instance.
        """
        if parent is None:
            raise ValueError("Parent automa is required to get worker instance")
        return parent._get_worker_instance(key)

    def should_trace_as_automa(
        self, is_top_level: bool, worker: Optional["Worker"] = None
    ) -> bool:
        """
        Determine whether to trace a worker execution.

        For top-level automa, always trace. For nested workers, only trace
        if the worker is not itself an automa instance (to avoid double tracing).

        Parameters
        ----------
        is_top_level : bool
            Whether this is a top-level automa execution.
        worker : Optional[Worker]
            The worker instance. If None, assumes should trace.

        Returns
        -------
        bool
            True if should trace, False otherwise.
        """
        if is_top_level:
            return True

        # Check if worker is decorated and wraps an automa
        if hasattr(worker, "_decorated_worker_is_automa"):
            decorated_worker_is_automa = worker._decorated_worker_is_automa
            # If the decorated worker is an automa, don't trace at this level
            # (it will be traced when the automa itself runs)
            return decorated_worker_is_automa

        return False

    def _create_trace_data(self, trace_name: Optional[str] = None) -> trace.TraceData:
        """
        Create a new trace data instance.

        Parameters
        ----------
        trace_name : Optional[str]
            Name for the trace. If None, uses default.

        Returns
        -------
        trace.TraceData
            The created trace data instance.
        """
        trace_data = trace.TraceData(
            name=trace_name,
            metadata={"created_from": "bridgic"},
            project_name=self._project_name,
        )

        return trace_data

    def start_trace(self, trace_name: Optional[str] = None) -> None:
        """
        Start a new trace for top-level automa execution.

        Parameters
        ----------
        trace_name : Optional[str]
            Name for the trace. If None, uses default.
        """
        # Check if there's an existing trace in context
        self._owns_trace_context = False
        self._trace_inputs = None
        existing_trace_data: Optional[trace.TraceData] = opik_context_storage.get_trace_data()
        if existing_trace_data is not None:
            self._opik_trace_data = existing_trace_data
            # We don't own this trace, but we still want to measure our segment duration
            # for metadata purposes on end (without closing the trace).
            self._trace_start_time = time.time()
            self._trace_inputs = None
        else:
            self._opik_trace_data = self._create_trace_data(trace_name=trace_name)
            opik_context_storage.set_trace_data(self._opik_trace_data)
            self._owns_trace_context = True
            self._trace_start_time = time.time()
            self._trace_inputs = None

            if (
                self._opik_client.config.log_start_trace_span
                and tracing_runtime_config.is_tracing_active()
            ):
                self._opik_client.trace(
                    **self._opik_trace_data.as_start_parameters
                )

    def end_trace(
        self,
        output: Optional[Dict[str, Any]] = None,
        error_info: Optional[ErrorInfoDict] = None,
    ) -> None:
        """
        End the current trace.

        Parameters
        ----------
        output : Optional[Dict[str, Any]]
            Final output of the trace.
        error_info : Optional[ErrorInfoDict]
            Error information if trace ended with an error.
        """
        if self._opik_trace_data is None:
            return

        if not self._owns_trace_context:
            # We don't own the trace: update current trace fields, but do NOT end it.
            execution_time_metadata = {}
            if self._trace_start_time is not None:
                execution_time_metadata["execution_time"] = time.time() - self._trace_start_time
            # Prefer explicit output; else try worker stored outputs
            effective_output = output
            if effective_output is None and self._worker_metadata is not None:
                effective_output = self._worker_metadata.get("outputs")
            if effective_output is not None:
                effective_output = OpikCallbackHandler._sanitize_data(effective_output)
            # Prefer stored root inputs if present
            effective_input = self._trace_inputs
            if effective_input is None and self._worker_metadata is not None:
                effective_input = self._worker_metadata.get("inputs")
            # Apply updates to the live trace in context
            try:
                opik_context.update_current_trace(
                    metadata=execution_time_metadata if execution_time_metadata else None,
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
        execution_time_metadata = {}
        if self._trace_start_time is not None:
            execution_time_metadata["execution_time"] = time.time() - self._trace_start_time
        if execution_time_metadata:
            self._opik_trace_data.metadata = OpikCallbackHandler._merge_metadata(
                self._opik_trace_data.metadata,
                execution_time_metadata,
            )

        root_input = None
        if self._trace_inputs is not None:
            root_input = self._trace_inputs
        elif self._worker_metadata is not None:
            # Fallback to worker metadata inputs if available
            root_input = self._worker_metadata.get("inputs")

        if output is not None:
            self._opik_trace_data.update(output=output)
        elif self._worker_metadata is not None:
            result = self._worker_metadata.get("outputs")
            if result is not None:
                self._opik_trace_data.update(output=result)

        if root_input is not None:
            self._opik_trace_data.input = root_input

        if output is not None:
            self._opik_trace_data.output = output
        if error_info is not None:
            self._opik_trace_data.update(error_info=error_info)

        if tracing_runtime_config.is_tracing_active():
            self._opik_client.trace(**self._opik_trace_data.as_parameters)

        # Pop the trace data from the context if we own it
        opik_context_storage.pop_trace_data(
            ensure_id=self._opik_trace_data.id  # type: ignore[arg-type]
        )

        # Flush trace data to ensure it's sent to the backend
        self.flush()

        self._opik_trace_data = None
        self._owns_trace_context = False
        self._trace_start_time = None
        self._trace_inputs = None

    def add_trace_step(
        self,
        step_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_step_trace_context: Optional[span.SpanData] = None,
    ) -> span.SpanData:
        """
        Create a new trace step (span) for worker execution.

        Parameters
        ----------
        step_name : str
            Name of the step/worker.
        inputs : Optional[Dict[str, Any]]
            Input arguments for the worker.
        metadata : Optional[Dict[str, Any]]
            Additional metadata for the span.
        parent_step_trace_context : Optional[span.SpanData]
            Parent span context if this is a nested worker.

        Returns
        -------
        span.SpanData
            The created span data instance.
        """
        if self._opik_trace_data is None:
            # If no trace exists, create one
            self.start_trace()

        parent_span_id = None
        if parent_step_trace_context is not None:
            parent_span_id = parent_step_trace_context.id

        # Resolve project name for child span
        from .api_objects import helpers

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
            type="general",
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

    def set_outputs(
        self,
        step_name: str,
        outputs: Optional[Dict[str, Any]] = None,
        output_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set outputs for a trace step. This is a helper method that stores
        output information to be used when ending the step.

        Note: This method doesn't directly update the span. The actual update
        happens in end_trace_step.

        Parameters
        ----------
        step_name : str
            Name of the step/worker.
        outputs : Optional[Dict[str, Any]]
            Output data from the worker execution.
        output_metadata : Optional[Dict[str, Any]]
            Additional metadata for the output.
        """
        # Store outputs in metadata for later use
        if self._worker_metadata is None:
            self._worker_metadata = {}
        self._worker_metadata["outputs"] = (
            OpikCallbackHandler._sanitize_data(outputs)
            if outputs is not None
            else outputs
        )
        self._worker_metadata["output_metadata"] = output_metadata

    def end_trace_step(self, step_trace_context: span.SpanData) -> None:
        """
        End a trace step (span).

        Parameters
        ----------
        step_trace_context : span.SpanData
            The span data instance to end.
        """
        if step_trace_context is None:
            return

        # Get stored outputs and metadata
        outputs = None
        output_metadata = None
        if self._worker_metadata:
            outputs = self._worker_metadata.get("outputs")
            output_metadata = self._worker_metadata.get("output_metadata")

        # Update span with outputs and metadata
        if outputs is not None:
            step_trace_context.update(output=outputs)
        if output_metadata is not None:
            # Merge output_metadata into span metadata
            current_metadata = step_trace_context.metadata or {}
            current_metadata.update(output_metadata)
            step_trace_context.update(metadata=current_metadata)

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
                LOGGER.warning(f"Failed to get worker instance for key '{key}': {e}")
                return

        # Check if should trace as automa
        should_trace_as_automa = self.should_trace_as_automa(is_top_level, worker)
        # For top-level automa, start trace
        if is_top_level:
            trace_name = f"automa_{key}" if key else "top_level_automa"
            self.start_trace(trace_name=trace_name)
            # Assign trace context to the top-level automa
            span_context_var.set(self._opik_trace_data)
            if self._opik_trace_data is not None:
                execution_id = f"trace_{uuid.uuid4().hex[:16]}"
                sanitized_arguments = (
                    OpikCallbackHandler._sanitize_data(arguments)
                    if arguments is not None
                    else None
                )
                metadata = {
                    "execution_id": execution_id,
                    "automa_key": key,
                    "automa_type": "top_level",
                }
                self._opik_trace_data.metadata = OpikCallbackHandler._merge_metadata(
                    self._opik_trace_data.metadata,
                    metadata,
                )

                if sanitized_arguments is not None:
                    self._opik_trace_data.input = sanitized_arguments
                    self._trace_inputs = sanitized_arguments
                else:
                    self._trace_inputs = None

            return

        # For workers, create a span
        if worker is None:
            return

        # Get execution context
        worker_context_info = worker._get_execution_context()
        worker_key = (worker._decorated_worker.name if should_trace_as_automa else worker_context_info.worker_key) or key

        # Generate execution ID
        execution_id = f"exec_{uuid.uuid4().hex[:16]}"

        # Prepare metadata
        sanitized_arguments = OpikCallbackHandler._sanitize_data(arguments)
        metadata = {
            "worker_id": worker._unique_id if hasattr(worker, "_unique_id") else None,
            "worker_key": worker_key,
            "dependencies": worker_context_info.dependencies,
            "worker_context_info": worker_context_info.to_metadata_dict(),
            "execution_id": execution_id,
            "inputs": sanitized_arguments,
        }

        # Get parent trace context if exists
        parent_step_trace_context = None
        if hasattr(worker, "parent") and worker.parent is not None:
            parent_step_trace_context = span_context_var.get()

        # Create trace step (span)
        step_trace_context = self.add_trace_step(
            step_name=worker_key,
            inputs=arguments,
            metadata=metadata,
            parent_step_trace_context=parent_step_trace_context,
        )

        # Assign trace context to worker for later use
        span_context_var.set(step_trace_context)
        # Store root trace input if this is top-level
        if is_top_level:
            self._trace_inputs = arguments

        # Store state for on_worker_end/on_worker_error
        self._worker_start_time = time.time()
        self._worker_metadata = metadata
        self._worker_context_info = worker_context_info

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
        # For top-level automa, end trace
        if is_top_level:
            sanitized_output = {
                "result_type": type(result).__name__ if result is not None else None,
                "result": OpikCallbackHandler._sanitize_data(result),
            }
            if self._opik_trace_data is not None:
                self._opik_trace_data.metadata = OpikCallbackHandler._merge_metadata(
                    self._opik_trace_data.metadata,
                    {"execution_status": "completed"},
                )
                self._opik_trace_data.output = sanitized_output
            self.end_trace(output=sanitized_output)
            return

        # Get worker instance
        worker = None
        if parent is not None:
            try:
                worker = self._get_worker_instance(key, parent)
            except (KeyError, ValueError) as e:
                LOGGER.warning(f"Failed to get worker instance for key '{key}': {e}")
                return

        # Check if should trace
        should_trace_as_automa = self.should_trace_as_automa(False, worker)

        # Calculate execution time
        execution_time = None
        if self._worker_start_time is not None:
            execution_time = time.time() - self._worker_start_time

        # Prepare outputs
        outputs = {
            "result_type": type(result).__name__ if result is not None else None,
            "result": result,
        }

        # Prepare output metadata
        output_metadata = {}
        if execution_time is not None:
            output_metadata["execution_time"] = execution_time
        if self._worker_metadata:
            output_metadata.update(self._worker_metadata)

        # Set outputs
        self.set_outputs(
            step_name=self._worker_metadata.get("worker_key", key) if self._worker_metadata else key,
            outputs=outputs,
            output_metadata=output_metadata,
        )

        # End trace step
        span_context = span_context_var.get()
        if span_context is not None:
            self.end_trace_step(span_context)
            span_context_var.set(None)

        # Clean up state
        self._worker_start_time = None
        self._worker_metadata = None
        self._worker_context_info = None

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
            sanitized_output = {
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
            }
            if self._opik_trace_data is not None:
                self._opik_trace_data.metadata = OpikCallbackHandler._merge_metadata(
                    self._opik_trace_data.metadata,
                    {"execution_status": "failed"},
                )
                self._opik_trace_data.output = sanitized_output
            self.end_trace(output=sanitized_output, error_info=error_info)
            return False

        # Get worker instance
        worker = None
        if parent is not None:
            try:
                worker = self._get_worker_instance(key, parent)
            except (KeyError, ValueError) as e:
                LOGGER.warning(f"Failed to get worker instance for key '{key}': {e}")
                return False

        # Check if should trace
        should_trace_as_automa = self.should_trace_as_automa(False, worker)

        # Calculate execution time
        execution_time = None
        if self._worker_start_time is not None:
            execution_time = time.time() - self._worker_start_time

        # Prepare error outputs
        outputs = {
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
        }

        # Prepare output metadata
        output_metadata = {}
        if execution_time is not None:
            output_metadata["execution_time"] = execution_time
        if self._worker_metadata:
            output_metadata.update(self._worker_metadata)

        # Set outputs
        self.set_outputs(
            step_name=self._worker_metadata.get("worker_key", key) if self._worker_metadata else key,
            outputs=outputs,
            output_metadata=output_metadata,
        )

        # Collect error info and update span
        error_info = error_info_collector.collect(error) if error else None
        if error_info is not None:
            span_context = span_context_var.get()
            if span_context is not None:
                span_context.update(error_info=error_info)

        # End trace step
        span_context = span_context_var.get()
        if span_context is not None:
            self.end_trace_step(span_context)
            span_context_var.set(None)

        # Clean up state
        self._worker_start_time = None
        self._worker_metadata = None
        self._worker_context_info = None

        return False

    def flush(self) -> None:
        """
        Send pending Opik data to the backend.

        This method should be called to ensure all trace data is sent
        before the application exits.
        """
        self._opik_client.flush()

