"""Opik tracing callback handler for Bridgic."""

import time
from bridgic.core.automa import Automa
from typing_extensions import override
import warnings
from typing import Any, Dict, Optional

import opik.decorator.tracing_runtime_config as tracing_runtime_config
from opik import context_storage as opik_context_storage
from opik.api_objects import helpers, opik_client, span, trace
from opik.decorator import error_info_collector
from opik.types import ErrorInfoDict

from bridgic.core.automa.worker import WorkerCallback, Worker
from bridgic.core.utils._collection import serialize_data, merge_optional_dicts
from bridgic.core.utils._worker_tracing import build_worker_tracing_dict, get_worker_tracing_step_name

class OpikTraceCallback(WorkerCallback):
    """
    Opik tracing callback handler for Bridgic.

    This callback handler integrates Opik tracing with Bridgic framework,
    providing step-level tracing for worker execution and automa orchestration.
    It tracks worker execution, creates spans for each worker, and manages
    trace lifecycle for top-level automa instances.

    **Configuration Scope**

    This callback requires access to the automa context and can only be configured
    at the **Automa level** (via `RunningOptions`) or **Global level** (via `GlobalSetting`).
    It does not support worker-level configuration (via `@worker` decorator).

    Parameters
    ----------
    project_name : Optional[str], default=None
        The project name for Opik tracing. If None, uses default project name.
    """

    _project_name: Optional[str]
    _is_ready: bool
    _opik_client: opik_client.Opik

    def __init__(self, project_name: Optional[str] = None):
        super().__init__()
        self._project_name = project_name
        self._is_ready = False
        self._setup_opik()

    def _setup_opik(self) -> None:
        self._opik_client = opik_client.Opik(_use_batching=True, project_name=self._project_name)
        missing_configuration, _ = self._opik_client._config.get_misconfiguration_detection_results()
        if missing_configuration:
            self._is_ready = False # for serialization compatibility
            return
        self._check_opik_auth()
    
    def _check_opik_auth(self) -> None:
        try:
            self._opik_client.auth_check()
        except Exception as e:
            self._is_ready = False # for serialization compatibility
            warnings.warn(f"Opik auth check failed, OpikTracer will be disabled: {e}")
        else:
            self._is_ready = True
    
    def _get_worker_instance(self, key: str, parent: Optional["Automa"]) -> Worker:
        """
        Get worker instance from parent automa.
        
        Returns
        -------
        Worker
            The worker instance.
        """
        if parent is None:
            raise ValueError("Parent automa is required to get worker instance")
        return parent._get_worker_instance(key)

    def _create_trace_data(self, trace_name: Optional[str] = None) -> trace.TraceData:
        return trace.TraceData(
            name=trace_name, 
            metadata={"created_from": "bridgic"}, 
            project_name=self._project_name
        )

    def _log_if_active(self, log_func, **params) -> None:
        """Log to Opik if tracing is active."""
        if tracing_runtime_config.is_tracing_active():
            log_func(**params)

    def _get_or_create_trace_data(self, trace_name: Optional[str] = None) -> trace.TraceData:
        """Initialize or reuse existing trace."""
        existing_trace = opik_context_storage.get_trace_data()
        if existing_trace:
            return existing_trace
        
        # Create new trace and set in context
        trace_data = self._create_trace_data(trace_name)
        opik_context_storage.set_trace_data(trace_data)
        
        if self._opik_client.config.log_start_trace_span:
            self._log_if_active(self._opik_client.trace, **trace_data.as_start_parameters)
        return trace_data

    def _complete_trace(self, output: Optional[Dict[str, Any]], error_info: Optional[ErrorInfoDict]) -> None:
        """Finalize and log trace we own."""
        trace_data = opik_context_storage.get_trace_data()
        if trace_data is None:
            return
            
        trace_data.init_end_time()
        
        # Compute execution duration from trace start_time
        if trace_data.start_time:
            end_time = trace_data.end_time.timestamp() if trace_data.end_time else time.time()
            start_time = trace_data.start_time.timestamp()
            trace_data.metadata = merge_optional_dicts(
                trace_data.metadata,
                {"execution_duration": end_time - start_time, "end_time": end_time}
            )

        if output:
            trace_data.update(output=output)

        if error_info:
            trace_data.update(error_info=error_info)

        self._log_if_active(self._opik_client.trace, **trace_data.as_parameters)
        opik_context_storage.pop_trace_data(ensure_id=trace_data.id)
        self._flush()

    def _start_span(
        self,
        step_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start a span for a worker execution step and push it to context."""
        trace_data = opik_context_storage.get_trace_data()

        parent_span = opik_context_storage.top_span_data()

        project_name = helpers.resolve_child_span_project_name(
            parent_project_name=trace_data.project_name,
            child_project_name=self._project_name,
            show_warning=True,
        )

        span_data = span.SpanData(
            trace_id=trace_data.id,
            name=step_name,
            parent_span_id=parent_span.id if parent_span else None,
            type="tool",
            input=inputs,
            metadata=metadata,
            project_name=project_name,
        )
        # Store start_time in metadata for later duration calculation
        if span_data.start_time and metadata is not None:
            metadata["start_time"] = span_data.start_time.timestamp()
            span_data.update(metadata=metadata)
        # Add span to context stack
        opik_context_storage.add_span_data(span_data)

        if self._opik_client.config.log_start_trace_span:
            self._log_if_active(self._opik_client.span, **span_data.as_start_parameters)

    def _finish_span(self, span_data: span.SpanData, worker_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Finish a worker span with metadata and output, then pop from context."""
        if worker_metadata:
            output = worker_metadata.get("output")
            # Merge all metadata except 'output' into span metadata
            current_metadata = span_data.metadata or {}
            current_metadata.update({k: v for k, v in worker_metadata.items() if k != "output"})
            span_data.update(metadata=current_metadata)
            
            if output is not None:
                span_data.update(output=output)

        span_data.init_end_time()
        self._log_if_active(self._opik_client.span, **span_data.as_parameters)
        
        # Pop span from context stack
        opik_context_storage.pop_span_data(ensure_id=span_data.id)

    def _start_top_level_trace(self, key: str, arguments: Optional[Dict[str, Any]]) -> None:
        """Start trace initialization for top-level automa."""
        trace_data = self._get_or_create_trace_data(trace_name=key or "top_level_automa")
        
        serialized_args = serialize_data(arguments)
        metadata_updates = {"key": key, "nesting_level": 0}
        if trace_data.start_time:
            metadata_updates["start_time"] = trace_data.start_time.timestamp()
        
        trace_data.metadata = merge_optional_dicts(trace_data.metadata, metadata_updates)
        
        if serialized_args:
            trace_data.input = serialized_args

    def _start_worker_span(self, key: str, worker: Worker, parent: "Automa", arguments: Optional[Dict[str, Any]]) -> None:
        """Start a span for worker execution."""
        step_name = get_worker_tracing_step_name(key, worker)
        worker_tracing_dict = build_worker_tracing_dict(worker, parent)
        self._start_span(
            step_name=step_name,
            inputs=serialize_data(arguments),
            metadata=worker_tracing_dict,
        )

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
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
        is_top_level : bool, default=False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa], default=None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Optional[Dict[str, Any]], default=None
            Execution arguments with keys "args" and "kwargs".
        """
        if not self._is_ready:
            return
        if is_top_level:
            self._start_top_level_trace(key, arguments)
            return

        try:
            worker = self._get_worker_instance(key, parent)
        except (KeyError, ValueError) as e:
            warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
            return

        self._start_worker_span(key, worker, parent, arguments)

    def _finish_current_span(self, output: Dict[str, Any], error: Optional[Exception] = None) -> None:
        """Finish the current span and pop it from context."""
        current_span = opik_context_storage.top_span_data()
        if not current_span:
            warnings.warn("No span found in context when finishing worker span")
            return
        
        # Calculate execution timing
        end_time = time.time()
        start_time = current_span.start_time.timestamp() if current_span.start_time else end_time
        
        # Build worker metadata with timing and output
        worker_metadata = {
            "end_time": end_time,
            "execution_duration": end_time - start_time,
            "output": serialize_data(output),
        }
        
        # Handle error if present
        if error:
            error_info = error_info_collector.collect(error)
            if error_info:
                current_span.update(error_info=error_info)
        
        # Finish the span (this will merge metadata and pop from context)
        self._finish_span(current_span, worker_metadata=worker_metadata)

    def _build_output_payload(self, result: Any = None, error: Optional[Exception] = None) -> Dict[str, Any]:
        """Build a standardized output dictionary for results or errors."""
        if error:
            return {"error_type": type(error).__name__, "error_message": str(error)}
        return {
            "result_type": type(result).__name__ if result is not None else None,
            "result": serialize_data(result),
        }

    def _complete_worker_execution(self, output: Dict[str, Any], is_top_level: bool, error: Optional[Exception] = None) -> None:
        """Complete worker or trace execution."""
        if is_top_level:
            trace_data = opik_context_storage.get_trace_data()
            if trace_data:
                execution_status = "failed" if error else "completed"
                trace_data.metadata = merge_optional_dicts(
                    trace_data.metadata, {"execution_status": execution_status}
                )
            
            error_info = error_info_collector.collect(error) if error else None
            self._complete_trace(output, error_info)
        else:
            self._finish_current_span(output=output, error=error)

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
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
        is_top_level : bool, default=False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa], default=None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Optional[Dict[str, Any]], default=None
            Execution arguments with keys "args" and "kwargs".
        result : Any, default=None
            Worker execution result.
        """
        if not self._is_ready:
            return
        output = self._build_output_payload(result=result)
        self._complete_worker_execution(output, is_top_level)

    async def on_worker_error(
        self,
        key: str,
        is_top_level: bool = False,
        parent: Optional["Automa"] = None,
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
        is_top_level : bool, default=False
            Whether the worker is the top-level automa. When True, parent will be the automa itself (parent is self).
        parent : Optional[Automa], default=None
            Parent automa instance containing this worker. For top-level automa, parent is the automa itself.
        arguments : Optional[Dict[str, Any]], default=None
            Execution arguments with keys "args" and "kwargs".
        error : Exception, default=None
            The exception raised during worker execution.

        Returns
        -------
        bool
            Always returns False, indicating the exception should not be suppressed.
        """
        if not self._is_ready:
            return False
        if not is_top_level and parent:
            try:
                self._get_worker_instance(key, parent)
            except (KeyError, ValueError) as e:
                warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
                return False

        output = self._build_output_payload(error=error)
        self._complete_worker_execution(output, is_top_level, error=error)
        return False

    def _flush(self) -> None:
        self._opik_client.flush()

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["project_name"] = self._project_name
        state_dict["is_ready"] = self._is_ready
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._project_name = state_dict["project_name"]
        self._is_ready = state_dict["is_ready"]
        self._setup_opik() # if opik is not ready, it will be set to False
