"""LangWatch tracing callback handler for Bridgic."""

import json
import warnings
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import langwatch
from langwatch.state import get_instance
from langwatch.types import BaseAttributes
from langwatch.telemetry.span import LangWatchSpan
from langwatch.telemetry.tracing import LangWatchTrace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from langwatch.domain import BaseAttributes, SpanProcessingExcludeRule

from bridgic.core.automa.worker import Worker, WorkerCallback
from bridgic.core.utils._collection import serialize_data
from bridgic.core.utils._worker_tracing import (
    build_worker_tracing_dict,
    get_worker_tracing_step_name,
)

# import logging

# logging.getLogger("langwatch.client").setLevel(logging.WARNING)


if TYPE_CHECKING:
    from bridgic.core.automa import Automa

class LangWatchTraceCallback(WorkerCallback):
    """
    LangWatch tracing callback handler for Bridgic.

    This callback handler integrates LangWatch tracing with Bridgic framework,
    providing step-level tracing for worker execution and automa orchestration.
    It tracks worker execution, creates spans for each worker, and manages
    trace lifecycle for top-level automa instances.

    Parameters
    ----------
    api_key : Optional[str], default=None
        The API key for the LangWatch tracing service, if none is provided, 
        the `LANGWATCH_API_KEY` environment variable will be used.
    endpoint_url : Optional[str], default=None
        The URL of the LangWatch tracing service, if none is provided, 
        the `LANGWATCH_ENDPOINT` environment variable will be used. If that is not provided, the default value will be https://app.langwatch.ai.
    base_attributes : Optional[BaseAttributes], default=None
        The base attributes to use for the LangWatch tracing client.
    
    Notes
    ------
    Since tracing requires the execution within an automa to establish the corresponding record root,
    only global configurations (via `GlobalSetting`) and automa-level configurations (via `RunningOptions`) will take effect. 
    In other words, if you set the callback by using `@worker` or `add_worker`, it will not work.
    """

    _api_key: Optional[str]
    _endpoint_url: Optional[str]
    _base_attributes: BaseAttributes
    _is_ready: bool
    _current_trace: ContextVar[Optional[LangWatchTrace]]
    _current_span_stack: ContextVar[Tuple[LangWatchSpan, ...]]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        base_attributes: Optional[BaseAttributes] = None,
    ):
        super().__init__()
        self._is_ready = False
        self._api_key = api_key
        self._endpoint_url = endpoint_url
        self._base_attributes = base_attributes
        self._current_trace = ContextVar(
            "langwatch_current_trace", default=None
        )
        self._current_span_stack = ContextVar(
            "langwatch_current_span_stack", default=()
        )
        self._setup_langwatch()

    def _setup_langwatch(self) -> None:
        """
        Initialize LangWatch and mark the callback as ready if configuration succeeds.
        """
        try:
            if get_instance() is None or self._api_key != get_instance().api_key:
                langwatch.setup(api_key=self._api_key, endpoint_url=self._endpoint_url, base_attributes=self._base_attributes)
        except Exception as exc:
            self._is_ready = False
            warnings.warn(f"LangWatch setup failed, callback disabled: {exc}")
        else:
            self._is_ready = True

    def _stringify_value(self, value: Any) -> str:
        """Serialize a value into a JSON string, falling back to str() when needed."""
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return str(value)

    def _normalize_attribute_value(self, value: Any) -> Any:
        """
        Normalize arbitrary attribute values into LangWatch-safe primitives.

        Attempts serialization through `serialize_data` and gracefully falls back to
        JSON strings when complex structures remain.
        """
        primitive_types = (str, bool, int, float, bytes)

        if isinstance(value, primitive_types) or value is None:
            return value

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            normalized = []
            for item in value:
                normalized.append(self._normalize_attribute_value(item))
            return normalized

        serialized = serialize_data(value)
        if isinstance(serialized, primitive_types) or serialized is None:
            return serialized
        if isinstance(serialized, Sequence) and not isinstance(
            serialized, (str, bytes, bytearray)
        ):
            normalized = []
            for item in serialized:
                normalized.append(self._normalize_attribute_value(item))
            return normalized

        return self._stringify_value(serialized)

    async def _complete_trace(
        self,
        output: Dict[str, Any],
        error: Optional[Exception] = None,
    ) -> None:
        """Finalize the active trace context with output and optional error."""
        trace_data = self._current_trace.get()
        if trace_data is None:
            return
        self._current_trace.set(None)
        try:
            trace_data.update(output=output, error=error)
        finally:
            await trace_data.__aexit__(
                type(error) if error else None,
                error,
                error.__traceback__ if error else None,
            )
    
    async def _finish_worker_span(
        self,
        output: Dict[str, Any],
        error: Optional[Exception] = None,
    ) -> None:
        """Finalize the current worker span and propagate outputs/errors."""
        stack = self._current_span_stack.get()
        if not stack:
            warnings.warn(
                "No active LangWatch span context found when finishing worker span"
            )
            return
        span_data = stack[-1]
        self._current_span_stack.set(stack[:-1])

        try:
            span_data.update(output=output, error=error)
        except Exception:
            warnings.warn("Failed to update LangWatch span when finishing worker span")
        await span_data.__aexit__(
            type(error) if error else None,
            error,
            error.__traceback__ if error else None,
        )

    def _build_output_payload(
        self,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        """Create a normalized payload for either successful results or errors."""
        if error:
            return {"error_type": type(error).__name__, "error_message": str(error)}
        return {
            "result_type": type(result).__name__ if result is not None else None,
            "result": serialize_data(result),
        }

    async def _start_worker_span(
        self,
        key: str,
        worker: "Worker",
        parent: "Automa",
        arguments: Optional[Dict[str, Any]],
    ) -> None:
        """
        Start a LangWatch span for the worker execution using normalized metadata.
        """
        step_name = get_worker_tracing_step_name(key, worker)
        worker_tracing_dict = build_worker_tracing_dict(worker, parent)
        normalized_worker_tracing = {
            key: self._normalize_attribute_value(value)
            for key, value in worker_tracing_dict.items()
        }
        serialized_args = serialize_data(arguments)

        # LangWatch refers to span metadata as "attributes".
        span = langwatch.span(
            name=step_name,
            input=serialized_args,
            type="agent" if worker.is_automa() else "tool",
            attributes={
                **normalized_worker_tracing,
                # TODO: Investigate why LangWatch coerces integers into dict form; keep string for now.
                "nesting_level": str(worker_tracing_dict["nesting_level"]),
            },
        )
        await span.__aenter__()
        stack = self._current_span_stack.get()
        self._current_span_stack.set((*stack, span))

    async def _start_top_level_trace(self, key: str, arguments: Optional[Dict[str, Any]]) -> None:
        serialized_args = serialize_data(arguments)
        trace_metadata = {
            "created_from": "bridgic", 
            "key": key, 
            "nesting_level": "0",
        }
        trace_data = langwatch.trace(
            name=key or "top_level_automa",
            input=serialized_args,
            metadata=trace_metadata,
            type="agent",
        )
        await trace_data.__aenter__()
        self._current_trace.set(trace_data)

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

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: "Automa" = None,
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
            await self._start_top_level_trace(key, arguments)
            return

        try:
            worker = self._get_worker_instance(key, parent)
        except (KeyError, ValueError) as e:
            warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
            return

        # Check if tracing is disabled for this worker
        if hasattr(worker, 'trace') and not worker.trace:
            return

        await self._start_worker_span(key, worker, parent, arguments)

    async def _complete_worker_execution(
        self,
        output: Dict[str, Any],
        is_top_level: bool,
        error: Optional[Exception] = None,
    ) -> None:
        if is_top_level:
            await self._complete_trace(output, error)
        else:
            await self._finish_worker_span(output, error)

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
        if not is_top_level:
            try:
                worker = self._get_worker_instance(key, parent)
                # Check if tracing is disabled for this worker
                if hasattr(worker, 'trace') and not worker.trace:
                    return
            except (KeyError, ValueError):
                # If we can't get worker instance, continue with tracing
                pass
        output = self._build_output_payload(result=result)
        await self._complete_worker_execution(output, is_top_level)

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
        if not is_top_level:
            try:
                worker = self._get_worker_instance(key, parent)
                # Check if tracing is disabled for this worker
                if hasattr(worker, 'trace') and not worker.trace:
                    return False
            except (KeyError, ValueError) as e:
                warnings.warn(f"Failed to get worker instance for key '{key}': {e}")
                return False
        output = self._build_output_payload(error=error)
        await self._complete_worker_execution(output, is_top_level, error=error)
        return False

    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["api_key"] = self._api_key
        state_dict["endpoint_url"] = self._endpoint_url
        state_dict["base_attributes"] = self._base_attributes
        return state_dict

    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._api_key = state_dict.get("api_key")
        self._endpoint_url = state_dict.get("endpoint_url")
        self._base_attributes = state_dict.get("base_attributes")
        self._current_trace = ContextVar(
            "langwatch_current_trace", default=None
        )
        self._current_span_stack = ContextVar(
            "langwatch_current_span_stack", default=()
        )
        self._setup_langwatch()

