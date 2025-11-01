"""LangWatch tracer implementation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast, get_args

from langwatch.telemetry.span import LangWatchSpan
from langwatch.types import SpanTypes
from typing_extensions import override

from agent_tracer.base import BaseTracer
from agent_tracer.service import StepTraceContext
from agent_tracer.utils import serialize_value

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langwatch.tracer import ContextSpan

    from agent_tracer.schema import Log

logger = logging.getLogger(__name__)

SpanTypesTuple = get_args(SpanTypes)

class LangWatchTracer(BaseTracer):
    """LangWatch tracer for observability."""

    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID | str,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id

        try:
            self._ready: bool = self.setup_langwatch()
            if not self._ready:
                return

            self.trace = self._client.trace(
                metadata={"trace_id": str(self.trace_id)}  # pass trace_id through metadata
            )
            self.trace.__enter__()
            self.spans: dict[str, LangWatchSpan] = {}

            self.trace.root_span.update(
                name=trace_name,
                type="agent",
            )
        except Exception:
            logger.debug("Error setting up LangWatch tracer", exc_info=True)
            self._ready = False

    @property
    def ready(self):
        return self._ready

    def setup_langwatch(self) -> bool:
        if "LANGWATCH_API_KEY" not in os.environ:
            return False
        try:
            import langwatch

            self._client = langwatch
        except ImportError:
            logger.exception("Could not import langwatch. Please install it with `pip install agent-tracer[langwatch]`.")
            return False
        return True

    @override
    def add_trace(
        self,
        trace_id: str,
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        custom_data: dict[str, Any] | None = None,
        parent_step_trace_context: StepTraceContext | None = None,
    ) -> None:
        if not self._ready:
            return

        # If user is using session_id, track it, only for chat
        if "session_id" in inputs:
            self.trace.update(metadata=(self.trace.metadata or {}) | {"thread_id": inputs["session_id"]})

        # Resolve parent span: prefer explicit parent context -> cached span; else graph edges; else root
        parent_span = self.trace.root_span
        if parent_step_trace_context and parent_step_trace_context.trace_id in self.spans:
            parent_span = self.spans[parent_step_trace_context.trace_id]

        span = self.trace.span(
            name=trace_name,
            type= cast(SpanTypes, trace_type) if trace_type in SpanTypesTuple else "span",
            parent=parent_span,
            input=self._convert_to_langwatch_types(inputs) if inputs else None,
        )
        span.__enter__()
        self.spans[trace_id] = cast(LangWatchSpan, span)

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        if not self._ready:
            return
        if self.spans.get(trace_id):
            span = self.spans[trace_id]
            span.end(output=self._convert_to_langwatch_types(outputs), error=error)
            span.__exit__(type(error) if error else None, error, None)

    @override
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ready:
            return
        self.trace.root_span.end(
            input=self._convert_to_langwatch_types(inputs) if self.trace.root_span.input is None else None,
            output=self._convert_to_langwatch_types(outputs) if self.trace.root_span.output is None else None,
            error=error,
        )

        if metadata:
            self.trace.update(metadata=(self.trace.metadata or {}) | metadata)

        if self.trace.api_key or self._client._api_key:
            try:
                self.trace.__exit__(None, None, None)
            except ValueError:  # ignoring token was created in a different Context errors
                return

    def _convert_to_langwatch_types(self, io_dict: dict[str, Any] | None):
        from langwatch.utils import autoconvert_typed_values

        if io_dict is None:
            return None
        converted = {}
        for key, value in io_dict.items():
            converted[key] = self._convert_to_langwatch_type(value)
        return autoconvert_typed_values(converted)

    def _convert_to_langwatch_type(self, value):
        if isinstance(value, dict):
            value = {key: self._convert_to_langwatch_type(val) for key, val in value.items()}
        elif isinstance(value, list):
            value = [self._convert_to_langwatch_type(v) for v in value]
        else:
            value = serialize_value(value)
        return value

    @override
    def get_langchain_callback(self) -> Any | None:
        if self.trace is None:
            return None

        return self.trace.get_langchain_callback()

