"""LangWatch tracer implementation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from agent_tracer.base import BaseTracer
from agent_tracer.utils import serialize_value

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langwatch.tracer import ContextSpan

    from agent_tracer.schema import Log

logger = logging.getLogger(__name__)


class LangWatchTracer(BaseTracer):
    """LangWatch tracer for observability."""

    flow_id: str

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
        self.flow_id = str(trace_name).split(" - ")[-1]

        try:
            self._ready: bool = self.setup_langwatch()
            if not self._ready:
                return

            self.trace = self._client.trace(
                trace_id=str(self.trace_id),
            )
            self.trace.__enter__()
            self.spans: dict[str, ContextSpan] = {}

            import nanoid

            name_without_id = " - ".join(str(trace_name).split(" - ")[0:-1])
            name_without_id = project_name if name_without_id == "None" else name_without_id
            self.trace.root_span.update(
                # nanoid to make the span_id globally unique, which is required for LangWatch for now
                span_id=f"{self.flow_id}-{nanoid.generate(size=6)}",
                name=name_without_id,
                type="workflow",
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
    ) -> None:
        if not self._ready:
            return

        import nanoid

        # If user is using session_id, track it
        if "session_id" in inputs and inputs["session_id"] != self.flow_id:
            self.trace.update(metadata=(self.trace.metadata or {}) | {"thread_id": inputs["session_id"]})

        name_without_id = " (".join(str(trace_name).split(" (")[0:-1])

        # Handle parent relationships if custom_data contains edge information
        previous_spans = []
        if custom_data and "vertex" in custom_data:
            vertex = custom_data["vertex"]
            if hasattr(vertex, "incoming_edges") and len(vertex.incoming_edges) > 0:
                previous_spans = [span for key, span in self.spans.items() for edge in vertex.incoming_edges if key == edge.source_id]

        span = self.trace.span(
            # Add a nanoid to make the span_id globally unique, which is required for LangWatch for now
            span_id=f"{trace_id}-{nanoid.generate(size=6)}",
            name=name_without_id,
            type="component",
            parent=(previous_spans[-1] if len(previous_spans) > 0 else self.trace.root_span),
            input=self._convert_to_langwatch_types(inputs),
        )
        self.trace.set_current_span(span)
        self.spans[trace_id] = span

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
            self.spans[trace_id].end(output=self._convert_to_langwatch_types(outputs), error=error)

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

        if metadata and "flow_name" in metadata:
            self.trace.update(metadata=(self.trace.metadata or {}) | {"labels": [f"Flow: {metadata['flow_name']}"]})

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

