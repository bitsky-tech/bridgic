"""Base tracer abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from agent_tracer.schema import Log

    from agent_tracer.service import StepTraceContext


class BaseTracer(ABC):
    """Abstract base class for all tracer implementations."""

    trace_id: UUID | str

    @abstractmethod
    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID | str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the tracer.

        Args:
            trace_name: Name of the trace
            trace_type: Type of trace (e.g., "chain", "agent", "tool")
            project_name: Name of the project
            trace_id: Unique identifier for the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Check if tracer is ready to use."""
        raise NotImplementedError

    @abstractmethod
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
        """Add a child trace/span.

        Args:
            trace_id: Unique identifier for this trace
            trace_name: Name of the trace
            trace_type: Type of trace
            inputs: Input data for the trace
            metadata: Optional metadata
            custom_data: Optional custom data (e.g., vertex information)
            parent_step_trace_context: Optional parent step trace context (used for nested traces)
        """
        raise NotImplementedError

    @abstractmethod
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        """End a child trace/span.

        Args:
            trace_id: Identifier of the trace to end
            trace_name: Name of the trace
            outputs: Output data from the trace
            error: Optional error that occurred
            logs: Optional log entries
        """
        raise NotImplementedError

    @abstractmethod
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """End the root trace.

        Args:
            inputs: All inputs collected during the trace
            outputs: All outputs collected during the trace
            error: Optional error that occurred
            metadata: Optional metadata
        """
        raise NotImplementedError

    @abstractmethod
    def get_langchain_callback(self) -> Any | None:
        """Get LangChain-compatible callback handler if available.

        Returns:
            BaseCallbackHandler or None
        """
        raise NotImplementedError

