"""Console tracer implementation - outputs traces to console only."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from agent_tracer.base import BaseTracer
from agent_tracer.utils import serialize_value

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from agent_tracer.schema import Log


class ConsoleTracer(BaseTracer):
    """Console tracer that outputs traces to console/terminal only."""

    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID | str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the console tracer.

        Args:
            trace_name: Name of the trace
            trace_type: Type of trace (e.g., "chain", "agent", "tool")
            project_name: Name of the project
            trace_id: Unique identifier for the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.project_name = project_name
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self._ready = True
        self._spans: dict[str, dict[str, Any]] = {}
        self._start_time = datetime.now()

        # Print trace start
        self._print_header("=" * 80)
        self._print_header(f"🔍 Trace Started: {trace_name}")
        self._print_header(f"   Project: {project_name}")
        self._print_header(f"   Trace ID: {trace_id}")
        self._print_header(f"   Type: {trace_type}")
        if user_id:
            self._print_header(f"   User ID: {user_id}")
        if session_id:
            self._print_header(f"   Session ID: {session_id}")
        self._print_header(f"   Started at: {self._start_time.isoformat()}")
        self._print_header("=" * 80)

    @property
    def ready(self) -> bool:
        """Console tracer is always ready."""
        return self._ready

    def _print_header(self, message: str) -> None:
        """Print a header message."""
        print(f"[TRACE] {message}")

    def _print_data(self, label: str, data: dict[str, Any] | None) -> None:
        """Print formatted data."""
        if not data:
            return
        print(f"\n[TRACE] {label}:")
        try:
            formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            for line in formatted.split("\n"):
                print(f"  {line}")
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            print(f"  {data}")

    def _format_logs(self, logs: Sequence[Log | dict]) -> list[dict[str, Any]]:
        """Format logs for printing."""
        formatted_logs = []
        for log in logs:
            if isinstance(log, dict):
                formatted_logs.append(log)
            else:
                formatted_logs.append(
                    {
                        "name": log.name,
                        "message": serialize_value(log.message),
                        "type": log.type,
                    }
                )
        return formatted_logs

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
        """Add a child trace/span.

        Args:
            trace_id: Unique identifier for this trace
            trace_name: Name of the trace
            trace_type: Type of trace
            inputs: Input data for the trace
            metadata: Optional metadata
            custom_data: Optional custom data (e.g., vertex information)
        """
        start_time = datetime.now()
        self._spans[trace_id] = {
            "trace_id": trace_id,
            "trace_name": trace_name,
            "trace_type": trace_type,
            "inputs": serialize_value(inputs),
            "metadata": serialize_value(metadata) if metadata else None,
            "custom_data": serialize_value(custom_data) if custom_data else None,
            "start_time": start_time,
            "logs": [],
        }

        print(f"\n[TRACE] ⏩ Step Started: {trace_name} ({trace_type})")
        print(f"  Trace ID: {trace_id}")
        print(f"  Started at: {start_time.isoformat()}")
        self._print_data("  Inputs", inputs)
        if metadata:
            self._print_data("  Metadata", metadata)
        if custom_data:
            self._print_data("  Custom Data", custom_data)

    @override
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
        if trace_id not in self._spans:
            print(f"\n[TRACE] ⚠️  Warning: Step {trace_name} ({trace_id}) not found")
            return

        span = self._spans[trace_id]
        end_time = datetime.now()
        duration = (end_time - span["start_time"]).total_seconds()

        print(f"\n[TRACE] ⏹️  Step Ended: {trace_name}")
        print(f"  Trace ID: {trace_id}")
        print(f"  Ended at: {end_time.isoformat()}")
        print(f"  Duration: {duration:.3f}s")

        if outputs:
            self._print_data("  Outputs", outputs)

        if logs:
            formatted_logs = self._format_logs(logs)
            print(f"\n[TRACE]   Logs ({len(formatted_logs)} entries):")
            for log in formatted_logs:
                log_type_icon = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "debug": "🐛",
                }.get(log.get("type", "").lower(), "📝")
                print(f"    {log_type_icon} [{log.get('type', 'unknown')}] {log.get('name', 'unknown')}: {log.get('message', '')}")

        if error:
            print(f"\n[TRACE] ❌ Error:")
            print(f"  Type: {type(error).__name__}")
            print(f"  Message: {str(error)}")
            import traceback

            print(f"  Traceback:")
            for line in traceback.format_exc().split("\n"):
                print(f"    {line}")

        del self._spans[trace_id]

    @override
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
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()

        print(f"\n[TRACE] {'=' * 80}")
        print(f"[TRACE] ✅ Trace Completed: {self.trace_name}")
        print(f"[TRACE]    Duration: {duration:.3f}s")
        print(f"[TRACE]    Ended at: {end_time.isoformat()}")

        if inputs:
            self._print_data("   Final Inputs", inputs)
        if outputs:
            self._print_data("   Final Outputs", outputs)
        if metadata:
            self._print_data("   Metadata", metadata)

        if error:
            print(f"\n[TRACE] ❌ Error occurred:")
            print(f"   Type: {type(error).__name__}")
            print(f"   Message: {str(error)}")
            import traceback

            print(f"   Traceback:")
            for line in traceback.format_exc().split("\n"):
                print(f"     {line}")

        print(f"[TRACE] {'=' * 80}\n")

    @override
    def get_langchain_callback(self) -> Any | None:
        """Console tracer doesn't provide LangChain callback."""
        return None

