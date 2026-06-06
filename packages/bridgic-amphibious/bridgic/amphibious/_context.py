import functools
import inspect
from types import MethodType
from typing import (
    Any, Callable, ClassVar, Dict, List, Optional, Tuple, Iterator,
)

from pydantic import BaseModel, Field, ConfigDict
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.amphibious._type import OTARecord


################################################################################################################
# _to_tool_spec — normalize a declared tool (callable | bound method | ToolSpec) into a ToolSpec
################################################################################################################

def _to_tool_spec(obj: Any) -> ToolSpec:
    """Normalize a declared tool into a :class:`ToolSpec`.

    Backing :meth:`OTAContext.tool`, this accepts whatever a context declares
    and returns a ready-to-use spec. The caller need not care which form it
    passes:

    * a :class:`ToolSpec` — returned unchanged.
    * a **bound method** (``isinstance(obj, MethodType)``) — kept bound to its
      own ``self``. ``FunctionToolSpec.from_raw`` rejects bound methods
      (``isinstance(func, MethodType) -> ValueError``), so the method is wrapped
      in a ``functools.wraps`` async closure that calls it (awaiting if the
      result is awaitable); ``_invoke.__signature__`` is pinned to the bound
      method's signature (which already excludes ``self``) before handing it to
      ``from_raw``. Capturing the already-bound method is what preserves the
      original instance as ``self``.
    * any other **callable** — ``FunctionToolSpec.from_raw(obj)``.

    Parameters
    ----------
    obj : Any
        A :class:`ToolSpec`, a bound method, or a plain callable.

    Returns
    -------
    ToolSpec
        The normalized tool spec.
    """
    if isinstance(obj, ToolSpec):
        return obj

    if isinstance(obj, MethodType):
        bound = obj

        @functools.wraps(bound)
        async def _invoke(*args: Any, **kwargs: Any) -> Any:
            result = bound(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        # A bound-method signature already excludes ``self``.
        _invoke.__signature__ = inspect.signature(bound)
        return FunctionToolSpec.from_raw(_invoke)

    return FunctionToolSpec.from_raw(obj)


class Context(BaseModel):
    """
    Base class for agent context — fields + an overridable ``summary``.

    Collapsed (per the small-loop redesign) to its essentials:

    * :meth:`_raw_fields` — the most primitive view: every field's raw value
      as ``{name: value}`` (no rendering, no filtering).
    * :meth:`summary` — the **overridable** method the framework injects with
      that raw dict. The default returns the dict unchanged; an override is
      handed the ``fields`` dict and composes whatever it wants (usually a
      ``str``) — without fetching anything itself.

    A bare ``Context`` is free-form cross-turn state (the big-loop half):
    declare fields and override ``summary``, and that is all. Tools are **not**
    a base-context concern — they belong to the OTA loop that actually acts, so
    the tool registry (``tools`` field + :meth:`~OTAContext.tool` /
    :meth:`~OTAContext.add_tool`) lives on :class:`OTAContext`.

    Examples
    --------
    >>> class MyContext(Context):
    ...     goal: str = ""
    ...     def summary(self, fields):   # ``fields`` is auto-injected (the raw dict)
    ...         return f"Goal: {fields['goal']}"
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ############################################################################
    # Initialize Context
    ############################################################################

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # A subclass that overrides ``summary`` uses ``fields`` directly.
        if "summary" in cls.__dict__:
            cls.summary = cls._summary_injecting(cls.__dict__["summary"])

    @staticmethod
    def _summary_injecting(user_summary: Callable) -> Callable:
        """
        Wrap an overridden ``summary`` so its ``fields`` argument is always
        the raw per-field dict (built via :meth:`_raw_fields` when omitted).
        """
        @functools.wraps(user_summary)
        def _wrapped(self, fields: Optional[Dict[str, Any]] = None) -> Any:
            if fields is None:
                fields = self._raw_fields()
            return user_summary(self, fields)
        return _wrapped

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        # Call __post_init__ if defined in subclass (provides dataclass-like API)
        if hasattr(self, '__post_init__') and callable(getattr(self, '__post_init__')):
            self.__post_init__()

    ############################################################################
    # Core Methods
    ############################################################################
    
    def _raw_fields(self) -> Dict[str, Any]:
        """Every model field's raw value, unprocessed: ``{name: value}``.

        No rendering, no per-field summarising, no filtering — the most
        primitive view of the context. This is the dict the framework hands
        to :meth:`summary`.
        """
        return {name: getattr(self, name) for name in type(self).model_fields}

    def summary(self, fields: Optional[Dict[str, Any]] = None) -> Any:
        """Assemble this context for the prompt (the **overridable** method).

        An overridden ``summary`` is auto-wrapped (see :meth:`__init_subclass__`)
        so ``fields`` is always the raw per-field dict (:meth:`_raw_fields`) — the
        override just uses it (``fields.get(...)``) and composes whatever it wants
        (typically a ``str``), with zero manual fetch. The default (no override)
        returns the raw dict unchanged.

        Parameters
        ----------
        fields : Optional[Dict[str, Any]]
            The raw per-field dict, auto-injected into an overridden ``summary``.

        Returns
        -------
        Any
            The raw dict by default, or whatever an override composes.
        """
        return fields if fields is not None else self._raw_fields()

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Yield ``(field_name, field_value)`` for every model field on this context."""
        for field_name in type(self).model_fields:
            yield field_name, getattr(self, field_name)

    def __str__(self) -> str:
        """Return a formatted, human-readable view of every field."""
        lines = [f"{'=' * 50}", f"  {self.__class__.__name__}", f"{'=' * 50}"]
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if value is not None:
                lines.append(f"\n[{field_name}]")
                lines.append(f"  {value}")
        lines.append(f"\n{'=' * 50}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise representation of the context."""
        parts = []
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if value is not None:
                parts.append(f"{field_name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


################################################################################################################
# Small-loop context (framework-owned)
################################################################################################################

class OTAContext(Context):
    """Small-loop working context: one run's input + its OTA round trace + tools.

    The **framework-owned** half of the two-loop model. During a run the
    automa drives it directly through the per-round result accessors
    (``ota_ctx.obs_result = ...`` / ``.think_result`` / ``.action_result``),
    :meth:`open_record`, and :meth:`add_tool`. Each round is one
    :class:`OTARecord` (observe/think/action results, ``extra="allow"`` so a
    ``before_action`` / ``after_action`` hook can fold custom per-round fields
    like a ``permission_result`` via :meth:`_current_record`).

    Its ``tools`` are **declared on the class** via :meth:`tool` — the registry
    lives here, not on the base :class:`Context`, because tools are an OTA-loop
    concern. The framework no longer merges any tools in; whatever the context
    declares is what the small loop carries.

    Attributes
    ----------
    user_input : str
        The single question / objective this OTA run answers.
    ota_record : List[OTARecord]
        The observe-think-act round trace (one :class:`OTARecord` per round).

    Examples
    --------
    >>> ota = OTAContext(user_input="Find the bug")
    >>> ota.open_record()                  # framework brackets each OTA cycle
    >>> ota.obs_result = "saw a stack trace"
    >>> ota.think_result = decision
    >>> ota.action_result = tool_output
    """
    _declared_tools: ClassVar[List[ToolSpec]] = []  # Tools declared on this class via ``tool`` (per-subclass, inherits bases).

    user_input: str = Field(default="", description="This run's question / objective")
    ota_record: List[OTARecord] = Field(
        default_factory=list,
        description="Observe-think-act round trace (one OTARecord per round)",
    )
    tools: List[ToolSpec] = Field(
        default_factory=list,
        description="Action-phase tool affordances carried by this OTA run",
    )

    ############################################################################
    # Tool registry (action-phase affordances the OTA loop carries)
    ############################################################################
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # A subclass inherits its bases' declared tools and can add more via its own ``tool`` calls.
        seeded: List[ToolSpec] = []
        seen: set = set()
        for base in cls.__bases__:
            for spec in getattr(base, "_declared_tools", []):
                if spec.tool_name in seen:
                    continue
                seen.add(spec.tool_name)
                seeded.append(spec)
        cls._declared_tools = seeded

    @classmethod
    def tool(cls, obj):
        """Declare a tool on this OTA context — usable as a decorator **and** a call.

        Every context now declares the tools it carries; nothing is
        auto-injected by the framework. ``obj`` is normalized by
        :func:`_to_tool_spec` and appended to this class's
        :attr:`_declared_tools`. The original ``obj`` is returned so this works
        transparently as a decorator.

        * ``@MyOTACtx.tool`` on a standalone ``def f(...)`` — registers ``f``,
          returns ``f``.
        * ``MyOTACtx.tool(bash_tool)`` — registers an existing :class:`ToolSpec`.
        * ``MyOTACtx.tool(obj.method)`` — registers a bound method, keeping
          ``obj`` as its ``self`` (see :func:`_to_tool_spec`).

        Parameters
        ----------
        obj : Callable | ToolSpec
            A plain callable, a bound method, or an existing tool spec.

        Returns
        -------
        Callable | ToolSpec
            ``obj`` unchanged, so this may be used as a decorator.
        """
        cls._declared_tools.append(_to_tool_spec(obj))
        return obj

    def add_tool(self, tool: ToolSpec) -> None:
        """Register a tool into this run's action-phase toolset."""
        self.tools.append(tool)

    def model_post_init(self, __context: Any) -> None:
        # Seed this run's toolset from the class's declared tools before the
        # base hook fires, so a subclass ``__post_init__`` can rely on it. An
        # explicit ``tools=`` (e.g. a narrowed delegation set) is preserved.
        if not self.tools:
            self.tools = list(type(self)._declared_tools)
        super().model_post_init(__context)

    ############################################################################
    # Round lifecycle + per-round result accessors
    ############################################################################
    def _current_record(self) -> OTARecord:
        """The in-flight (latest) round; opens one lazily if none exists yet.

        This is the record the result accessors write to, and the fold-point
        hooks attach custom fields onto (e.g.
        ``ota_ctx._current_record().permission_result = verdict``).
        """
        if not self.ota_record:
            self.open_record()
        return self.ota_record[-1]
    
    def open_record(self) -> None:
        """Explicitly open a new round (append a new record to the trace)."""
        self.ota_record.append(OTARecord())

    @property
    def obs_result(self) -> Any:
        return self.ota_record[-1].observation_result if self.ota_record else None

    @obs_result.setter
    def obs_result(self, value: Any) -> None:
        self._current_record().observation_result = value

    @property
    def think_result(self) -> Any:
        return self.ota_record[-1].think_result if self.ota_record else None

    @think_result.setter
    def think_result(self, value: Any) -> None:
        self._current_record().think_result = value

    @property
    def action_result(self) -> Any:
        return self.ota_record[-1].action_result if self.ota_record else None

    @action_result.setter
    def action_result(self, value: Any) -> None:
        self._current_record().action_result = value

    ############################################################################
    # Prompt rendering (overridable)
    ############################################################################
    def summary(self, fields: Optional[Dict[str, Any]] = None) -> str:
        """Render this run's small-loop state for the prompt.

        Default: the user input + the OTA round trace. ``fields`` is the
        auto-injected raw dict (available to an override that prefers it);
        subclass and override to customise how the run is summarised.

        Returns
        -------
        str
            A prompt-facing rendering of the input + round trace.
        """
        parts: List[str] = [f"User input: {self.user_input}"]
        for i, record in enumerate(self.ota_record):
            parts.append(f"[Round {i}]")
            if record.observation_result is not None:
                parts.append(f"  Observation: {record.observation_result}")
            if record.think_result is not None:
                parts.append(f"  Think: {record.think_result}")
            if record.action_result is not None:
                parts.append(f"  Action: {record.action_result}")
            for key, value in (getattr(record, "model_extra", None) or {}).items():
                parts.append(f"  {key}: {value}")
        return "\n".join(parts)

