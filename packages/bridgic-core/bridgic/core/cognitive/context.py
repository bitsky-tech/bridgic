import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar, Tuple, get_origin, Iterator

from pydantic import BaseModel, Field, ConfigDict
from bridgic.core.agentic.tool_specs import ToolSpec
from bridgic.core.model.types import Message


################################################################################################################
# Abstract Base Classes
################################################################################################################
T = TypeVar('T')


class Exposure(ABC, Generic[T]):
    """
    Abstract base class for field-level data exposure.

    Manages list-like data (e.g., history records, tool lists) with a unified interface.
    Subclasses determine the exposure strategy:
    - LayeredExposure: supports progressive disclosure (summary + per-item details)
    - EntireExposure: only provides summary (no per-item details)

    Methods
    -------
    add(item)
        Add an element and return its index.
    summary()
        Return a list of summary strings for all elements.
    get_all()
        Return a copy of all elements.
    """

    def __init__(self):
        self._items: List[T] = []

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def add(self, item: T) -> int:
        """
        Add an element to the collection.

        Parameters
        ----------
        item : T
            The element to add.

        Returns
        -------
        int
            Index of the newly added element (0-based).
        """
        self._items.append(item)
        return len(self._items) - 1

    def get_all(self) -> List[T]:
        """
        Return a copy of all elements.

        Returns
        -------
        List[T]
            A copy of all elements.
        """
        return self._items.copy()


class LayeredExposure(Exposure[T]):
    """
    Exposure with progressive disclosure support.

    Provides two-level information architecture:
    - summary(): overview of all items
    - get_details(index): detailed information for a specific item

    Use this for data where the LLM may need to request details
    about specific items (e.g., execution history, skills).
    """
    @abstractmethod
    def summary(self) -> List[str]:
        """
        Generate summary strings for all elements.

        Returns
        -------
        List[str]
            One summary string per element.
        """
        ...

    @abstractmethod
    def get_details(self, index: int) -> Optional[str]:
        """
        Get detailed information for a specific element.

        Parameters
        ----------
        index : int
            Element index (0-based).

        Returns
        -------
        Optional[str]
            Detailed information string, or None if index is invalid.
        """
        ...


class EntireExposure(Exposure[T]):
    """
    Exposure without progressive disclosure.

    Only provides summary() - all information is exposed at once.
    Use this for data where per-item details are not needed
    or the full information should always be available (e.g., tools).
    """
    @abstractmethod
    def summary(self) -> List[str]:
        """
        Generate summary strings for all elements.

        Returns
        -------
        List[str]
            One summary string per element.
        """
        ...


class Context(BaseModel):
    """
    Base class for agent context with automatic Exposure field detection.

    Provides unified access to context data through summary and detail retrieval.
    Automatically discovers Exposure-typed fields and distinguishes between
    LayeredExposure (supports details) and EntireExposure (summary only).

    Methods
    -------
    summary()
        Get a dictionary of all field values or Exposure summaries.
    get_details(field, idx)
        Get detailed information for a specific LayeredExposure item.
    get_exposable_fields()
        Get the list of all Exposure fields.
    get_layered_fields()
        Get the list of LayeredExposure fields (support details query).
    get_field_all(field)
        Get all elements from an Exposure field.

    Examples
    --------
    >>> class MyContext(Context):
    ...     model_config = ConfigDict(arbitrary_types_allowed=True)
    ...     goal: str
    ...     history: CognitiveHistory = Field(default_factory=CognitiveHistory)  # LayeredExposure
    ...     tools: CognitiveTools = Field(default_factory=CognitiveTools)  # EntireExposure
    ...
    >>> ctx = MyContext(goal="Complete task")
    >>> ctx.history.add(Step(content="Step 1", status=True))
    >>> ctx.summary()  # Returns dict with goal and summaries of history/tools
    >>> ctx.get_details("history", 0)  # Works - history is LayeredExposure
    >>> ctx.get_details("tools", 0)  # Returns None - tools is EntireExposure
    """

    # Class-level cache for detected fields
    _exposure_fields: Optional[Dict[str, Dict[str, Any]]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._exposure_fields = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.__class__._exposure_fields is None:
            self.__class__._exposure_fields = self._detect_exposure_fields()

        # Call __post_init__ if defined in subclass (provides dataclass-like API)
        if hasattr(self, '__post_init__') and callable(getattr(self, '__post_init__')):
            self.__post_init__()

    @classmethod
    def _detect_exposure_fields(cls) -> Dict[str, Dict[str, Any]]:
        """Detect all Exposure fields and classify them."""
        exposure_fields = {}

        def _get_exposure_type(field_type: Any) -> Optional[str]:
            """Return 'layered', 'entire', or None."""
            if inspect.isclass(field_type):
                try:
                    if issubclass(field_type, LayeredExposure):
                        return 'layered'
                    elif issubclass(field_type, EntireExposure):
                        return 'entire'
                    elif issubclass(field_type, Exposure):
                        # Base Exposure - treat as entire (no details)
                        return 'entire'
                except TypeError:
                    pass

            origin = get_origin(field_type)
            if origin and inspect.isclass(origin):
                try:
                    if issubclass(origin, LayeredExposure):
                        return 'layered'
                    elif issubclass(origin, EntireExposure):
                        return 'entire'
                    elif issubclass(origin, Exposure):
                        return 'entire'
                except TypeError:
                    pass

            return None

        for field_name, field_info in cls.model_fields.items():
            exposure_type = _get_exposure_type(field_info.annotation)
            if exposure_type:
                exposure_fields[field_name] = {
                    'field_name': field_name,
                    'exposure_type': exposure_type,  # 'layered' or 'entire'
                }

        return exposure_fields

    def summary(self) -> Dict[str, str]:
        """
        Generate a summary dictionary with formatted strings for each field.

        Subclasses should override this to provide custom formatting for each field.
        The returned dictionary maps field names to their formatted string representations.

        Returns
        -------
        Dict[str, str]
            Field name to formatted summary string mapping.
            Each value should be a complete, formatted string ready for prompt inclusion.
        """
        result = {}
        exposure_fields = self.__class__._exposure_fields or {}

        # Add non-Exposure fields as simple string
        for field_name in self.__class__.model_fields:
            if field_name not in exposure_fields:
                value = getattr(self, field_name)
                if value is not None:
                    result[field_name] = f"{field_name}: {value}"

        # Add Exposure field summaries
        for field_name in exposure_fields:
            field_value = getattr(self, field_name)
            if field_value and len(field_value) > 0:
                summaries = field_value.summary()
                result[field_name] = f"{field_name}:\n" + "\n".join(f"  {s}" for s in summaries)

        return result

    def format_summary(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        separator: str = "\n"
    ) -> str:
        """
        Format the summary dictionary into a string with field selection.

        Parameters
        ----------
        include : Optional[List[str]]
            If provided, only include these fields (takes priority over exclude).
        exclude : Optional[List[str]]
            If provided, exclude these fields from the output.
        separator : str
            Separator between field summaries. Default is newline.

        Returns
        -------
        str
            Formatted summary string with selected fields.

        Examples
        --------
        >>> ctx.format_summary()  # All fields
        >>> ctx.format_summary(include=['tools', 'skills'])  # Only capabilities
        >>> ctx.format_summary(exclude=['tools', 'skills'])  # Only task context
        """
        summary_dict = self.summary()

        if include is not None:
            fields = [f for f in include if f in summary_dict]
        elif exclude is not None:
            fields = [f for f in summary_dict if f not in exclude]
        else:
            fields = list(summary_dict.keys())

        return separator.join(summary_dict[f] for f in fields if summary_dict.get(f))

    def get_field(self, field: str) -> Tuple[Optional[List[str]], Any]:
        """
        Get field information with type-aware return.

        Parameters
        ----------
        field : str
            Name of the field.

        Returns
        -------
        Tuple[Optional[List[str]], Any]
            - If field is an Exposure: (summary list, all items via get_all())
            - Otherwise: (None, raw field value)
        """
        exposure_fields = self.__class__._exposure_fields or {}
        field_value = getattr(self, field, None)

        if field in exposure_fields:
            # Exposure field: return (summary, get_all())
            if field_value and hasattr(field_value, 'summary') and hasattr(field_value, 'get_all'):
                return (field_value.summary(), field_value.get_all())
            return ([], [])
        else:
            # Non-Exposure field: return (None, value)
            return (None, field_value)

    def get_details(self, field: str, idx: int) -> Optional[str]:
        """
        Get detailed information for a LayeredExposure field item.

        Only works for LayeredExposure fields. Returns None for EntireExposure.

        Parameters
        ----------
        field : str
            Name of the Exposure field.
        idx : int
            Item index within the field (0-based).

        Returns
        -------
        Optional[str]
            Detailed information string, or None if:
            - Field doesn't exist
            - Field is not a LayeredExposure
            - Index is invalid
        """
        exposure_fields = self.__class__._exposure_fields or {}

        if field not in exposure_fields:
            return None

        # Only LayeredExposure supports get_details
        if exposure_fields[field].get('exposure_type') != 'layered':
            return None

        field_value = getattr(self, field, None)
        if field_value and hasattr(field_value, 'get_details'):
            return field_value.get_details(idx)

        return None

    def __str__(self) -> str:
        """
        Return a formatted string representation of the context.

        Automatically formats all defined fields:
        - Exposure fields: displays their summary
        - Other fields: displays field name and value

        Returns
        -------
        str
            Formatted string representation.
        """
        exposure_fields = self.__class__._exposure_fields or {}
        lines = []
        separator = "-" * 50

        lines.append(f"{'=' * 50}")
        lines.append(f"  {self.__class__.__name__}")
        lines.append(f"{'=' * 50}")

        # Format non-Exposure fields first
        for field_name in self.__class__.model_fields:
            if field_name not in exposure_fields:
                value = getattr(self, field_name)
                if value is not None:
                    lines.append(f"\n[{field_name}]")
                    lines.append(f"  {value}")

        # Format Exposure fields
        for field_name, field_info in exposure_fields.items():
            field_value = getattr(self, field_name, None)
            lines.append(f"\n[{field_name}] ({field_info.get('exposure_type', 'unknown')})")
            if field_value and len(field_value) > 0:
                for i, summary in enumerate(field_value.summary()):
                    lines.append(f"  [{i}] {summary}")
            else:
                lines.append("  (empty)")

        lines.append(f"\n{'=' * 50}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise representation of the context."""
        exposure_fields = self.__class__._exposure_fields or {}
        parts = []

        for field_name in self.__class__.model_fields:
            if field_name not in exposure_fields:
                value = getattr(self, field_name)
                if value is not None:
                    parts.append(f"{field_name}={value!r}")

        for field_name in exposure_fields:
            field_value = getattr(self, field_name, None)
            count = len(field_value) if field_value else 0
            parts.append(f"{field_name}=[{count} items]")

        return f"{self.__class__.__name__}({', '.join(parts)})"


################################################################################################################
# Default Implementations
################################################################################################################

class CognitiveTools(EntireExposure[ToolSpec]):
    """
    Manages available tools (EntireExposure - no progressive disclosure).

    All tool information is exposed in summary. Use get_all() to access
    the full ToolSpec list when detailed information is needed.
    """

    def summary(self) -> List[str]:
        """
        Generate summary strings for all tools.

        Returns
        -------
        List[str]
            Summary for each tool in format: "• {name}: {description}".
        """
        result = []
        for tool in self._items:
            desc = tool.tool_description
            result.append(f"• {tool.tool_name}: {desc}")
        return result


class Skill(BaseModel):
    """
    A single skill definition following Claude Code SKILL.md format.

    Attributes
    ----------
    name : str
        Skill name (used as /command-name).
    description : str
        What the skill does and when to use it (triggers skill invocation).
    content : str
        The full markdown instructions that Claude follows when skill is invoked.
    metadata : Dict[str, Any]
        Additional YAML frontmatter fields (e.g., disable-model-invocation, allowed-tools).
    """
    name: str
    description: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CognitiveSkills(LayeredExposure[Skill]):
    """
    Manages available skills with progressive disclosure (LayeredExposure).

    Provides skill storage, summary generation (name + description),
    and detailed content retrieval (full SKILL.md content).

    Methods
    -------
    add_from_markdown(markdown_text)
        Parse and add a skill from SKILL.md format.
    get_by_name(name)
        Get a skill by its name.
    """

    def add_from_markdown(self, markdown_text: str) -> int:
        """
        Parse a SKILL.md file and add it as a skill.

        Parameters
        ----------
        markdown_text : str
            Full content of a SKILL.md file (YAML frontmatter + markdown).

        Returns
        -------
        int
            Index of the newly added skill.

        Raises
        ------
        ValueError
            If the markdown doesn't contain valid YAML frontmatter or required fields.
        """
        import yaml

        # Split frontmatter and content
        parts = markdown_text.split('---')
        if len(parts) < 3:
            raise ValueError("Invalid SKILL.md format: missing YAML frontmatter")

        frontmatter_text = parts[1].strip()
        content = '---'.join(parts[2:]).strip()

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter: {e}")

        # Validate required fields
        if not isinstance(frontmatter, dict):
            raise ValueError("Frontmatter must be a YAML dictionary")
        if 'name' not in frontmatter:
            raise ValueError("Missing required field: name")
        if 'description' not in frontmatter:
            raise ValueError("Missing required field: description")

        # Extract fields
        name = frontmatter.pop('name')
        description = frontmatter.pop('description')
        metadata = frontmatter  # Remaining fields go to metadata

        # Create and add skill
        skill = Skill(
            name=name,
            description=description,
            content=content,
            metadata=metadata
        )
        return self.add(skill)

    def add_from_file(self, file_path: str) -> int:
        """
        Load and add a skill from a SKILL.md file.

        Parameters
        ----------
        file_path : str
            Path to the SKILL.md file.

        Returns
        -------
        int
            Index of the newly added skill.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file doesn't contain valid SKILL.md format.
        """
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {file_path}")

        markdown_text = path.read_text(encoding='utf-8')
        return self.add_from_markdown(markdown_text)

    def load_from_directory(self, directory: str, pattern: str = "**/SKILL.md") -> int:
        """
        Load all SKILL.md files from a directory recursively.

        Parameters
        ----------
        directory : str
            Path to the directory containing SKILL.md files.
        pattern : str, optional
            Glob pattern for finding skill files (default: "**/SKILL.md").

        Returns
        -------
        int
            Number of skills successfully loaded.
        """
        from pathlib import Path

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        count = 0
        for skill_file in dir_path.glob(pattern):
            try:
                self.add_from_file(str(skill_file))
                count += 1
            except (ValueError, FileNotFoundError) as e:
                # Log or handle errors for individual files
                print(f"Warning: Failed to load {skill_file}: {e}")

        return count

    def summary(self) -> List[str]:
        """
        Generate summary strings for all skills.

        Returns
        -------
        List[str]
            Summary for each skill in format: "/{name} - {description}".
        """
        result = []
        for skill in self._items:
            result.append(f"{skill.name} - {skill.description}")
        return result

    def get_details(self, index: int) -> Optional[str]:
        """
        Get detailed information for a specific skill (full SKILL.md content).

        Parameters
        ----------
        index : int
            Skill index (0-based).

        Returns
        -------
        Optional[str]
            Full skill details including frontmatter and markdown content.
            Returns None if index is out of range.
        """
        if index < 0 or index >= len(self._items):
            return None

        skill = self._items[index]
        lines = [
            f"Skill: {skill.name}",
            f"Description: {skill.description}",
            ""
        ]

        # Add metadata if present
        if skill.metadata:
            lines.append("Metadata:")
            for key, value in skill.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Add full markdown content
        lines.append("Instructions:")
        lines.append("-" * 40)
        lines.append(skill.content)

        return "\n".join(lines)


class Step(BaseModel):
    """
    A single step in task execution.

    Attributes
    ----------
    content : str
        Step content or description.
    status : bool
        Whether the step completed successfully.
    result : Optional[Any]
        Step execution result.
    metadata : Dict[str, Any]
        Additional metadata (e.g., tools used, timestamps).
    """
    content: str
    status: bool
    result: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CognitiveHistory(LayeredExposure[Step]):
    """
    Manages execution history with layered memory architecture.

    Memory layers:
    - Working memory (most recent N steps): Full details displayed directly
    - Short-term memory (next N steps): Summary only, details available on request
    - Long-term memory (older steps): Compressed into a single summary

    Parameters
    ----------
    working_memory_size : int
        Number of recent steps to show with full details (default: 5).
    short_term_size : int
        Number of steps to show as summary before working memory (default: 5).
    """

    def __init__(
        self,
        working_memory_size: int = 5,
        short_term_size: int = 5,
    ):
        super().__init__()
        self.working_memory_size = working_memory_size
        self.short_term_size = short_term_size

        # Compression state
        self.compressed_summary: str = ""
        self.compressed_count: int = 0

        # LLM for compression (set via set_llm)
        self._llm: Optional[Any] = None

    # TODO: Is it necessary to incorporate the setting of LLM as part of the context base class's capabilities, 
    # and then only require internal elements to implement the set_llm interface? An LLM will be automatically injected.
    def set_llm(self, llm: Any) -> None:
        """
        Set the LLM used for history compression.

        Parameters
        ----------
        llm : BaseLlm
            LLM instance with agenerate() method.
        """
        self._llm = llm

    def add(self, item: Step) -> int:
        """
        Add a step to history.

        Parameters
        ----------
        item : Step
            The step to add.

        Returns
        -------
        int
            Index of the newly added step.
        """
        index = super().add(item)
        return index

    async def compress_if_needed(self) -> bool:
        """
        Compress old history if needed.

        Call this after adding steps to check and perform compression.
        Requires LLM to be set via set_llm().

        Returns
        -------
        bool
            True if compression was performed, False otherwise.
        """
        if not self.needs_compression():
            return False

        if self._llm is None:
            return False

        await self._do_compress()
        return True

    def needs_compression(self) -> bool:
        """Check if there are steps that need to be compressed."""
        return len(self._get_steps_to_compress()) > 0

    def _get_steps_to_compress(self) -> List[Step]:
        """Get steps that should be compressed."""
        total = len(self._items)
        working_start = max(0, total - self.working_memory_size)
        short_term_start = max(0, working_start - self.short_term_size)

        # Steps to compress: from compressed_count to short_term_start
        if short_term_start > self.compressed_count:
            return self._items[self.compressed_count:short_term_start]
        return []

    async def _do_compress(self) -> None:
        """Perform compression using LLM."""
        steps_to_compress = self._get_steps_to_compress()
        if not steps_to_compress:
            return

        # Build compression prompt
        formatted_steps = self._format_steps_for_compression(steps_to_compress)

        system_prompt = (
            "You are a history compression assistant. Compress the execution history "
            "into a concise summary while preserving critical information.\n"
            "- Keep key data (IDs, numbers, names) that may be needed later\n"
            "- Note any failed attempts\n"
            "- Integrate with existing summary if present\n"
            "- Output a single concise paragraph"
        )

        user_parts = []
        if self.compressed_summary:
            user_parts.append(f"Existing Summary: {self.compressed_summary}")
        user_parts.append(f"New Steps:\n{formatted_steps}")
        user_parts.append("Compressed summary:")
        user_prompt = "\n\n".join(user_parts)

        # Call LLM
        new_summary = await self._llm.agenerate(
            messages=[
                Message.from_text(text=system_prompt, role="system"),
                Message.from_text(text=user_prompt, role="user")
            ]
        )

        # Update compression state
        self.compressed_summary = new_summary
        self.compressed_count += len(steps_to_compress)

    def _format_steps_for_compression(self, steps: List[Step]) -> str:
        """Format steps for compression prompt."""
        lines = []
        for i, step in enumerate(steps):
            status = "SUCCESS" if step.status else "FAILED"
            lines.append(f"{i+1}. [{status}] {step.content}")
            if step.result:
                result_str = str(step.result)[:200]
                lines.append(f"   Result: {result_str}")
        return "\n".join(lines)

    def _format_step_detail(self, step: Step, max_result_len: int = 500) -> str:
        """Format a step with full details for working memory display."""
        status = "✓" if step.status else "✗"
        lines = [f"{status} {step.content}"]

        if step.result is not None:
            result_str = str(step.result)
            if len(result_str) > max_result_len:
                result_str = result_str[:max_result_len] + "..."
            lines.append(f"   Result: {result_str}")

        return "\n".join(lines)

    def _format_step_summary(self, step: Step) -> str:
        """Format a step as brief summary for short-term memory display."""
        status = "✓" if step.status else "✗"
        return f"{status} {step.content}"

    def summary(self) -> List[str]:
        """
        Generate layered summary.

        Returns
        -------
        List[str]
            Formatted strings for each memory layer:
            - Long-term: compressed summary (if exists)
            - Short-term: step summaries with indices (queryable)
            - Working: full step details with indices
        """
        result = []
        total = len(self._items)

        # Calculate boundaries
        working_start = max(0, total - self.working_memory_size)
        short_term_start = max(0, working_start - self.short_term_size)

        # 1. Long-term memory: compressed summary
        if self.compressed_summary:
            result.append(f"[History Summary] {self.compressed_summary}")

        # 2. Short-term memory: summary only, queryable for details
        for i in range(short_term_start, working_start):
            step = self._items[i]
            summary = self._format_step_summary(step)
            result.append(f"{summary}")

        # 3. Working memory: full details
        for i in range(working_start, total):
            step = self._items[i]
            detail = self._format_step_detail(step)
            result.append(f"{detail}")

        return result

    def get_details(self, index: int) -> Optional[str]:
        """
        Get detailed information for a specific step.

        Parameters
        ----------
        index : int
            Step index (0-based).

        Returns
        -------
        Optional[str]
            Formatted step details, or None if index is out of range.
        """
        if index < 0 or index >= len(self._items):
            return None

        step = self._items[index]
        status_text = "SUCCESS" if step.status else "FAILED"
        lines = [
            f"Status: {status_text}",
            f"Content: {step.content}",
        ]

        if step.result is not None:
            result_str = str(step.result)
            lines.append(f"Result: {result_str}")

        if step.metadata:
            lines.append("Metadata:")
            for key, value in step.metadata.items():
                val_str = str(value)
                if len(val_str) > 150:
                    val_str = val_str[:150] + "..."
                lines.append(f"  {key}: {val_str}")

        return "\n".join(lines)


class CognitiveContext(Context):
    """
    The default implementation of the CognitiveContext.

    Provides all fields and methods needed by CognitiveWorker.
    Users can extend this class to add custom fields.

    Attributes
    ----------
    goal : str
        The goal to achieve.
    tools : CognitiveTools
        Available tools (EntireExposure - summary only, no per-item details).
    skills : CognitiveSkills
        Available skills (LayeredExposure - supports progressive disclosure).
    cognitive_history : CognitiveHistory
        History of cognitive steps (LayeredExposure - supports progressive disclosure).
    finish : bool
        Whether the goal is achieved.

    Examples
    --------
    >>> ctx = CognitiveContext(goal="Complete task")
    >>> ctx.tools.add(tool_spec)  # Add tools
    >>> ctx.skills.add_from_file("skills/travel-planning/SKILL.md")  # Add skills
    >>> ctx.add_info(Step(content="Step 1", status=True))
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal: str = Field(description="The goal to achieve")
    tools: CognitiveTools = Field(default_factory=CognitiveTools, description="Available tools")
    skills: CognitiveSkills = Field(default_factory=CognitiveSkills, description="Available skills")
    cognitive_history: CognitiveHistory = Field(default_factory=CognitiveHistory)
    finish: bool = Field(default=False)

    # Internal state for persisting disclosed details
    _disclosed_details: List[Tuple[str, int, str]] = []  # (field, index, detail)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize internal state
        object.__setattr__(self, '_disclosed_details', [])

    def get_details(self, field: str, idx: int) -> Optional[str]:
        """
        Get detailed information for a LayeredExposure field item.

        Overrides parent to persist disclosed details. Once retrieved,
        details are stored and included in subsequent summary() calls.

        Parameters
        ----------
        field : str
            Name of the Exposure field.
        idx : int
            Item index within the field (0-based).

        Returns
        -------
        Optional[str]
            Detailed information string, or None if unavailable.
        """
        detail = super().get_details(field, idx)

        if detail is not None:
            # Persist if not already disclosed
            disclosed = object.__getattribute__(self, '_disclosed_details')
            if not any(d[0] == field and d[1] == idx for d in disclosed):
                disclosed.append((field, idx, detail))

        return detail

    def summary(self) -> Dict[str, str]:
        """
        Generate a summary dictionary with formatted strings for each field.

        Returns a dictionary where each key is a field name and each value is
        a formatted string ready for prompt inclusion. Includes previously
        disclosed details.

        Returns
        -------
        Dict[str, str]
            Field name to formatted summary string mapping:
            - goal: "Goal: {goal}"
            - status: "Status: {In Progress|Completed}"
            - tools: formatted tool list
            - skills: formatted skill list with indices
            - cognitive_history: formatted history with indices
            - disclosed_details: previously disclosed details (if any)
        """
        result = {}

        # Format goal
        result['goal'] = f"Goal: {self.goal}"

        # Format status
        status = "Completed" if self.finish else "In Progress"
        result['status'] = f"Status: {status}"

        # Format tools (EntireExposure - no detail queries)
        if len(self.tools) > 0:
            lines = ["Available Tools:"]
            for tool_summary in self.tools.summary():
                lines.append(f"  {tool_summary}")
            result['tools'] = "\n".join(lines)

        # Format skills (LayeredExposure - with indices for detail queries)
        if len(self.skills) > 0:
            lines = ["Available Skills (request details via details_needed: {field: 'skills', index: N}):"]
            for i, skill_summary in enumerate(self.skills.summary()):
                lines.append(f"  [{i}] {skill_summary}")
            result['skills'] = "\n".join(lines)

        # Format history (layered memory architecture)
        if len(self.cognitive_history) > 0:
            total = len(self.cognitive_history)
            working_start = max(0, total - self.cognitive_history.working_memory_size)
            short_term_start = max(0, working_start - self.cognitive_history.short_term_size)

            lines = ["Execution History:"]
            if self.cognitive_history.compressed_summary:
                lines.append("  (Older history compressed into summary)")
            if short_term_start < working_start:
                lines.append(f"  (Steps [{short_term_start}-{working_start-1}]: summary only, query details via details_needed)")

            # history.summary() already returns formatted layered output
            for summary_line in self.cognitive_history.summary():
                lines.append(f"  {summary_line}")

            result['cognitive_history'] = "\n".join(lines)
        else:
            result['cognitive_history'] = "Execution History: (none)"

        # Format disclosed details (persisted from previous get_details calls)
        disclosed = object.__getattribute__(self, '_disclosed_details')
        if disclosed:
            lines = ["Previously Disclosed Details:"]
            for field, idx, detail in disclosed:
                lines.append(f"\n[{field}[{idx}]]:\n{detail}")
            result['disclosed_details'] = "\n".join(lines)

        return result

    def add_info(self, info: Step) -> int:
        """
        Add an execution step to history.

        Parameters
        ----------
        info : Step
            The step to add.

        Returns
        -------
        int
            Index of the added step.
        """
        return self.cognitive_history.add(info)

    def set_finish(self) -> None:
        """
        Set the finish flag to True.
        """
        self.finish = True

