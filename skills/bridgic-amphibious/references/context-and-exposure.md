# Context & Exposure — State Management

## Table of Contents

- [Why Exposure exists](#why-exposure-exists)
- [Exposure hierarchy](#exposure-hierarchy)
- [Context base classes](#context-base-classes)
- [Extending CognitiveContext](#extending-cognitivecontext)
- [Creating custom Exposure](#creating-custom-exposure)
- [CognitiveHistory — 3-tier memory](#cognitivehistory--3-tier-memory)
- [Runtime skills](#runtime-skills)

---

## Why Exposure exists

LLMs have limited context windows. You can't dump all data into every prompt. **Exposure** is a pattern for managing list-like data with controlled visibility:

- **What the LLM sees** depends on the exposure strategy
- **Data grows** (history, documents, skills) but prompt stays manageable
- **On-demand details** — LLM can request more info when it needs it (via acquiring policy)

## Exposure hierarchy

```
Exposure[T]                # Abstract base: list of items with add(), summary(), get_all()
├── LayeredExposure[T]     # Progressive disclosure
│                          summary() → brief overviews
│                          get_details(index) → full content, cached in _revealed
│                          LLM requests details via acquiring policy
└── EntireExposure[T]      # Full disclosure
                           summary() → all info, no detail layer
```

**When to use which:**

| Type | Use when | Example |
|------|----------|---------|
| `LayeredExposure` | Data is large, LLM only needs details selectively | Skills, history, documents |
| `EntireExposure` | Data is small or LLM always needs full info | Tools (always need full spec) |

### How progressive disclosure works (LayeredExposure)

1. **Round 1**: LLM sees summaries only: `[0] search_web — Search the web for info`
2. **LLM decides**: "I need details on skill 0" → fills `details: [{field: "skills", index: 0}]` (acquiring policy)
3. **System calls** `reveal(0)` → fetches full content, caches in `_revealed`
4. **Round 2**: LLM sees summary **plus** disclosed details for item 0
5. **Acquiring closes**: Policy fires once, then LLM must decide with what it has

`snapshot()` can clear `_revealed` via `keep_revealed=None` to start fresh in a new phase.

## Context base classes

```python
from bridgic.amphibious import Context, CognitiveContext
```

**`Context`** — Pydantic BaseModel that auto-detects Exposure fields. Provides `summary()`, `get_details()`, `get_field()`. Rarely used directly.

**`CognitiveContext`** — The standard context with built-in fields:

| Field | Type | What it does |
|-------|------|--------------|
| `goal` | `str` | Injected into every prompt as the agent's objective |
| `tools` | `CognitiveTools` (EntireExposure) | Registered tools, always fully visible |
| `skills` | `CognitiveSkills` (LayeredExposure) | Skills with progressive disclosure |
| `cognitive_history` | `CognitiveHistory` (LayeredExposure) | 3-tier memory of past steps |
| `observation` | `Optional[str]` | Hidden from summary (`display=False`), set by hooks |

**Use `CognitiveContext` directly** for simple agents. Extend it when you need domain-specific fields.

## Extending CognitiveContext

Extend when your agent needs domain-specific state that the LLM should see or that hooks should update:

```python
from pydantic import Field, ConfigDict
from bridgic.amphibious import CognitiveContext

class BrowserContext(CognitiveContext):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Visible to LLM (shown in prompt via summary())
    current_url: str = Field(default="", description="Current page URL")

    # Hidden from LLM (internal tracking only)
    page_html: str = Field(default="", json_schema_extra={"display": False})

    # Custom Exposure field (auto-detected, no registration needed)
    bookmarks: MyBookmarkExposure = Field(default_factory=MyBookmarkExposure)
```

**Field visibility rules:**

| Annotation | LLM sees it? | Use case |
|-----------|--------------|----------|
| `Field(description="...")` | Yes, in prompt | Domain state LLM should reason about |
| `json_schema_extra={"display": False}` | No | Internal tracking, raw data, counters |
| `json_schema_extra={"use_llm": True}` | N/A | Auto-inject agent's LLM into this field (e.g. for CognitiveHistory compression) |

**`__post_init__`** — Called automatically after construction. Use for initialization logic:

```python
def __post_init__(self):
    for tool in get_default_tools():
        self.tools.add(tool)
```

## Creating custom Exposure

### LayeredExposure — progressive disclosure for large collections

Subclass `LayeredExposure` when you have a growing collection that's too large for full display:

```python
from bridgic.amphibious import LayeredExposure

class DocumentExposure(LayeredExposure):
    def summary(self) -> List[str]:
        """Brief overview per item — what LLM sees by default."""
        return [f"[{i}] {doc['title']} ({doc['source']})" for i, doc in enumerate(self._items)]

    def get_details(self, index: int) -> Optional[str]:
        """Full content — returned when LLM requests via acquiring."""
        if 0 <= index < len(self._items):
            return f"Title: {self._items[index]['title']}\n{self._items[index]['content']}"
        return None
```

### EntireExposure — full display with custom business logic

Subclass `EntireExposure` when items are small enough to always show fully, but you need custom behavior like deduplication or progress formatting:

```python
from bridgic.amphibious import EntireExposure

class ItemTracker(EntireExposure[ProcessedItem]):
    """Track processed items with dedup — a common pattern for pipelines."""
    def __init__(self):
        super().__init__()
        self._seen_ids: Dict[str, int] = {}

    def add(self, item: ProcessedItem) -> int:
        """Override add() to prevent duplicates."""
        key = item.item_id.strip()
        if key in self._seen_ids:
            return self._seen_ids[key]
        idx = super().add(item)
        self._seen_ids[key] = idx
        return idx

    def summary(self) -> List[str]:
        """Custom summary — shows progress and prevents LLM from re-processing."""
        lines = [f"id={it.item_id}, title={it.title}, [{it.status}]" for it in self._items]
        lines.append(f"Total: {len(self._items)} processed. (Do NOT re-process.)")
        return lines
```

**Why custom Exposure?** The `summary()` return value is injected directly into the LLM prompt. By formatting it well, you guide the LLM's behavior: showing "Do NOT re-process" prevents duplicates, showing progress helps the LLM decide when to finish.

## CognitiveHistory — 3-tier memory

The problem: as an agent takes more steps, history grows unboundedly. CognitiveHistory solves this with automatic tiering:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Long-term    │ Compressed summary (LLM-generated, single string)   │
│              │ + pending buffer (brief summaries awaiting compress) │
│──────────────│─────────────────────────────────────────────────────│
│ Short-term   │ Summary only, details queryable via acquiring       │
│              │ "[Short-term Memory] [5] Searched for AI trends"    │
│──────────────│─────────────────────────────────────────────────────│
│ Working      │ Full details displayed directly                     │
│              │ "[Working Memory] [8] Called search('AI')→'Found…'" │
└─────────────────────────────────────────────────────────────────────┘
             ↑ most recent steps                    oldest steps ↑
```

**What LLM sees each round:**
- **Working** (last N steps): full tool calls, results, reasoning — complete context for current decisions
- **Short-term** (previous M steps): one-line summaries, can request details via acquiring
- **Long-term** (older): single compressed paragraph covering all ancient history

**Parameters:**
- `working_memory_size=5` — how many recent steps shown in full
- `short_term_size=20` — how many steps kept as queryable summaries
- `compress_threshold=10` — batch size before LLM compresses pending into long-term summary

**Compression** is automatic: when the pending buffer reaches `compress_threshold`, an LLM call merges it into the running compressed summary. This keeps prompt size bounded regardless of agent run length.

## Runtime skills

Skills are markdown instructions that guide agent behavior. They use the same SKILL.md format:

```yaml
---
name: my-skill
description: "When to use this skill"
---
# Skill content
Instructions the agent follows when this skill is activated...
```

**Loading skills into context:**

```python
from bridgic.amphibious import Skill

# Individual skill
ctx.skills.add(Skill(name="x", description="...", content="..."))

# From file
ctx.skills.add_from_file("path/to/SKILL.md")

# From directory (loads all SKILL.md files)
ctx.skills.load_from_directory("skills/")

# From raw markdown string
ctx.skills.add("---\nname: x\ndescription: '...'\n---\n# Content...")
```

Skills are `LayeredExposure` — LLM sees summaries first, can request full content via acquiring. Use `think_unit(..., skills=["skill_name"])` to filter which skills a worker can see.
