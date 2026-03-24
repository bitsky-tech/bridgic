# Example: Amphibious Mode — Dynamic Workflow with Loops

## Table of Contents

- [The scenario](#the-scenario-abstracted)
- [context.py — Custom context + Exposure](#contextpy--custom-context--exposure)
- [tools.py — Async tool definitions](#toolspy--async-tool-definitions)
- [helpers.py — Extracting dynamic data](#helperspy--extracting-dynamic-data-from-observations)
- [agent.py — Amphibious agent](#agentpy--amphibious-agent-with-dynamic-workflow)
- [main.py — Entry point](#mainpy--entry-point)
- [Key patterns explained](#key-patterns-explained)

---

A complete example showing the most common real-world pattern: a deterministic workflow that iterates over dynamically discovered items, with agent fallback for complex sub-tasks.

**Patterns demonstrated:**
- Custom context with hidden runtime dependencies + custom Exposure for state tracking
- Amphibious mode: `on_agent` for full LLM-driven fallback, `on_workflow` for deterministic path
- Python control flow (loops, conditionals) in `on_workflow`
- Reading `ctx.observation` between steps to extract dynamic data
- `before_action` for argument sanitization and state tracking
- Agent-level `observation()` for domain-specific environment state

## The scenario (abstracted)

An agent that automates a **list → detail → extract → save** pipeline on a web application:

1. **Navigate** to a list page and apply filters (deterministic)
2. **Discover** items from the page (read observation, extract refs)
3. **Loop** over each item: open detail, extract data (agent fallback), close detail
4. **Track** processed items to avoid duplicates

This pattern applies to: CRM data extraction, order processing, content scraping, ticket triage, inventory auditing, etc.

## context.py — Custom context + Exposure

```python
"""Custom context with a hidden runtime dependency and a domain-specific Exposure."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from bridgic.amphibious import CognitiveContext, EntireExposure


class ProcessedItem(BaseModel):
    """A single processed record."""
    item_id: str
    title: str
    detail: str
    status: str = "collected"


class ItemTracker(EntireExposure[ProcessedItem]):
    """Track processed items with dedup by item_id.

    Subclassing EntireExposure lets the agent see progress in its prompt
    automatically — no extra code needed. The custom add() prevents
    duplicates, which is critical when workflows retry or resume.
    """
    def __init__(self):
        super().__init__()
        self._seen_ids: Dict[str, int] = {}

    def add(self, item: ProcessedItem) -> int:
        """Add an item, deduplicating by item_id."""
        key = item.item_id.strip()
        if key in self._seen_ids:
            return self._seen_ids[key]
        idx = super().add(item)
        self._seen_ids[key] = idx
        return idx

    def summary(self) -> List[str]:
        """Format progress for the LLM prompt."""
        lines = [
            f"id={it.item_id}, title={it.title}, [{it.status}]"
            for it in self._items
        ]
        lines.append(
            f"Total: {len(self._items)} processed. "
            f"(Do NOT re-process items already listed here.)"
        )
        return lines


class PipelineContext(CognitiveContext):
    """Context for the list→detail→extract pipeline.

    Design decisions:
    - `app_client` is hidden (display=False): it's a runtime dependency
      the LLM never sees, but tools and observation() need it.
    - `item_tracker` is visible: the LLM sees processing progress,
      which helps it avoid re-processing and know when to finish.
    - `current_page_id` is hidden: internal navigation state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Runtime dependency — hidden from LLM, used by observation() and tools
    app_client: Optional[object] = Field(
        default=None,
        description="Application client instance",
        json_schema_extra={"display": False},
    )

    # Domain state — visible to LLM (auto-included in prompt via summary())
    item_tracker: ItemTracker = Field(
        default_factory=ItemTracker,
        description="Items processed so far",
    )

    # Internal navigation state — hidden from LLM
    current_page_id: str = Field(
        default="",
        json_schema_extra={"display": False},
    )
```

**Key design principles:**
- **Hidden fields** (`display=False`) for runtime dependencies and internal state the LLM doesn't need
- **Custom Exposure** with dedup logic — critical for idempotent workflows
- **`summary()`** tells the LLM what's been done, preventing redundant work

## tools.py — Async tool definitions

```python
"""Tool definitions. Each tool is async, type-hinted, and returns str."""
from typing import List
from bridgic.core.agentic.tool_specs import FunctionToolSpec


async def navigate_to(url: str) -> str:
    """Navigate the browser to the given URL.

    Parameters
    ----------
    url : str
        The target URL.
    """
    # Implementation: call browser.navigate(url)
    return f"Navigated to {url}"


async def click_element(ref: str) -> str:
    """Click an interactive element on the page.

    Parameters
    ----------
    ref : str
        The ref attribute of the element (e.g. 'e1', 'a3f2').
    """
    # Implementation: call browser.click(ref)
    return f"Clicked element [ref={ref}]"


async def wait_for(time_seconds: str) -> str:
    """Wait for a specified number of seconds.

    Parameters
    ----------
    time_seconds : str
        Number of seconds to wait.
    """
    import asyncio
    await asyncio.sleep(float(time_seconds))
    return f"Waited {time_seconds}s"


async def save_record(
    item_id: str,
    title: str,
    detail: str,
) -> str:
    """Save an extracted record.

    Parameters
    ----------
    item_id : str
        Unique identifier for the item.
    title : str
        Item title.
    detail : str
        Extracted detail content.
    """
    return f"Saved record: {item_id}, {title}"


def get_tools() -> List[FunctionToolSpec]:
    return [
        FunctionToolSpec.from_raw(navigate_to),
        FunctionToolSpec.from_raw(click_element),
        FunctionToolSpec.from_raw(wait_for),
        FunctionToolSpec.from_raw(save_record),
    ]
```

## helpers.py — Extracting dynamic data from observations

```python
"""Helper functions to extract structured data from page observations.

These functions bridge the gap between raw observation text (e.g., an
accessibility tree or API response) and the structured values that
workflow steps need (refs, IDs, page handles).

Why helpers instead of LLM? Because these extractions are deterministic
pattern matches — faster, cheaper, and more reliable than an LLM call.
Reserve AgentFallback for tasks that genuinely need reasoning.
"""
import re
from typing import List, Optional, Tuple


def extract_item_refs(observation: str) -> List[Tuple[str, str]]:
    """Extract (ref, item_id) pairs from the page observation.

    Adapt the regex pattern to match your application's DOM structure.
    Returns a list of (ref_value, item_id) tuples.
    """
    if not observation:
        return []
    results = []
    for match in re.finditer(r'link\s+"(\w+)"\s+\[ref=(\w+)\]', observation):
        item_id, ref = match.group(1), match.group(2)
        results.append((ref, item_id))
    return results


def find_newest_page(observation: str, exclude: Optional[str] = None) -> Optional[str]:
    """Find the most recently opened page/tab ID from observation text."""
    if not observation:
        return None
    page_ids = re.findall(r'(page_\d+)', observation)
    if exclude:
        page_ids = [p for p in page_ids if p != exclude]
    return page_ids[-1] if page_ids else None


def find_active_page(observation: str) -> Optional[str]:
    """Find the currently active page/tab ID."""
    if not observation:
        return None
    match = re.search(r'(page_\d+)\s*\(active\)', observation)
    return match.group(1) if match else None
```

## agent.py — Amphibious agent with dynamic workflow

```python
"""The agent: deterministic workflow with agent fallback and dynamic loops."""
import re
from typing import List, Optional, Tuple

from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveWorker, ErrorStrategy, RunMode,
    step, AgentFallback, think_unit,
)
from bridgic.core.model.types import ToolCall
from bridgic.core.agentic.tool_specs import ToolSpec

from .context import PipelineContext, ProcessedItem
from .helpers import extract_item_refs, find_newest_page, find_active_page


class DataPipelineAgent(AmphibiousAutoma[PipelineContext]):
    """List → Detail → Extract → Save pipeline with amphibious fallback.

    on_workflow: Deterministic path for the happy case.
    on_agent: Full LLM-driven fallback when the workflow can't handle it.
    """

    # Agent-mode worker (used by on_agent and AgentFallback)
    main_think = think_unit(
        CognitiveWorker.inline(
            "You are a data extraction agent. Execute exactly one action per round.\n"
            "Check the observation for current page state before acting.\n"
            "Set finish=True when the current sub-task is fully complete."
        ),
        max_attempts=50,
        on_error=ErrorStrategy.RAISE,
    )

    # ── Agent-level observation ─────────────────────────────────────────
    async def observation(self, ctx: PipelineContext) -> Optional[str]:
        """Build environment state string from the runtime dependency.

        Called before every OTA cycle (each _run_once call).
        The returned string is stored as ctx.observation before the Think phase.
        In workflow mode, this is called before each yield step(...).
        """
        client = ctx.app_client
        if client is None:
            return "No application client available."

        # Example: get page snapshot from a browser, API response, etc.
        snapshot = await client.get_snapshot()
        if snapshot is None:
            return "No page is currently open."

        return (
            "# Current Page State\n"
            f"{snapshot}\n"
        )

    # ── on_agent: full LLM-driven fallback ──────────────────────────────
    async def on_agent(self, ctx: PipelineContext) -> None:
        """Fallback: let the LLM drive the entire task if workflow fails."""
        await self.main_think

    # ── on_workflow: deterministic path with dynamic loops ──────────────
    async def on_workflow(self, ctx: PipelineContext):
        """Deterministic workflow with Python control flow.

        Key patterns:
        1. Sequential steps for setup (navigate, filter, search)
        2. yield step(...) to trigger observation refresh
        3. Read ctx.observation between steps to extract dynamic data
        4. Python for-loop over discovered items
        5. AgentFallback for steps that need LLM reasoning
        """
        # ── Phase 1: Setup (sequential, deterministic) ──────────────────
        yield step("navigate_to", description="Open list page",
                    url="https://app.example.com/items")

        yield step("click_element", description="Open status filter",
                    ref="filter_dropdown")

        yield step("click_element", description="Select 'Pending' status",
                    ref="option_pending")

        yield step("click_element", description="Trigger search",
                    ref="search_btn")

        # ── Transition: wait & read dynamic data from observation ───────
        # Each yield step() triggers a new observation cycle.
        # After it completes, ctx.observation contains fresh page state.
        yield step("wait_for", description="Wait for results to load",
                    time_seconds="3")

        # Now extract dynamic data from the latest observation.
        # This is Python code — no LLM call, fast and reliable.
        list_page_id = find_active_page(ctx.observation)
        item_refs = extract_item_refs(ctx.observation)

        # Filter out already-processed items (idempotency)
        unprocessed = [
            (ref, item_id) for ref, item_id in item_refs
            if item_id not in ctx.item_tracker._seen_ids
        ]

        # ── Phase 2: Loop over items (dynamic, data-driven) ────────────
        for ref, item_id in unprocessed:
            # Step 1: Click item to open detail view
            yield step("click_element",
                        description=f"Open item {item_id}",
                        ref=ref)

            # Step 2: Wait for new page/tab to appear
            yield step("wait_for",
                        description="Wait for detail to load",
                        time_seconds="2")

            # Step 3: Compute dynamic value (detail page ID) from observation
            detail_page_id = find_newest_page(ctx.observation,
                                              exclude=list_page_id)

            # Step 4: Switch to the detail page (if applicable)
            if detail_page_id and detail_page_id != list_page_id:
                yield step("switch_tab",
                            description="Switch to detail page",
                            page_id=detail_page_id)

            # Step 5: Delegate extraction to agent — pages are unstructured,
            # the LLM needs to understand the content and extract fields.
            yield AgentFallback(
                goal=(
                    f"Extract the item ID, title, and full detail from this page. "
                    f"Then call save_record with these fields. "
                    f"Set finish=True after saving."
                ),
                tools=["save_record"],  # Restrict to only the save tool
                max_attempts=3,
            )

            # Step 6: Close detail and return to list
            if detail_page_id and detail_page_id != list_page_id:
                yield step("close_tab",
                            description="Close detail page",
                            page_id=detail_page_id)

    # ── before_action: sanitize & track ─────────────────────────────────
    async def before_action(
        self,
        matched_list: List[Tuple[ToolCall, ToolSpec]],
        context: PipelineContext,
    ) -> List[Tuple[ToolCall, ToolSpec]]:
        """Intercept tool calls for two purposes:

        1. Sanitize arguments — fix common LLM mistakes before execution.
        2. Track state — record processed items in the tracker.

        This runs before every tool execution, for both agent and workflow mode.
        """
        for tool_call, _ in matched_list:
            name = tool_call.name
            args = tool_call.arguments

            # Sanitize: ensure wait_for gets a valid number
            if name == "wait_for":
                try:
                    float(args.get("time_seconds", "1"))
                except (ValueError, TypeError):
                    args["time_seconds"] = "1"

            # Track: record saved items in the context
            elif name == "save_record":
                context.item_tracker.add(ProcessedItem(
                    item_id=args.get("item_id", ""),
                    title=args.get("title", ""),
                    detail=args.get("detail", ""),
                ))

        return matched_list
```

## main.py — Entry point

```python
import asyncio
from bridgic.amphibious import RunMode
from bridgic.core.model import OpenAI

from .agent import DataPipelineAgent
from .context import PipelineContext
from .tools import get_tools


async def main():
    llm = OpenAI(model="gpt-4o")
    agent = DataPipelineAgent(llm=llm, verbose=True)

    # Initialize context with runtime dependency
    ctx = PipelineContext(
        goal="Extract all pending items from the list page and save their details.",
    )
    ctx.app_client = create_app_client()  # Replace with your runtime dependency (browser, API client, etc.)
    for tool in get_tools():
        ctx.tools.add(tool)

    # Run in amphibious mode: workflow path with agent fallback
    result = await agent.arun(
        context=ctx,
        mode=RunMode.AMPHIBIOUS,
        will_fallback=True,
        max_consecutive_fallbacks=2,
        trace_running=True,
    )

    print(f"Processed: {len(ctx.item_tracker)} items")
    print(f"Tokens: {agent.spend_tokens}, Time: {agent.spend_time:.1f}s")

    # Save trace for debugging
    agent._agent_trace.save("pipeline_trace.json")


if __name__ == "__main__":
    asyncio.run(main())
```

## Key patterns explained

### 1. Dynamic data between workflow steps

The workflow is deterministic, but the **data** it operates on is discovered at runtime. The bridge between "deterministic workflow" and "dynamic data" is:

```python
yield step("wait_for", time_seconds="3")  # ① Triggers new observation
item_refs = extract_item_refs(ctx.observation)  # ② Read from observation
for ref, item_id in item_refs:  # ③ Python loop over dynamic data
    yield step("click_element", ref=ref)  # ④ Use dynamic values in steps
```

Each `yield step(...)` triggers a fresh observation cycle. After it completes, `ctx.observation` contains the latest page state. You can then use **Python code** (regex, parsing, computation) to extract structured data and feed it into subsequent steps.

### 2. When to use `step()` vs `AgentFallback`

| Use | When |
|-----|------|
| `step("tool_name", ...)` | You know exactly which tool to call and what arguments to pass |
| `AgentFallback(goal="...")` | The task needs LLM reasoning (page understanding, content extraction, decision-making) |

In the loop, clicking and navigating are deterministic (`step`), but extracting unstructured data from a detail page requires understanding (`AgentFallback`).

### 3. Custom Exposure for state tracking

`ItemTracker` extends `EntireExposure` with dedup logic. This gives you:
- **Automatic prompt inclusion**: the LLM sees processing progress without extra code
- **Idempotency**: the workflow can safely retry without duplicating items
- **State sharing**: both `on_workflow` and `on_agent` see the same tracker

### 4. `before_action` for dual purpose

The hook serves two roles simultaneously:
- **Sanitize**: fix LLM mistakes (invalid numbers, malformed arguments) before they cause tool errors
- **Track**: record state changes (items saved) that the context needs to know about, regardless of whether the save tool itself succeeds or fails

### 5. Hidden runtime dependencies

The `app_client` field is `display=False` — the LLM never sees it, but `observation()` and tools use it. This keeps the prompt clean while giving the agent access to the runtime environment.
