# Bridgic Amphibious — Project Template

Use this template when scaffolding a new bridgic-amphibious agent project.
Replace the example domain (web research) with the user's actual domain.

---

## File: tools.py

```python
"""Web Research Agent — Tool Definitions.

Async functions that serve as the agent's capabilities.
Each function must have: type hints, docstring, and return str.
"""
from typing import List
from bridgic.core.agentic.tool_specs import FunctionToolSpec


async def search_web(query: str) -> str:
    """Search the web for information on a topic.

    Parameters
    ----------
    query : str
        The search query to execute.

    Returns
    -------
    str
        Search results as formatted text.
    """
    # Replace with actual search implementation
    return f"Results for '{query}': [result 1], [result 2], [result 3]"


async def read_page(url: str) -> str:
    """Read and extract the main content from a web page.

    Parameters
    ----------
    url : str
        The URL of the page to read.

    Returns
    -------
    str
        Extracted text content from the page.
    """
    # Replace with actual page reading implementation
    return f"Content from {url}: ..."


async def save_finding(topic: str, summary: str, source_url: str) -> str:
    """Save a research finding to the collection.

    Parameters
    ----------
    topic : str
        The topic this finding relates to.
    summary : str
        A concise summary of the finding.
    source_url : str
        The URL where this information was found.
    """
    return f"Saved finding on '{topic}' from {source_url}"


def get_tools() -> List[FunctionToolSpec]:
    """Get all tool specs for this project."""
    return [
        FunctionToolSpec.from_raw(search_web),
        FunctionToolSpec.from_raw(read_page),
        FunctionToolSpec.from_raw(save_finding),
    ]
```

---

## File: context.py (optional — only if CognitiveContext needs extension)

```python
"""Web Research Agent — Custom Context."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from bridgic.amphibious import CognitiveContext, EntireExposure


class Finding(BaseModel):
    topic: str
    summary: str
    source_url: str


class FindingsTracker(EntireExposure[Finding]):
    """Track collected findings with dedup by source_url."""
    def __init__(self):
        super().__init__()
        self._seen_urls: Dict[str, int] = {}

    def add(self, item: Finding) -> int:
        if item.source_url in self._seen_urls:
            return self._seen_urls[item.source_url]
        idx = super().add(item)
        self._seen_urls[item.source_url] = idx
        return idx

    def summary(self) -> List[str]:
        lines = [f"[{i}] {f.topic}: {f.summary} ({f.source_url})" for i, f in enumerate(self._items)]
        lines.append(f"Total: {len(self._items)} findings collected.")
        return lines


class ResearchContext(CognitiveContext):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Visible to LLM — research progress
    findings: FindingsTracker = Field(
        default_factory=FindingsTracker,
        description="Research findings collected so far",
    )

    # Hidden from LLM — internal state
    urls_visited: List[str] = Field(
        default_factory=list,
        json_schema_extra={"display": False},
    )
```

---

## File: workers.py (optional — only if inline workers are insufficient)

```python
"""Web Research Agent — Custom Workers."""
from bridgic.amphibious import CognitiveWorker, _DELEGATE


class ResearchWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return (
            "You are a web research assistant. Plan ONE immediate next step.\n"
            "Search for information, read promising pages, and save key findings.\n"
            "Set finish=True when you have collected enough findings on the topic."
        )

    async def observation(self, context):
        # Return custom observation, or _DELEGATE to use agent-level observation
        return _DELEGATE

    async def after_action(self, step_result, ctx):
        # Track visited URLs after page reads
        for r in getattr(step_result, 'results', []):
            if r.tool_name == "read_page" and r.success:
                ctx.urls_visited.append(r.tool_arguments.get("url", ""))
        return _DELEGATE
```

---

## File: agent.py

```python
"""Web Research Agent — Agent Definition."""
from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveWorker,
    think_unit,
    ErrorStrategy,
    # step, AgentFallback,  # Uncomment for workflow mode
)
from .context import ResearchContext, Finding
from .tools import get_tools


class ResearchAgent(AmphibiousAutoma[ResearchContext]):
    # Declare think units
    research = think_unit(
        CognitiveWorker.inline(
            "You are a web research assistant. Plan ONE immediate next step.\n"
            "Search for information, read promising pages, and save key findings.\n"
            "Set finish=True when you have collected enough findings on the topic."
        ),
        max_attempts=20,
        on_error=ErrorStrategy.RAISE,
    )

    async def on_agent(self, ctx: ResearchContext):
        """Run the research loop."""
        await self.research

    # Agent-level hook: track findings from save_finding calls
    async def after_action(self, step_result, ctx):
        for r in getattr(step_result, 'results', []):
            if r.tool_name == "save_finding" and r.success:
                ctx.findings.add(Finding(
                    topic=r.tool_arguments.get("topic", ""),
                    summary=r.tool_arguments.get("summary", ""),
                    source_url=r.tool_arguments.get("source_url", ""),
                ))

    # Optional: workflow mode (uncomment if needed)
    # async def on_workflow(self, ctx):
    #     yield step("search_web", query="initial query")
    #     yield AgentFallback(goal="Analyze results and save findings", max_attempts=5)
```

---

## File: main.py

```python
"""Web Research Agent — Entry Point."""
import asyncio
from bridgic.core.model import OpenAI
from .agent import ResearchAgent
from .context import ResearchContext
from .tools import get_tools


async def main():
    # 1. Create LLM
    llm = OpenAI(model="gpt-4o")

    # 2. Create agent
    agent = ResearchAgent(llm=llm, verbose=True)

    # 3. Run
    result = await agent.arun(
        goal="Research the latest developments in AI agents and collect key findings.",
        tools=get_tools(),
        # trace_running=True,  # Enable execution tracing
    )

    print(f"Result: {result}")
    print(f"Final answer: {agent.final_answer}")
    print(f"Tokens: {agent.spend_tokens}, Time: {agent.spend_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
```
