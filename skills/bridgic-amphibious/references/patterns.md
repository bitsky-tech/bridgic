# Bridgic Amphibious Code Patterns

## Table of Contents
- [Minimal Agent (Agent Mode)](#minimal-agent-agent-mode)
- [Workflow Mode](#workflow-mode)
- [Human-in-the-Loop](#human-in-the-loop)
- [Amphibious Mode](#amphibious-mode)
- [Custom Worker](#custom-worker)
- [Structured Output (output_schema)](#structured-output-output_schema)
- [Custom Context](#custom-context)
- [Phase Annotation](#phase-annotation)
- [Cognitive Policies](#cognitive-policies)
- [OTC Hooks](#otc-hooks)
- [Skills Usage](#skills-usage)
- [Memory Configuration](#memory-configuration)
- [Conditional Loops](#conditional-loops)
- [Tool & Skill Filtering](#tool--skill-filtering)
- [Execution Tracing](#execution-tracing)

---

## Minimal Agent (Agent Mode)

```python
from bridgic.amphibious import (
    AmphibiousAutoma, CognitiveContext, CognitiveWorker, think_unit,
)
from bridgic.core.agentic.tool_specs import FunctionToolSpec

# 1. Define tools
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

get_weather_tool = FunctionToolSpec.from_raw(get_weather)

# 2. Define agent
class WeatherAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline("Look up weather and provide a summary."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        await self.planner

# 3. Run
agent = WeatherAgent(llm=llm, verbose=True)
result = await agent.arun(
    goal="Check the weather in Tokyo and London.",
    tools=[get_weather_tool],
)
print(agent.final_answer)
```

## Workflow Mode

```python
from bridgic.amphibious import ActionCall

class WeatherWorkflow(AmphibiousAutoma[CognitiveContext]):
    async def on_agent(self, ctx: CognitiveContext):
        pass  # Required but not used in pure workflow

    async def on_workflow(self, ctx: CognitiveContext):
        tokyo = yield ActionCall("get_weather", city="Tokyo")
        london = yield ActionCall("get_weather", city="London")

        tokyo_val = tokyo[0].result if tokyo else "N/A"
        london_val = london[0].result if london else "N/A"
        self.set_final_answer(f"Tokyo: {tokyo_val}, London: {london_val}")

workflow = WeatherWorkflow(llm=llm)
result = await workflow.arun(
    goal="Check weather",
    tools=[get_weather_tool],
)
```

## Human-in-the-Loop

Three entry points for requesting human input:

### Entry 1: Code-level in on_agent()

```python
class InteractiveAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(CognitiveWorker.inline("Execute step."), max_attempts=10)

    async def on_agent(self, ctx: CognitiveContext):
        await self.worker
        feedback = await self.request_human("Task complete. Any follow-up?")
        if feedback != "no":
            async with self.snapshot(goal=feedback):
                await self.worker
```

### Entry 2: HumanCall in on_workflow()

```python
from bridgic.amphibious import ActionCall, HumanCall

class ConfirmableWorkflow(AmphibiousAutoma[CognitiveContext]):
    async def on_agent(self, ctx): pass

    async def on_workflow(self, ctx: CognitiveContext):
        result = yield ActionCall("search_flights", origin="Beijing", destination="Tokyo", date="2024-06-01")
        feedback = yield HumanCall(prompt="Found flights. Book CA123?")
        if feedback == "yes":
            yield ActionCall("book_flight", flight_number="CA123")
        else:
            self.set_final_answer("Booking cancelled by user.")
```

### Entry 3: LLM tool (autonomous)

```python
from bridgic.amphibious.buildin_tools import human_request_tool

class AutonomousAgent(AmphibiousAutoma[CognitiveContext]):
    worker = think_unit(
        CognitiveWorker.inline("Execute the task. Use ask_human when you need user input."),
        max_attempts=10,
    )
    async def on_agent(self, ctx): await self.worker

# human_request_tool is a plain FunctionToolSpec — use like any other tool
agent = AutonomousAgent(llm=llm)
await agent.arun(goal="Plan a trip", tools=[search_tool, human_request_tool])
```

### Custom UI Integration

```python
class WebAgent(AmphibiousAutoma[CognitiveContext]):
    async def human_input(self, data):
        """Override to use WebSocket instead of stdin."""
        prompt = data["prompt"]
        return await websocket.send_and_receive(prompt)

    async def on_agent(self, ctx): ...
```

## Amphibious Mode

```python
from bridgic.amphibious import RunMode, AgentCall, ActionCall

class FormFiller(AmphibiousAutoma[CognitiveContext]):
    fixer = think_unit(
        CognitiveWorker.inline("Diagnose the problem and fix it."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        await self.fixer

    async def on_workflow(self, ctx: CognitiveContext):
        yield ActionCall("fill_field", field_name="username", value="john")
        yield ActionCall("fill_field", field_name="email", value="john@example.com")
        yield ActionCall("click_button", button_name="submit")

# Workflow runs; on failure, agent takes over automatically
agent = FormFiller(llm=llm, verbose=True)
result = await agent.arun(
    goal="Fill and submit the form",
    tools=[fill_field_tool, click_button_tool, solve_captcha_tool],
    mode=RunMode.AMPHIBIOUS,
    will_fallback=True,
    max_consecutive_fallbacks=2,
)
```

### AgentCall in Workflow

```python
async def on_workflow(self, ctx: CognitiveContext):
    yield ActionCall("search_price", platform="Amazon", product="laptop")
    yield ActionCall("search_price", platform="eBay", product="laptop")

    # Delegate complex analysis to LLM (clean context snapshot)
    yield AgentCall(
        goal="Analyze prices and decide if we need more platforms.",
        max_attempts=3,
    )
```

## Custom Worker

```python
class DestinationAnalyzer(CognitiveWorker):
    async def thinking(self) -> str:
        return "Analyze the destination and suggest a day-by-day plan."

    async def observation(self, context: CognitiveContext):
        return (
            f"Current goal: {context.goal}\n"
            f"Tip: Visit attractions early morning to avoid crowds."
        )

class TravelPlanner(AmphibiousAutoma[CognitiveContext]):
    analyzer = think_unit(DestinationAnalyzer(), max_attempts=3)
    planner = think_unit(
        CognitiveWorker.inline("Create a detailed itinerary."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        await self.analyzer
        await self.planner
```

## Structured Output (output_schema)

```python
from pydantic import BaseModel, Field

class PlanResult(BaseModel):
    phases: list[str] = Field(description="Execution phases")
    estimated_steps: int = Field(description="Total steps needed")

class PlannerAgent(AmphibiousAutoma[CognitiveContext]):
    planner = think_unit(
        CognitiveWorker.inline(
            "Create a step-by-step execution plan.",
            output_schema=PlanResult,
        ),
        max_attempts=1,
    )

    async def on_agent(self, ctx: CognitiveContext):
        plan = await self.planner  # Returns PlanResult instance
        print(plan.phases)
```

## Custom Context

```python
from pydantic import Field, ConfigDict
from bridgic.amphibious import CognitiveContext, CognitiveHistory, ActionResult

class DocumentContext(CognitiveContext):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_document: str = Field(
        default="",
        description="Name of the document being analyzed"
    )
    analysis_results: dict = Field(
        default_factory=dict,
        description="Accumulated results keyed by document name"
    )
    internal_state: str = Field(
        default="",
        json_schema_extra={"display": False}  # Hidden from LLM
    )

class DocumentAnalyzer(AmphibiousAutoma[DocumentContext]):
    analyzer = think_unit(
        CognitiveWorker.inline("Analyze the current document."),
        max_attempts=5,
    )

    async def after_action(self, step_result, ctx: DocumentContext):
        """Keep custom context in sync with tool results."""
        action_result = step_result.result
        if not isinstance(action_result, ActionResult):
            return
        for step in action_result.results:
            if step.success and step.tool_name == "read_document":
                doc_name = step.tool_arguments.get("doc_name", "")
                ctx.current_document = doc_name
                ctx.analysis_results[doc_name] = step.tool_result

    async def on_agent(self, ctx: DocumentContext):
        await self.analyzer
```

## Phase Annotation

```python
class ContentCreator(AmphibiousAutoma[CognitiveContext]):
    researcher = think_unit(
        CognitiveWorker.inline("Research the topic thoroughly."),
        max_attempts=3,
    )
    writer = think_unit(
        CognitiveWorker.inline("Write the article using gathered research."),
        max_attempts=5,
    )

    async def on_agent(self, ctx: CognitiveContext):
        # Phase 1: Research
        async with self.snapshot(goal="Gather research material on renewable energy"):
            await self.researcher

        # Phase 2: Write
        async with self.snapshot(goal="Write the article using the research"):
            await self.writer
```

## Cognitive Policies

```python
# Enable all three policies
class AnalystAgent(AmphibiousAutoma[CognitiveContext]):
    analyst = think_unit(
        CognitiveWorker.inline(
            "Perform a comprehensive analysis.",
            enable_rehearsal=True,    # Mental simulation
            enable_reflection=True,   # Information assessment
            # Acquiring is always active by default
        ),
        max_attempts=10,
    )

    async def on_agent(self, ctx: CognitiveContext):
        await self.analyst
```

## OTC Hooks

### observation — Inject Custom Perception

```python
class SecurityWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return "Analyze the system for security issues."

    async def observation(self, context: CognitiveContext):
        return f"Security policy: Read-only audit mode."

class SecurityAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(SecurityWorker(), max_attempts=5)

    # Agent-level observation: shared across all workers
    async def observation(self, ctx: CognitiveContext):
        return f"System: production-server-01, Uptime: 45 days"

    async def on_agent(self, ctx):
        await self.auditor
```

### build_messages — Reshape LLM Messages

```python
from bridgic.core.model.types import Message

class StrictWorker(CognitiveWorker):
    async def thinking(self) -> str:
        return "Perform a security audit."

    async def build_messages(self, think_prompt, tools_description,
                             output_instructions, context_info):
        rules = "\n\nRULES:\n1. NEVER call delete_file.\n2. NEVER read .env files."
        system = f"{think_prompt}{rules}\n\n{tools_description}\n\n{output_instructions}"
        return [
            Message.from_text(text=system, role="system"),
            Message.from_text(text=context_info, role="user"),
        ]
```

### before_action — Filter Dangerous Calls

```python
class SafeAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(CognitiveWorker.inline("Audit the system."), max_attempts=5)

    async def before_action(self, decision_result, ctx):
        if isinstance(decision_result, list):
            blocked = {"delete_file", "drop_table"}
            return [(tc, ts) for tc, ts in decision_result
                    if ts.tool_name not in blocked] or decision_result
        return decision_result

    async def on_agent(self, ctx):
        await self.auditor
```

### after_action — Update Context After Execution

```python
class TrackingAgent(AmphibiousAutoma[MyContext]):
    worker = think_unit(CognitiveWorker.inline("Process data."), max_attempts=5)

    async def after_action(self, step_result, ctx: MyContext):
        action_result = step_result.result
        if isinstance(action_result, ActionResult):
            for r in action_result.results:
                if r.success:
                    ctx.processed_count += 1

    async def on_agent(self, ctx):
        await self.worker
```

### action_custom_output — Post-process Structured Output

```python
from pydantic import BaseModel

class AuditReport(BaseModel):
    findings: list[str]
    risk_level: str

class RedactingAgent(AmphibiousAutoma[CognitiveContext]):
    auditor = think_unit(
        CognitiveWorker.inline("Produce an audit report.", output_schema=AuditReport),
        max_attempts=1,
    )

    async def action_custom_output(self, decision_result, ctx):
        if isinstance(decision_result, AuditReport):
            decision_result.findings = [
                f.replace("sk-xxx", "[REDACTED]") for f in decision_result.findings
            ]
        return decision_result

    async def on_agent(self, ctx):
        await self.auditor
```

## Skills Usage

```python
from bridgic.amphibious import Skill

fundamental_skill = Skill(
    name="fundamental-analysis",
    description="Evaluate stock's intrinsic value using financial metrics",
    content="## Procedure\n1. Get financials\n2. Evaluate P/E ratio\n...",
)

agent = MyAgent(llm=llm)
result = await agent.arun(
    goal="Analyze AAPL stock",
    tools=[get_financials_tool],
    skills=[fundamental_skill],
)

# Or load from file
ctx = CognitiveContext(goal="...")
ctx.skills.add_from_file("skills/analysis/SKILL.md")
ctx.skills.load_from_directory("skills/")
```

## Memory Configuration

```python
from bridgic.amphibious import CognitiveHistory

# Short tasks: large working memory
history = CognitiveHistory(working_memory_size=10, short_term_size=30)

# Long tasks: aggressive compression
history = CognitiveHistory(
    working_memory_size=2,
    short_term_size=5,
    compress_threshold=3,
)

agent = MyAgent(llm=llm)
result = await agent.arun(
    goal="Long running task",
    tools=[...],
    cognitive_history=history,
)
```

## Conditional Loops

```python
class IterativeAgent(AmphibiousAutoma[CognitiveContext]):
    researcher = think_unit(
        CognitiveWorker.inline("Research ONE aspect of the topic."),
        max_attempts=10,
    )

    async def on_agent(self, ctx: CognitiveContext):
        # Loop until condition met
        await self.researcher.until(
            lambda ctx: len(ctx.cognitive_history) >= 3,
        )

        # Loop with dynamic override
        await self.researcher.until(
            lambda ctx: some_condition(ctx),
            max_attempts=50,
            tools=["search"],
        )
```

## Tool & Skill Filtering

```python
class MultiPhaseAgent(AmphibiousAutoma[CognitiveContext]):
    searcher = think_unit(
        CognitiveWorker.inline("Search for information."),
        max_attempts=5,
        tools=["search", "browse"],       # Only these tools visible
        skills=["research"],              # Only these skills visible
    )
    writer = think_unit(
        CognitiveWorker.inline("Write the report."),
        max_attempts=3,
        tools=["write_file"],
    )

    async def on_agent(self, ctx):
        await self.searcher
        await self.writer
```

## Execution Tracing

```python
agent = MyAgent(llm=llm, verbose=True)
result = await agent.arun(
    goal="...",
    tools=[...],
    trace_running=True,
)

# Access trace
trace = agent._agent_trace.build()
# trace["phases"]: steps grouped by self.snapshot() blocks
# trace["orphan_steps"]: steps outside any phase annotation

for step in trace["orphan_steps"]:
    content = step.step_content if hasattr(step, "step_content") else step.get("step_content", "")
    print(f"  {content[:80]}")
    tool_calls = step.tool_calls if hasattr(step, "tool_calls") else step.get("tool_calls", [])
    for tc in tool_calls:
        name = tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool_name", "?")
        print(f"    -> {name}")

# Save / Load
agent._agent_trace.save("trace.json")
loaded = AgentTrace.load("trace.json")  # Returns plain dict
```
