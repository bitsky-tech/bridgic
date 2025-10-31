# Agent Tracer

A framework-agnostic tracing library for AI agents and LLM applications. Built to integrate seamlessly with any agent framework while providing powerful observability through multiple tracing backends.

## Features

- 🔌 **Framework Agnostic**: Works with any Python-based agent framework (LangChain, CrewAI, AutoGPT, custom frameworks, etc.)
- 🎯 **Multiple Backends**: Support for LangSmith, LangFuse, LangWatch, Arize Phoenix, Opik, and Traceloop
- 🚀 **Easy Integration**: Simple API that doesn't require framework-specific knowledge
- 🔄 **Async Support**: Built with async/await for modern Python applications
- 📊 **Rich Context**: Capture inputs, outputs, metadata, logs, and errors
- 🛡️ **Privacy First**: Automatic masking of sensitive data (API keys, passwords, etc.)
- 🔗 **Hierarchical Tracing**: Support for nested traces (workflows → agents → tools)

## Installation

### Basic Installation

```bash
pip install agent-tracer
```

### With Specific Tracers

```bash
# Install with LangSmith support
pip install agent-tracer[langsmith]

# Install with LangFuse support
pip install agent-tracer[langfuse]

# Install with all tracers
pip install agent-tracer[all]
```

## Quick Start

### Basic Usage

```python
from agent_tracer import TracingService
import asyncio

# Initialize the tracing service
tracer = TracingService()

async def main():
    # Start tracing a workflow
    await tracer.start_trace(
        trace_id="workflow-123",
        trace_name="Customer Support Bot",
        project_name="MyProject"
    )
    
    # Trace a component/step
    async with tracer.trace_step(
        step_name="query_analysis",
        inputs={"query": "What's the weather?"}
    ):
        # Your agent logic here
        result = analyze_query("What's the weather?")
        
        # Set outputs
        tracer.set_outputs(
            step_name="query_analysis",
            outputs={"intent": "weather_query", "entities": ["weather"]}
        )
    
    # End the trace
    await tracer.end_trace(outputs={"response": "The weather is sunny"})

asyncio.run(main())
```

### Configuration

Configure tracers via environment variables:

```bash
# LangSmith
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_PROJECT="your-project"

# LangFuse
export LANGFUSE_SECRET_KEY="your-secret"
export LANGFUSE_PUBLIC_KEY="your-public"
export LANGFUSE_HOST="https://cloud.langfuse.com"

# LangWatch
export LANGWATCH_API_KEY="your-key"

# Arize Phoenix
export ARIZE_API_KEY="your-key"
export ARIZE_SPACE_ID="your-space-id"

# Opik
export OPIK_API_KEY="your-key"
export OPIK_WORKSPACE="your-workspace"

# Traceloop
export TRACELOOP_API_KEY="your-key"
```

## Framework Integration Examples

### LangChain Integration

```python
from agent_tracer import TracingService
from langchain.agents import AgentExecutor

tracer = TracingService()

async def run_langchain_agent(query: str):
    await tracer.start_trace(
        trace_id="lc-agent-run",
        trace_name="LangChain Agent",
        project_name="MyProject"
    )
    
    # Get LangChain callbacks
    callbacks = tracer.get_langchain_callbacks()
    
    # Run your agent with callbacks
    result = agent_executor.invoke(
        {"input": query},
        config={"callbacks": callbacks}
    )
    
    await tracer.end_trace(outputs={"result": result})
    return result
```

### CrewAI Integration

```python
from agent_tracer import TracingService
from crewai import Agent, Task, Crew

tracer = TracingService()

async def run_crew(task_description: str):
    await tracer.start_trace(
        trace_id="crew-run",
        trace_name="Research Crew",
        project_name="CrewAI"
    )
    
    # Trace individual agent tasks
    async with tracer.trace_step(
        step_name="researcher_task",
        inputs={"task": task_description}
    ):
        result = crew.kickoff()
        tracer.set_outputs("researcher_task", {"result": result})
    
    await tracer.end_trace(outputs={"final_result": result})
    return result
```

### Custom Framework Integration

```python
from agent_tracer import TracingService

tracer = TracingService()

class MyCustomAgent:
    def __init__(self):
        self.tracer = TracingService()
    
    async def run(self, task: str):
        # Start workflow trace
        await self.tracer.start_trace(
            trace_id="custom-agent",
            trace_name="Custom Agent Workflow",
            project_name="CustomFramework"
        )
        
        try:
            # Trace planning phase
            async with self.tracer.trace_step(
                step_name="planning",
                inputs={"task": task}
            ):
                plan = await self.plan(task)
                self.tracer.set_outputs("planning", {"plan": plan})
            
            # Trace execution phase
            async with self.tracer.trace_step(
                step_name="execution",
                inputs={"plan": plan}
            ):
                result = await self.execute(plan)
                self.tracer.set_outputs("execution", {"result": result})
            
            await self.tracer.end_trace(outputs={"final_result": result})
            return result
            
        except Exception as e:
            await self.tracer.end_trace(error=e)
            raise
```

## API Reference

### TracingService

#### Methods

- `start_trace(trace_id, trace_name, project_name, user_id=None, session_id=None)`: Start a new trace
- `trace_step(step_name, inputs, metadata=None)`: Context manager for tracing a step/component
- `set_outputs(step_name, outputs, metadata=None)`: Set outputs for current step
- `add_log(step_name, log)`: Add a log entry to current step
- `end_trace(outputs=None, error=None)`: End the current trace
- `get_langchain_callbacks()`: Get LangChain-compatible callbacks

### Configuration Options

```python
from agent_tracer import TracingService, TracingConfig

config = TracingConfig(
    deactivate_tracing=False,  # Global on/off switch
    mask_sensitive_data=True,  # Auto-mask API keys, passwords
    sensitive_keywords=["api_key", "password", "token"],  # Custom keywords to mask
)

tracer = TracingService(config=config)
```

## Advanced Usage

### Nested Traces

```python
async def complex_workflow():
    await tracer.start_trace(
        trace_id="main-workflow",
        trace_name="Multi-Agent System"
    )
    
    # Parent task
    async with tracer.trace_step("coordinator", {"task": "Analyze data"}):
        
        # Child task 1
        async with tracer.trace_step("data_fetcher", {"source": "db"}):
            data = fetch_data()
            tracer.set_outputs("data_fetcher", {"data": data})
        
        # Child task 2
        async with tracer.trace_step("analyzer", {"data": data}):
            analysis = analyze(data)
            tracer.set_outputs("analyzer", {"analysis": analysis})
        
        tracer.set_outputs("coordinator", {"complete": True})
    
    await tracer.end_trace(outputs={"status": "success"})
```

### Adding Logs

```python
async with tracer.trace_step("processing", {"input": data}):
    tracer.add_log("processing", {
        "name": "debug_info",
        "message": "Processing started",
        "type": "info"
    })
    
    result = process(data)
    
    tracer.add_log("processing", {
        "name": "performance",
        "message": f"Processed in {elapsed}ms",
        "type": "metric"
    })
    
    tracer.set_outputs("processing", {"result": result})
```

### Error Handling

```python
async def safe_execution():
    await tracer.start_trace(trace_id="safe-run", trace_name="Error Example")
    
    try:
        async with tracer.trace_step("risky_operation", {"param": value}):
            result = risky_function()
            tracer.set_outputs("risky_operation", {"result": result})
    except Exception as e:
        # Error is automatically captured
        await tracer.end_trace(error=e)
        raise
    else:
        await tracer.end_trace(outputs={"success": True})
```

## Architecture

```
agent-tracer/
├── src/
│   └── agent_tracer/
│       ├── __init__.py
│       ├── base.py              # BaseTracer abstract class
│       ├── schema.py            # Data schemas
│       ├── service.py           # TracingService main API
│       ├── utils.py             # Utility functions
│       └── tracers/
│           ├── __init__.py
│           ├── langsmith.py     # LangSmith implementation
│           ├── langfuse.py      # LangFuse implementation
│           ├── langwatch.py     # LangWatch implementation
│           ├── arize_phoenix.py # Arize Phoenix implementation
│           ├── opik.py          # Opik implementation
│           └── traceloop.py     # Traceloop implementation
├── tests/
├── examples/
│   ├── langchain_example.py
│   ├── crewai_example.py
│   └── custom_framework_example.py
├── pyproject.toml
└── README.md
```

## Design Principles

1. **Framework Agnostic**: No dependencies on specific agent frameworks
2. **Minimal Dependencies**: Core library has minimal deps; tracer-specific deps are optional
3. **Type Safe**: Full type hints for better IDE support
4. **Async First**: Built for modern async Python applications
5. **Privacy Focused**: Automatic sensitive data masking
6. **Extensible**: Easy to add new tracer backends

## Supported Tracers

| Tracer | Status | Features |
|--------|--------|----------|
| LangSmith | ✅ Full Support | LangChain callbacks, nested traces |
| LangFuse | ✅ Full Support | User/session tracking, hierarchical spans |
| LangWatch | ✅ Full Support | Thread tracking, component traces |
| Arize Phoenix | ✅ Full Support | OpenTelemetry, session tracking |
| Opik | ✅ Full Support | Thread/user tracking, metadata |
| Traceloop | ✅ Full Support | OpenTelemetry, custom attributes |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

Originally extracted from [Langflow](https://github.com/logspace-ai/langflow) and adapted to be framework-agnostic.

