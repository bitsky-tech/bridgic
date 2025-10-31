# Quick Start Guide

Get started with agent-tracer in 5 minutes!

## Installation

```bash
# Basic installation
pip install agent-tracer

# With specific tracer
pip install agent-tracer[langsmith]
pip install agent-tracer[langfuse]

# With all tracers
pip install agent-tracer[all]
```

## Configuration

Set up your tracing backend(s) via environment variables:

### LangSmith

```bash
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="your-project-name"
```

### LangFuse

```bash
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

### LangWatch

```bash
export LANGWATCH_API_KEY="your-api-key"
```

### Arize Phoenix

```bash
# For Arize Cloud
export ARIZE_API_KEY="your-api-key"
export ARIZE_SPACE_ID="your-space-id"

# For Phoenix OSS (local)
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006"
```

### Opik

```bash
export OPIK_API_KEY="your-api-key"
export OPIK_WORKSPACE="your-workspace"
```

### Traceloop

```bash
export TRACELOOP_API_KEY="your-api-key"
```

### Console Output (Default)

The console tracer is enabled by default and outputs all traces to your terminal/console. No configuration needed!

To disable console output:
```python
from agent_tracer import TracingService, TracingConfig

config = TracingConfig(enable_console=False)
tracer = TracingService(config=config)
```

To enable console output only (no remote services):
- Simply don't set any environment variables for remote tracers
- Console output is enabled by default
- All trace data will be printed to your terminal

## Your First Trace

Create a file `my_agent.py`:

```python
import asyncio
from agent_tracer import TracingService

async def main():
    tracer = TracingService()
    
    # Start tracing
    await tracer.start_trace(
        trace_name="My First Agent",
        project_name="QuickStart"
    )
    
    try:
        # Trace a step
        async with tracer.trace_step(
            step_name="processing",
            inputs={"query": "Hello World"}
        ):
            # Your agent logic
            result = "Processed: Hello World"
            
            # Set outputs
            tracer.set_outputs("processing", {"result": result})
        
        # End successfully
        await tracer.end_trace(outputs={"final": result})
        print("✅ Trace completed!")
        
    except Exception as e:
        await tracer.end_trace(error=e)
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python my_agent.py
```

## Check Your Traces

Visit your configured tracing platform to see the results:

- **LangSmith**: https://smith.langchain.com
- **LangFuse**: https://cloud.langfuse.com
- **LangWatch**: https://app.langwatch.ai
- **Arize Phoenix**: https://app.phoenix.arize.com
- **Opik**: https://www.comet.com/opik
- **Traceloop**: https://app.traceloop.com

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for more use cases
2. **Integrate with Your Framework**: See integration guides in README.md
3. **Advanced Features**: Learn about nested tracing, custom metadata, and more

## Common Patterns

### Error Handling

```python
try:
    async with tracer.trace_step("risky_operation", {"input": data}):
        result = risky_function()
        tracer.set_outputs("risky_operation", {"result": result})
except Exception as e:
    await tracer.end_trace(error=e)
    raise
```

### Adding Logs

```python
async with tracer.trace_step("analysis", {"data": data}):
    tracer.add_log("analysis", {
        "name": "debug",
        "message": "Starting analysis",
        "type": "info"
    })
    
    result = analyze(data)
    
    tracer.add_log("analysis", {
        "name": "metrics",
        "message": f"Processed {len(data)} items",
        "type": "metric"
    })
    
    tracer.set_outputs("analysis", {"result": result})
```

### LangChain Integration

```python
from langchain.agents import AgentExecutor

# Get callbacks
callbacks = tracer.get_langchain_callbacks()

# Use with LangChain
result = agent_executor.invoke(
    {"input": query},
    config={"callbacks": callbacks}
)
```

## Troubleshooting

### No traces appearing?

1. Check environment variables are set correctly
2. Verify API keys are valid
3. Check network connectivity
4. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Multiple tracers not working?

- Ensure all required packages are installed
- Each tracer requires specific environment variables
- Check logs for initialization errors

### Performance concerns?

- Tracing runs in background workers by default
- Minimal overhead on your application
- Disable tracing in production if needed:
  ```python
  from agent_tracer import TracingService, TracingConfig
  
  config = TracingConfig(deactivate_tracing=True)
  tracer = TracingService(config=config)
  ```

## Need Help?

- 📖 Read the full documentation: [README.md](README.md)
- 💬 Open an issue: [GitHub Issues](https://github.com/yourusername/agent-tracer/issues)
- 💡 Check examples: [examples/](examples/)
- 🤝 Contribute: [CONTRIBUTING.md](CONTRIBUTING.md)

