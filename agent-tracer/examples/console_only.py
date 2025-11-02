"""Example: Console-only tracing (no remote services)."""

import asyncio

from agent_tracer import TracingService, TracingConfig


async def main():
    """Example of using console-only tracing."""
    # Configure to only use console output
    # By default, enable_console=True, so console output is enabled
    # Other tracers (LangFuse, LangWatch, Opik, etc.) will only initialize if their
    # environment variables are set
    
    # Option 1: Use default config (console enabled by default)
    tracer = TracingService()
    
    # Option 2: Explicitly enable console only (disable remote tracers by not setting env vars)
    # config = TracingConfig(enable_console=True)
    # tracer = TracingService(config=config)
    
    # Option 3: Disable console output if you only want remote tracers
    # config = TracingConfig(enable_console=False)
    # tracer = TracingService(config=config)

    # Start a trace for a workflow
    await tracer.start_trace(
        trace_name="Console-Only Example",
        project_name="Examples",
    )

    try:
        # Trace step 1: Planning
        async with tracer.trace_step(
            step_name="planning",
            inputs={"task": "Analyze customer feedback"},
        ):
            # Your agent logic here
            plan = "1. Load feedback data\n2. Analyze sentiment\n3. Generate summary"

            # Add a log entry
            tracer.add_log(
                "planning",
                {"name": "debug", "message": "Plan created successfully", "type": "info"},
            )

            # Set outputs
            tracer.set_outputs("planning", {"plan": plan})

        # Trace step 2: Execution
        async with tracer.trace_step(
            step_name="execution",
            inputs={"plan": plan},
        ):
            result = "Positive sentiment: 80%, Negative: 20%"

            tracer.set_outputs("execution", {"result": result})
            
            # Add some logs
            tracer.add_log(
                "execution",
                {"name": "progress", "message": "Processing feedback data", "type": "info"},
            )
            tracer.add_log(
                "execution",
                {"name": "warning", "message": "Some feedback entries missing", "type": "warning"},
            )

        # End the trace successfully
        await tracer.end_trace(outputs={"status": "success", "final_result": result})

        print("\n✅ Tracing completed! All output was shown in console above.")

    except Exception as e:
        # End trace with error
        await tracer.end_trace(error=e)
        raise


if __name__ == "__main__":
    asyncio.run(main())

