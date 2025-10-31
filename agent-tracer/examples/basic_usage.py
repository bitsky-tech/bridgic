"""Basic usage example of agent-tracer."""

import asyncio

from agent_tracer import TracingService


async def main():
    """Basic tracing example."""
    # Initialize the tracing service
    tracer = TracingService()

    # Start a trace for a workflow
    await tracer.start_trace(
        trace_name="Basic Agent Workflow",
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

        # End the trace successfully
        await tracer.end_trace(outputs={"status": "success", "final_result": result})

        print("✅ Tracing completed successfully!")
        print(f"Check your configured tracing backends for results")

    except Exception as e:
        # End trace with error
        await tracer.end_trace(error=e)
        raise


if __name__ == "__main__":
    asyncio.run(main())

