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
        project_name="ExamplesNests",
    )

    try:
        # Trace step 1: Planning
        # async with tracer.trace_step(
        #     step_name="planning",
        #     inputs={"task": "Analyze customer feedback"},
        # ):
        #     # Your agent logic here
        #     plan = "1. Load feedback data\n2. Analyze sentiment\n3. Generate summary"
        plan = "1. Load feedback data\n2. Analyze sentiment\n3. Generate summary"
        #     # Add a log entry
        #     tracer.add_log(
        #         "planning",
        #         {"name": "debug", "message": "Plan created successfully", "type": "info"},
        #     )

        #     # Set outputs
        #     tracer.set_outputs("planning", {"plan": plan})

        # Trace step 2: Execution
        async with tracer.trace_step(
            step_name="execution",
            inputs={"plan": plan},
            metadata={
                "customer_id": "customer_123",
                "_opik_graph_definition": {
                    "format": "mermaid",
                    "data": "---\nconfig:\n  flowchart:\n    curve: linear\n---\ngraph TD;\n    __start__([<p>__start__</p>]):::first\n    handle_query(Handle Query)\n    classify_query(Classify Query)\n    contact_tool(Contact Tool)\n    sales_tool(Sales Tool)\n    fallback_tool(Fallback Tool)\n    rag_pipeline(RAG Pipeline)\n    format_response(Format Response)\n    __end__([<p>__end__</p>]):::last\n\n    __start__ --> handle_query;\n    handle_query --> classify_query;\n    classify_query -->|contact insights| contact_tool;\n    classify_query -->|sales analytics| sales_tool;\n    classify_query -->|other| fallback_tool;\n\n    contact_tool <--> rag_pipeline;\n    sales_tool <--> rag_pipeline;\n    contact_tool --> format_response;\n    sales_tool --> format_response;\n    fallback_tool --> format_response;\n    format_response --> __end__;\n\n    classDef default fill:#f2f0ff,line-height:1.2\n    linkStyle 6 stroke:#008b00, stroke-width:2px\n    linkStyle 7 stroke:#008b00, stroke-width:2px\n    classDef first fill-opacity:0\n    classDef last fill:#bfb6fc\n"
                },
                "tool_call": "fallback"
            }
        ) as execution_span:
            await asyncio.sleep(0)
            result = "Positive sentiment: 80%, Negative: 20%"

            tracer.set_outputs("execution", {"result": result})
            # run in loop
            for i in range(3):
                async with tracer.trace_step(
                    step_name=f"execution_{i}",
                    inputs={"plan": plan},
                    parent_step_trace_context=execution_span,
                ):
                    result = f"Positive sentiment: 80%, Negative: 20% - {i}"
                    tracer.set_outputs(f"execution_{i}", {"result": result, "iteration": i})
                    await asyncio.sleep(0)
            # Give the worker a chance to flush child span starts before ending parent
            await asyncio.sleep(0)

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

