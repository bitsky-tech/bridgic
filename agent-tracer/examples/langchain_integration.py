"""LangChain integration example."""

import asyncio
import os

from agent_tracer import TracingService


async def main():
    """Example of using agent-tracer with LangChain."""
    # Check if LangChain is available
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain.tools import Tool
    except ImportError:
        print("❌ LangChain not installed. Install with: pip install langchain langchain-openai")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to run this example.")
        return

    # Initialize the tracing service
    tracer = TracingService()

    # Start a trace
    await tracer.start_trace(
        trace_name="LangChain Agent",
        project_name="LangChain Examples",
    )

    try:
        # Define a simple tool
        def get_current_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"The weather in {location} is sunny and 72°F"

        tools = [
            Tool(
                name="Weather",
                func=get_current_weather,
                description="Get the current weather for a location",
            )
        ]

        # Create LangChain agent
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        )

        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # Get LangChain callbacks from tracer
        callbacks = tracer.get_langchain_callbacks()

        # Trace the agent execution
        async with tracer.trace_step(
            step_name="agent_execution",
            inputs={"query": "What's the weather in San Francisco?"},
        ):
            result = agent_executor.invoke(
                {"input": "What's the weather in San Francisco?"},
                config={"callbacks": callbacks},
            )

            tracer.set_outputs("agent_execution", {"result": result})

        # End the trace
        await tracer.end_trace(outputs={"final_result": result})

        print("\n✅ LangChain agent execution completed!")
        print(f"Result: {result}")

    except Exception as e:
        await tracer.end_trace(error=e)
        raise


if __name__ == "__main__":
    asyncio.run(main())

