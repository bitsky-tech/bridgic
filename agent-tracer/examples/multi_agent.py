"""Multi-agent system example with nested tracing."""

import asyncio
from typing import Any

from agent_tracer import TracingService


class SimpleAgent:
    """A simple agent that can be part of a multi-agent system."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    async def process(self, task: str) -> dict[str, Any]:
        """Process a task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "agent": self.name,
            "role": self.role,
            "result": f"{self.role} completed: {task}",
        }


class MultiAgentSystem:
    """A multi-agent system with coordinated tracing."""

    def __init__(self):
        self.tracer = TracingService()
        self.agents = [
            SimpleAgent("Agent-A", "Researcher"),
            SimpleAgent("Agent-B", "Analyst"),
            SimpleAgent("Agent-C", "Writer"),
        ]

    async def run(self, main_task: str) -> dict[str, Any]:
        """Run a multi-agent workflow with hierarchical tracing."""
        # Start main workflow trace
        await self.tracer.start_trace(
            trace_name="Multi-Agent System",
            project_name="Multi-Agent Examples",
        )

        try:
            all_results = []

            # Coordinator phase
            async with self.tracer.trace_step(
                step_name="coordinator",
                inputs={"main_task": main_task},
                trace_type="chain",
            ):
                # Decompose task into subtasks
                subtasks = [
                    "Research the topic",
                    "Analyze the findings",
                    "Write the report",
                ]

                self.tracer.add_log(
                    "coordinator",
                    {
                        "name": "task_decomposition",
                        "message": f"Decomposed into {len(subtasks)} subtasks",
                        "type": "info",
                    },
                )

                # Execute each subtask with a different agent
                for i, (agent, subtask) in enumerate(zip(self.agents, subtasks)):
                    async with self.tracer.trace_step(
                        step_name=f"agent_{agent.name}",
                        inputs={"subtask": subtask, "agent_role": agent.role},
                        trace_type="agent",
                    ):
                        result = await agent.process(subtask)
                        all_results.append(result)

                        self.tracer.add_log(
                            f"agent_{agent.name}",
                            {
                                "name": "agent_completed",
                                "message": f"{agent.name} completed subtask",
                                "type": "info",
                            },
                        )

                        self.tracer.set_outputs(f"agent_{agent.name}", {"result": result})

                # Aggregate results
                final_output = {
                    "main_task": main_task,
                    "agents_used": len(self.agents),
                    "results": all_results,
                    "status": "completed",
                }

                self.tracer.set_outputs("coordinator", {"final_output": final_output})

            # End trace
            await self.tracer.end_trace(outputs=final_output)

            return final_output

        except Exception as e:
            await self.tracer.end_trace(error=e)
            raise


async def main():
    """Run the multi-agent example."""
    system = MultiAgentSystem()

    print("🤖 Starting multi-agent system with tracing...")

    result = await system.run("Create a comprehensive market analysis report")

    print(f"\n✅ Multi-agent system completed!")
    print(f"Main task: {result['main_task']}")
    print(f"Agents used: {result['agents_used']}")
    print(f"Status: {result['status']}")
    print(f"\n📊 Results:")
    for r in result["results"]:
        print(f"  - {r['agent']} ({r['role']}): {r['result']}")
    print(f"\nCheck your configured tracing backends for hierarchical traces")


if __name__ == "__main__":
    asyncio.run(main())

