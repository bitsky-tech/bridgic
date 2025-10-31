"""Custom agent framework integration example."""

import asyncio
from typing import Any

from agent_tracer import TracingService


class CustomAgent:
    """A simple custom agent implementation with tracing."""

    def __init__(self, name: str):
        self.name = name
        self.tracer = TracingService()

    async def plan(self, task: str) -> dict[str, Any]:
        """Planning phase of the agent."""
        # Simulate planning
        await asyncio.sleep(0.1)
        return {
            "steps": ["Step 1: Analyze", "Step 2: Execute", "Step 3: Report"],
            "estimated_time": "5 minutes",
        }

    async def execute_step(self, step: str) -> dict[str, Any]:
        """Execute a single step."""
        # Simulate execution
        await asyncio.sleep(0.2)
        return {"step": step, "status": "completed", "data": f"Result from {step}"}

    async def run(self, task: str) -> dict[str, Any]:
        """Run the agent workflow with full tracing."""
        # Start workflow trace
        await self.tracer.start_trace(
            trace_name=f"{self.name} Workflow",
            project_name="Custom Framework",
        )

        try:
            results = []

            # Phase 1: Planning
            async with self.tracer.trace_step(
                step_name="planning",
                inputs={"task": task},
                trace_type="tool",
            ):
                plan = await self.plan(task)

                self.tracer.add_log(
                    "planning",
                    {
                        "name": "plan_created",
                        "message": f"Created plan with {len(plan['steps'])} steps",
                        "type": "info",
                    },
                )

                self.tracer.set_outputs("planning", {"plan": plan})

            # Phase 2: Execution
            for i, step in enumerate(plan["steps"]):
                async with self.tracer.trace_step(
                    step_name=f"execute_step_{i}",
                    inputs={"step": step, "step_number": i + 1},
                    trace_type="tool",
                ):
                    result = await self.execute_step(step)
                    results.append(result)

                    self.tracer.set_outputs(f"execute_step_{i}", {"result": result})

            # Phase 3: Finalization
            async with self.tracer.trace_step(
                step_name="finalization",
                inputs={"results": results},
                trace_type="chain",
            ):
                summary = {
                    "total_steps": len(results),
                    "completed_steps": len([r for r in results if r["status"] == "completed"]),
                    "final_data": [r["data"] for r in results],
                }

                self.tracer.set_outputs("finalization", {"summary": summary})

            # End trace successfully
            await self.tracer.end_trace(outputs={"status": "success", "summary": summary})

            return summary

        except Exception as e:
            await self.tracer.end_trace(error=e)
            raise


async def main():
    """Run the custom agent example."""
    agent = CustomAgent("MyCustomAgent")

    print("🚀 Starting custom agent with tracing...")

    result = await agent.run("Process customer data and generate insights")

    print(f"\n✅ Agent completed successfully!")
    print(f"Summary: {result}")
    print(f"\nCheck your configured tracing backends for detailed traces")


if __name__ == "__main__":
    asyncio.run(main())

