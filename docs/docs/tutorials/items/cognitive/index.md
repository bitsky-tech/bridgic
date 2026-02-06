# Cognitive Module

The Cognitive Module provides a framework for building fully custom LLM-powered agents with structured thinking, progressive information disclosure, and layered memory. It lets you conveniently develop agents with full control over every stage of agent behavior -- from how the agent observes and thinks, to how it selects and verifies tools, to how it processes results.

The module is built around three core concepts:

**AgentAutoma** -- The orchestrator. You declare thinking steps and wire them together using plain Python async control flow (`for`, `while`, `if/else`). This gives you full flexibility to build React-style loops, Plan-then-execute pipelines, or any hybrid pattern.

**CognitiveWorker** -- The thinker. Each worker represents one "observe-think-act" unit. It supports two thinking modes: `FAST` (merged thinking + tool selection in one LLM call) and `DEFAULT` (separate thinking and tool selection phases). Override its template methods to customize every stage of it.

**CognitiveContext** -- The memory. Built on the Exposure pattern, it provides layered information management: tools are always fully visible (`EntireExposure`), while history and skills support progressive disclosure (`LayeredExposure`) -- the LLM sees summaries first and can request details on demand.

Tutorials:

1. [Agent Automa](../cognitive/agent_automa.ipynb): Build a travel planning agent using all three core components. Learn how to define workers, declare thinking steps, and compose different agent strategies (React, Plan, Mix, conditional fallback).
2. [Customize Thinking](../cognitive/customize_thinking.ipynb): Deep dive into CognitiveWorker's template methods. Override `observation()`, `build_thinking_prompt()`, `verify_tools()`, `consequence()`, and `select_tools()` to customize every stage of the thinking cycle.
3. [Custom Context](../cognitive/custom_context.ipynb): Extend CognitiveContext with your own data using the Exposure pattern. Build custom `EntireExposure` and `LayeredExposure` fields to give your agent domain-specific knowledge.
