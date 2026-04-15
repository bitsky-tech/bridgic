# Amphibious

Bridgic Amphibious is a **dual-mode agent framework** that lets you build agents operating in both LLM-driven and deterministic modes, with automatic fallback between them. Instead of choosing between full autonomy and rigid workflows, Amphibious gives you both — in the same agent.

The framework is built on two core design principles:

- **Agent = Think Units + Context Orchestration** — An agent is defined by declaring `CognitiveWorker` think units and orchestrating them with scoped context, rather than wiring low-level LLM calls manually.
- **Functional Execution vs. Decision Making — Decoupled** — Tools and skills are pure capabilities, while *when* and *how* to invoke them is handled by either LLM reasoning (`on_agent`) or developer-defined workflows (`on_workflow`).
- **Human-in-the-Loop** — Three entry points for human interaction: code-level `request_human()`, workflow `HumanCall` yield, and the auto-injected `request_human` tool for LLM-autonomous requests.

## Essential

These tutorials cover the fundamental concepts you need to build amphibious agents:

1. [Quick Start](../amphibious/quick_start.ipynb): Build your first amphibious agent in 5 minutes — run both Agent mode and Workflow mode to see the dual-mode experience firsthand.
2. [Dual-Mode Orchestration](../amphibious/think.ipynb): Master the two orchestration modes — `on_agent` for LLM-driven decision making and `on_workflow` for deterministic step-by-step execution.
3. [RunMode](../amphibious/automa_mode.ipynb): Explore the four run modes and learn how Amphiflow mode automatically switches between Workflow and Agent when things go wrong.
4. [Customizing the OTC Cycle](../amphibious/custom_otc.ipynb): Override hooks in the Observe-Think-Act cycle — inject custom observations, reshape LLM messages, intercept tool calls, and post-process outputs.
5. [Execution Tracing](../amphibious/execution_tracing.ipynb): Record, export, and analyze the full execution trace of your agent for debugging and optimization.

## Advanced

These tutorials dive deeper into specific capabilities for advanced use cases:

6. [Phase Annotation](../amphibious/phase_annotation.ipynb): Organize complex agent execution into structured phases using the `snapshot` context manager.
7. [CognitiveWorker & think_unit](../amphibious/cognitive_worker.ipynb): Understand the framework's atomic building block — the pure thinking unit that only decides *what to do*, plus its declarative execution configuration and error strategies.
8. [Context & Exposure](../amphibious/context_and_exposure.ipynb): Control what the LLM sees — manage agent state with Context and choose between full disclosure and progressive reveal with Exposure strategies.
8. [Cognitive Policies](../amphibious/cognitive_policies.ipynb): Enable multi-round deliberation before action — let the agent gather details (Acquiring), mentally rehearse plans (Rehearsal), and assess information quality (Reflection).
10. [Cognitive History](../amphibious/cognitive_history.ipynb): Understand the three-tier memory system — Working Memory, Short-term Memory, and Long-term Memory — that automatically manages execution history.


This architecture makes Bridgic Amphibious a powerful platform for building agents that are both reliable and adaptive — bridging the precision of deterministic workflows with the creative problem-solving of LLM reasoning.
