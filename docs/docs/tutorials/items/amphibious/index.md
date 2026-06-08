# Amphibious

Bridgic Amphibious is a **dual-mode agent framework** that lets you build agents operating in both LLM-driven and deterministic modes, with automatic fallback between them. Instead of choosing between full autonomy and rigid workflows, Amphibious gives you both — in the same agent.

The framework is built on a few core design principles:

- **Two loops, two contexts** — every agent is parameterized by a small-loop `OTAContext` (framework-owned: this run's `user_input`, its observe-think-act round trace, and the tools it carries) and a free-form big-loop `Context` (cross-turn knowledge you render into the prompt). You write `AmphibiousAutoma[OTAContext, Context]`.
- **Agent = Think Units + Orchestration** — an agent is defined by declaring `CognitiveWorker` think units (and `AgentWorker` think agents) and orchestrating them with LLM reasoning (`on_agent`) or developer-defined workflows (`on_workflow`), rather than wiring low-level LLM calls by hand.
- **Tools are declared, not injected** — capabilities live on the OTA context. You declare exactly the tools a run carries via `OTAContext.tool(...)`; nothing is auto-injected.
- **Human-in-the-Loop** — two entry points share one `@human_channel` registry: the deterministic `HumanCall` yield (from `on_workflow`), and the LLM-driven `request_human` tool (declared on the OTA context, callable from any think unit).

## Essential

These tutorials cover the fundamentals you need to build amphibious agents:

1. [Quick Start](../amphibious/quick_start.ipynb): Build your first amphibious agent in 5 minutes — run both Agent mode and Workflow mode to see the dual-mode experience firsthand.
2. [Dual-Mode Orchestration](../amphibious/think.ipynb): Master the two orchestration modes — `on_agent` for LLM-driven decision making and `on_workflow` for deterministic step-by-step execution — and `EnterAgent` to switch between them.
3. [CognitiveWorker & think_unit](../amphibious/cognitive_worker.ipynb): Understand the framework's atomic building block — the in-process think unit whose single `thinking()` method assembles a prompt and calls the model — plus its declarative `think_unit` configuration (`max_attempts`, `until`, error strategies).
4. [Built-in Tools](../amphibious/built_in_tools.ipynb): Declare the seven shipped tools (bash, read_file/write_file/edit_file, glob, grep, request_human) on your OTA context via `OTAContext.tool`, and use them in both modes — including the read-before-modify safety model.
5. [RunMode](../amphibious/automa_mode.ipynb): Explore the four run modes and learn how Amphiflow mode recovers from a failed workflow step by running a bounded agent sub-run.
6. [Customizing the OTA Cycle](../amphibious/custom_otc.ipynb): Override the hooks of the Observe-Think-Act cycle — inject custom observations, reshape the decision before the act phase, and react to results afterward.
7. [Execution Tracing](../amphibious/execution_tracing.ipynb): Record, export, and analyze the full execution trace of your agent for debugging and optimization.

This architecture makes Bridgic Amphibious a powerful platform for building agents that are both reliable and adaptive — bridging the precision of deterministic workflows with the creative problem-solving of LLM reasoning.
