**Bridgic** is an agentic programming framework built around a novel **dynamic topology** orchestration model and a **component-oriented paradigm** that is realized through **ASL (Agent Structure Language)**—a powerful declarative DSL for composing, reusing, and nesting agentic structures. Together, these elements make it possible to develop the entire spectrum of agentic systems, ranging from deterministic workflows to autonomous agents.

## Core Features

* **Orchestration**: Bridgic introduces a novel orchestration model based on DDG (Dynamic Directed Graph).
* **Dynamic Routing**: Bridgic enables conditional branching and dynamic orchestration through an easy-to-use `ferry_to()` API.
* **Dynamic Topology**: The DDG-based orchestration topology can be changed at runtime in Bridgic to support highly autonomous AI applications.
* **ASL**: ASL (Agent Structure Language) is a powerful declarative DSL that embodies a component-oriented paradigm and is even capable of supporting dynamic topologies.
* **Modularity & Componentization**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Parameter Resolving**: Two mechanisms are designed to pass data among workers/automas—thereby eliminating the complexity of global state management when necessary.
* **Human-in-the-Loop**: A Bridgic-style agentic system can request feedback from human whenever needed to dynamically adjust its execution logic.
* **Serialization**: Bridgic employs a scalable serialization and deserialization mechanism to achieve state persistence and recovery, enabling human-in-the-loop in long-running AI systems.
* **Systematic Integration**: A wide range of tools, LLMs and tracing functionalities can be seamlessly integrated into the Bridgic world, in a systematic way.
* **Customization**: What Bridgic provides is not a "black box" approach. You have full control over every aspect of your AI applications, such as prompts, context windows, the control flow, and more.
