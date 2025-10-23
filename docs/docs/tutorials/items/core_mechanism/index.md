# Core Mechanism

Bridgic lets you build agentic systems by breaking down your workflows into modular building blocks called worker. Each worker represents a specific task or behavior, making it easy to organize complex processes.

Bridgic introduces clear abstractions for structuring flows, passing data between execution units, handling concurrency, and enabling dynamic control logic (such as conditional branching and routing). This design allows users to build systerm that scale efficiently from simple workflows to sophisticated agentic systems.

Key features include:

1. [Concurrency Mode](../core_mechanism/concurrency_mode): Organize your concurrent execution units systematically and conveniently.
2. [Parameter Binding](../core_mechanism/parameter_binding): Explore three ways for passing data between execution units, including Arguments Mapping, Arguments Injection, and Inputs Propagation.
3. [Dynamic Routing](../core_mechanism/dynamic_routing): Decide which execution unit to be executed in the short future dynamically.
4. [Modularity](../core_mechanism/modularity): Reuse and compose Automata by embedding one inside another for scalable workflows.
5. [Model Integration](../model_integration/llm_integration): Incorporate model to building a program with more autonomous capabilities.
6. [Human-in-the-loop](../core_mechanism/human_in_the_loop): Enable human interaction or external input during workflow execution.

This architectural foundation makes Bridgic a powerful platform for building agentic systems that are robust, adaptive, and easy to reason about, enabling you to bridge logic with the creative potential of AI.
