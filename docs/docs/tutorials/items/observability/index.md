# Observability

## Why Observability Matters

Developing and managing agentic systems comes with its own challenges. Unlike traditional software where execution flows are predictable, agentic systems involve dynamic decision-making, complex workflows, and interactions with external services. Without proper observability, these systems can become "black boxes"—you know inputs and outputs, but have little insight into what happens in between.

Effective observability provides several critical benefits:

- **Understanding Runtime Behavior**: See what's actually happening as your system executes, including which tasks are running, their execution order, and how they interact with each other.

- **Task Level Visibility**: Track inputs and outputs at the granularity of individual task units, enabling you to understand data flow, identify where transformations occur, and debug issues at the right level of detail.

- **Performance Optimization**: Help measure execution times and identify performance bottlenecks of your agentic system built with Bridgic. This visibility is essential for continuously optimizing agentic systems.


## How Observability Works

### Worker Execution

In the Bridgic framework, each task execution happens at the **Worker** granularity. Worker is the basic unit that represents a discrete task which can be executed. This worker-centric architecture provides natural boundaries for observability instrumentation, regardless of complexity. Whether you're building a simple sequential workflow or a complex graph-based agentic system (like ReAct), every operation runs as a worker.

### Callback Mechanism

Bridgic provides a flexible callback mechanism that hooks into worker execution at key lifecycle points. This callback system is designed to be non-intrusive—callbacks observe and instrument execution without modifying your core business logic. Refer to [Worker Callback Mechanism](../core_mechanism/worker_callback.md) for more about how this mechanism works in detail.


## Available Third-Party Integrations

Bridgic currently provides integrations with the following observability platforms:

- [Opik Integration](opik_integration)
