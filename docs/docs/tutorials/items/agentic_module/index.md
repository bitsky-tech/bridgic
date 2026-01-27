# Agentic Module

## Introduction

**Agentic Module** is important in the Bridgic framework for building agentic systems. It provides a comprehensive set of agentic classes in a systematic and modular way, derived from `GraphAutoma`. Whether you're building simple task orchestration or complex autonomous agent workflows, developers can flexibly combine various automas according to their business scenarios to build their own agentic systems. It also support full-process observability, helping you easily watch and debug every step of task execution, providing comprehensive support from task flow to state monitoring.

## Provided Agentic Modules

### ConcurrentAutoma

**ConcurrentAutoma** provides concurrent execution capabilities for multiple workers. All workers start simultaneously and execute concurrently, with results aggregated into a list upon completion.

**Features**:
- **Concurrent Execution**: All workers execute simultaneously, independent of each other
- **Result Aggregation**: Automatically aggregates all worker results into a list
- **Execution Modes**: Supports both async mode (Async Mode) and parallel mode (Parallel Mode)
- **Use Cases**: Scenarios requiring parallel execution of multiple independent tasks, such as calling multiple APIs simultaneously or processing multiple data sources in parallel

### SequentialAutoma

**SequentialAutoma** provides strict sequential execution capabilities. Workers execute in the order they are added, with each worker's output serving as the input for the next worker.

**Features**:
- **Strict Ordering**: Workers execute sequentially in the order they are registered
- **Data Passing**: The output of the previous worker is automatically passed to the next worker
- **Linear Workflow**: Ensures an ordered, step-by-step processing flow
- **Use Cases**: Scenarios requiring strict sequential dependencies, such as data processing pipelines or multi-step transformation processes

### ReCentAutoma

**ReCentAutoma** provides autonomous agent capabilities, enabling systems to autonomously make decisions and plan based on specific goals and dynamic environments at runtime. It is driven by Large Language Models (LLMs) at its core, allowing it to reason about the current state in stages based on high-level goals, dynamically select appropriate tools to invoke, and determine whether goals are achieved or actions should continue based on results.

**Core Features**:
- **Autonomous Reasoning and Planning**: LLMs can understand goals and context, automatically deciding the next action or tool combination
- **Intermediate Decision-Making and Looping**: Supports multi-round reasoning, with each decision based on the latest context results and historical trajectory
- **Goal-Driven Execution**: The entire process is anchored by "goal completion," automatically converging and ending; supports interruption and recovery
- **Memory Mechanism**: Built-in short-term and long-term memory capabilities, performing information compression and review for long-chain tasks
- **Adaptation to Complex Environments**: Capable of handling complex scenarios where task flows are uncertain and execution paths change dynamically

**Typical Use Cases**:
- Highly autonomous intelligent assistants and Q&A systems
- Complex tool and plugin orchestration (e.g., internet search + data analysis)
- Exploratory tasks with unknown requirements and processes

## Tutorials and Resources

- **[Core Mechanisms](../core_mechanism/index.md)**: Understand Bridgic's fundamental concepts, like `Worker`, `Automa`, etc.
- **[About ConcurrentAutoma](about_concurrent_automa.ipynb)**: Learn how to use `ConcurrentAutoma` through practical examples
- **[About SequentialAutoma](about_sequential_automa.ipynb)**: Learn how to use `SequentialAutoma` through practical examples
- **[About ReCentAutoma](about_recent_automa.ipynb)**: Learn how to use `ReCentAutoma` to build a more autonomous system