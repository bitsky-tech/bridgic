# ReCentAutoma

## Deep Dive into ReCENT Memory Algorithm

This document provides an in-depth exploration of ReCentAutoma, its underlying ReCENT (Recursive Compressed Episodic Node Tree) algorithm, and how it addresses critical challenges in long-running autonomous agent systems.

## Problem Statement

### Context Explosion

Traditional ReAct-based agents face a fundamental limitation: as conversation history grows linearly, the context window eventually overflows. This leads to:

- **Performance Degradation**: LLM performance decreases as context approaches token limits
- **Complete Failure**: Context overflow causes task failures
- **Inefficient Resource Usage**: Processing increasingly large contexts wastes computational resources

### Goal Drift

Without explicit goal management, agents tend to deviate from original objectives during extended execution:

- **Loss of Focus**: Agents forget or lose track of the original task
- **Suboptimal Solutions**: Agents may pursue local optima instead of the global goal
- **Incomplete Tasks**: Tasks may be abandoned before completion

## ReCENT Algorithm Overview

### Core Design Principles

ReCENT addresses these challenges through:

1. **Goal-Oriented Design**: Explicit goal management prevents drift
2. **Automatic Memory Compression**: Recursive compression mechanism summarizes history when thresholds are reached
3. **Enhanced Observation Loop**: Improved ReAct flow with dedicated goal achievement evaluation
4. **Dynamic Worker Creation**: Runtime creation of tool workers based on LLM decisions

### Algorithm Components

#### Episodic Memory Structure

ReCENT uses an episodic memory tree (`EpisodicNodeTree`) to organize conversation history:

- **GoalEpisodicNode**: Stores task goal and guidance (never compressed)
- **LeafEpisodicNode**: Stores original message sequences (can be compressed)
- **CompressionEpisodicNode**: Stores summaries of compressed nodes (supports recursive compression)

#### Memory Compression Mechanism

Compression is triggered when:

- **Node Count Threshold**: Number of nodes exceeds `max_node_size`
- **Token Count Threshold**: Estimated token count exceeds `max_token_size`

Compression process:

1. Identifies all non-goal nodes to compress
2. Creates a compression node placeholder
3. Asynchronously generates summary using LLM
4. Updates memory structure to reference compression node

#### Recursive Compression

Compression nodes can themselves be compressed, creating a recursive structure:

- **Multi-Level Summarization**: Deep conversation histories are summarized at multiple levels
- **Preserved Context**: Important information is retained through summarization
- **Efficient Storage**: Memory footprint grows sub-linearly with conversation length

## Architecture Deep Dive

### Worker Orchestration

ReCentAutoma leverages Bridgic's dynamic topology capabilities:

#### Core Workers

1. **`initialize_task_goal`**: Entry point that creates the initial goal node
2. **`observe`**: Evaluates goal achievement status using LLM with structured output
3. **`select_tools`**: Selects and dynamically creates tool workers
4. **`compress_memory`**: Checks and triggers memory compression when needed
5. **`finalize_answer`**: Generates the final comprehensive answer

#### Dynamic Worker Creation

- **Tool Workers**: Created dynamically based on LLM tool selection
- **Collect Results Worker**: Created to gather tool execution results
- **Automatic Cleanup**: Temporary workers are automatically removed after execution

#### Execution Flow

1. **Initialization**: Goal node is created
2. **Observation Loop**: Agent evaluates current state and goal achievement
3. **Tool Selection**: LLM selects tools to use (if goal not achieved)
4. **Concurrent Execution**: Tool execution and memory compression proceed in parallel
5. **Result Collection**: Tool results are collected and added to memory
6. **Iteration**: Process repeats until goal is achieved or stop conditions are met

### Memory Management

#### ReCentMemoryManager

Manages episodic memory through the `EpisodicNodeTree`:

- **Goal Management**: Creates and updates goal nodes
- **Message Appending**: Adds messages to current leaf node
- **Compression Triggering**: Checks if compression is needed
- **Context Building**: Constructs context from memory for LLM calls

#### Context Building Strategy

When building context for LLM calls:

1. Goal information is always included
2. Non-goal nodes are traversed in timestep order
3. Leaf nodes contribute original messages
4. Compression nodes contribute summaries (waited for if still generating)
5. Result is a compact, goal-focused context

### Configuration System

#### Memory Configuration

`ReCentMemoryConfig` controls compression behavior:

- **Thresholds**: Node and token count limits
- **Compression LLM**: LLM used for generating summaries
- **Compression Prompts**: Templates for compression summarization

#### Task Configurations

Separate configurations for different tasks:

- **ObservationTaskConfig**: Goal achievement evaluation
- **ToolTaskConfig**: Tool selection
- **AnswerTaskConfig**: Final answer generation

#### Stop Conditions

Configurable stop conditions:

- **Max Iterations**: Limit total number of observation cycles
- **Consecutive No-Tool-Selected**: Stop if agent repeatedly chooses no tools

## Advanced Topics

### Goal Management

Goals in ReCentAutoma serve multiple purposes:

- **Task Definition**: Clearly defines what the agent should accomplish
- **Drift Prevention**: Provides reference point to prevent deviation
- **Achievement Evaluation**: Enables structured evaluation of progress
- **Context Focus**: Helps compression preserve goal-relevant information

### Compression Quality

The effectiveness of ReCENT depends on compression quality:

- **Summary Fidelity**: How well summaries preserve important information
- **Goal Awareness**: Compression should be goal-aware to preserve relevant details
- **Recursive Summarization**: Multi-level compression requires careful prompt design

### Concurrency and Performance

ReCentAutoma uses concurrency effectively:

- **Parallel Tool Execution**: Multiple tools can execute concurrently
- **Concurrent Compression**: Memory compression proceeds alongside tool execution
- **Async Operations**: Non-blocking I/O throughout the system

### Integration with External Tools

ReCentAutoma seamlessly integrates with external tools:

- **MCP Tools**: Can use MCP tools through tool set builders
- **Dynamic Tool Creation**: Tools are created on-demand based on LLM decisions
- **Tool Result Integration**: Tool results are automatically added to memory

## Design Patterns

### Dynamic Topology

ReCentAutoma demonstrates advanced use of Bridgic's dynamic topology:

- **Runtime Worker Creation**: Workers are created based on runtime decisions
- **Dynamic Dependencies**: Dependency graphs are constructed dynamically
- **Automatic Cleanup**: Temporary workers are automatically removed

### Memory-Aware Agent Design

The episodic memory structure enables:

- **Long-Running Tasks**: Support for tasks that span many interactions
- **Context Efficiency**: Sub-linear memory growth
- **Goal Persistence**: Goals are never lost through compression

### Structured Output for Control Flow

LLM structured output is used for critical control decisions:

- **Goal Achievement Evaluation**: Structured assessment of progress
- **Tool Selection**: Structured tool choice with parameters
- **Type Safety**: Strong typing ensures correctness

## Best Practices

### Configuration Tuning

- Adjust compression thresholds based on your LLM's context window
- Configure appropriate stop conditions for your use case
- Use dedicated LLMs for compression if available

### Goal Definition

- Define clear, measurable goals
- Provide guidance to help the agent understand the goal
- Consider breaking complex goals into sub-goals

### Tool Selection

- Provide appropriate tools for the task
- Use tool set builders for exclusive resources
- Monitor tool usage patterns to optimize selection

### Memory Management

- Monitor memory growth and compression frequency
- Adjust thresholds if compression is too frequent or infrequent
- Review compression summaries to ensure quality
