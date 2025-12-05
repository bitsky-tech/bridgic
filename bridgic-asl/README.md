# ASL (Agent Structure Language)


**ASL (Agent Structure Language)** — an internal DSL(Domain Specific Language) in Python for building agents based on `bridgic-core`. It provides a declarative, intuitive syntax for defining complex workflow graphs and concurrent automata, making it easier to build and maintain agent-based applications.

## Features

- **Declarative Syntax**: Define workflows using intuitive Python syntax with context managers
- **Rich Operators**: Use `+`, `~`, `>>`, and `&` operators to express complex dependencies
- **Fragment Support**: Group workers into reusable logical fragments
- **Nested Graphs**: Support for hierarchical graph structures
- **Component Reuse**: Reuse existing automata as sub-graphs
- **Dynamic Workers**: Lambda-based dynamic worker generation for concurrent automata

## Installation

```shell
pip install bridgic-asl
```

## Quick Start

### Basic Example

```python
from bridgic.asl import ASLAutoma, graph

def add_one(x: int):
    return x + 1

def add_two(x: int):
    return x + 2

class MyGraph(ASLAutoma):
    with graph as g:
        a = add_one
        b = add_two
        
        +a >> ~b  # a is the start worker, b depends on a and is the output

# Run the graph
graph = MyGraph()
result = await graph.arun(x=1)  # result: 3 (1+1+2)
```

### Operators Explained

- `+` marks a worker as a **start worker** (entry point)
- `~` marks a worker as an **output worker** (exit point)
- `>>` defines a **dependency** (left must complete before right executes)
- `&` **groups** multiple workers together

### Fragment Example

```python
from bridgic.asl import ASLAutoma, graph

class MyGraph(ASLAutoma):
    with graph as g:
        a = worker1
        b = worker11
        c = worker12
        d = worker5
        e = worker6
        
        # Create fragments
        fragment_1 = +(a & b & c)  # Fragment with start workers
        fragment_2 = (d & e)       # Regular fragment
        
        # Chain fragments together
        fragment_1 >> fragment_2 >> ~merge
```

### Nested Graphs

```python
class MyGraph(ASLAutoma):
    with graph as g:
        a = SubGraph2()
        b = worker2
        
        +a >> b
        
        # Nested graph
        with graph as sub_graph:
            x = worker1
            y = worker2
            +x >> ~y
```

### Configuration and Parameter Injection

```python
from bridgic.asl import ASLAutoma, graph, Settings, Data
from bridgic.core.automa.args import ArgsMappingRule, From

class MyGraph(ASLAutoma):
    with graph as g:
        a = worker1
        b = worker2 *Settings(args_mapping_rule=ArgsMappingRule.AS_IS)
        c = merge *Data(y=From("a"))  # Inject parameter from worker 'a'
        
        +a >> b >> ~c
```

### Concurrent Automa with Dynamic Workers

```python
from bridgic.asl import ASLAutoma, concurrent, ASLField, Settings
from bridgic.core.automa.args import ResultDispatchingRule

class MyConcurrent(ASLAutoma):
    with concurrent(subtasks=ASLField(list, dispatching_rule=ResultDispatchingRule.IN_ORDER)) as sub_concurrent:
        dynamic_logic = lambda subtasks: (
            tasks_done *Settings(key=f"tasks_done_{i}")
            for i, subtask in enumerate(subtasks)
        )
```

## Core Concepts

### ASLAutoma

The main class that extends `GraphAutoma` from `bridgic-core`. Define your agent by inheriting from `ASLAutoma` and using the `graph` or `concurrent` context managers.

### Graph vs Concurrent

- **`graph`**: Creates a `GraphAutoma` for workflow execution
- **`concurrent`**: Creates a `ConcurrentAutoma` for parallel task execution

### Settings

Configure worker behavior:

```python
worker *Settings(
    key="custom_key",
    args_mapping_rule=ArgsMappingRule.AS_IS,
    result_dispatching_rule=ResultDispatchingRule.IN_ORDER
)
```

### Data

Inject default parameter values and modify function signatures:

```python
worker *Data(param1=value1, param2=From("other_worker"))
```

## Requirements

- Python >= 3.9
- bridgic-core >= 0.1.5.dev1