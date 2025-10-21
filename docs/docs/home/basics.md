Autonomous agents, powered by modern large language models, represent a major advance in the AI industry. However, when these "AI agents" are deployed in real-world production environments, many workflows remain deterministic and are not all that "agentic". Therefore, the community has largely reached a consensus to use the term "agentic system" to refer to the entire spectrum of AI systems, from deterministic workflows to fully autonomous agents.

Bridgic provides a consistent and unified approach for developing all types of agentic systems—no matter how agentic they are. This is precisely why the name "Bridgic" was chosen—**it bridges logic and magic!** The following sections will introduce and explain the fundamental concepts that form the foundation of Bridgic.

## Worker and Automa

Bridgic has two core concepts:

* **Worker**: the basic execution unit in Bridgic.
* **Automa**: an entity that manages and orchestrates a group of workers. An automa itself is also a worker, which enables the nesting of automa instances within each other.

A worker is an entity that actually performs tasks. In real-world systems, a worker can represent a precise execution logic, such as a function or an API call, or it can be something highly autonomous, like an agent. In other words, a worker can be any entity capable of carrying out actions, regardless of its level of autonomy.

In Bridgic, the `Worker` class is defined in Python as follows:

```python
class Worker:
    async def arun(self, *args, **kwargs) -> Any:
        ...
```

The `arun` method of the `Worker` class is called to execute a task. You can pass any required arguments to `arun`, and it will return a value of any type as the result of the task.

!!! hint "Hint"
    In fact, in addition to the `arun` method, a `Worker` also has a `run` method. This relates to Bridgic's concurrency mode. Please refer to the relevant sections for more details.

Besides worker, "automa" is another core concept in Bridgic. An automa acts as a container for a group of workers. Rather than performing tasks by itself, an automa schedules and orchestrates the workers it contains, executing them according to a predefined or dynamic control flow in order to accomplish the overall task.

<br>
<div style="text-align: center;">
<img src="/home/imgs/automa_workers_basic.png" alt="An automa contains several workers" width="512">
</div>
<br>

In Bridgic, the `Automa` class is defined in Python as follows:

```python
class Automa(Worker):
        ...
```

Note that the `Automa` class inherits from `Worker`. This means that every automa is also a worker and can be seamlessly nested within another automa. This design abstraction allows for powerful, modular programming by enabling automa to be composed layer by layer. We will elaborate on this later.

## GraphAutoma

`GraphAutoma` is a concrete implementation of the automa concept. In this implementation, workers are organized into a directed graph (DG), in which workers are nodes and 

Two types of relationships exist between workers:
* Predefined dependencies.
* Dynamic 




## How to use/call/invoke in Python?

worker 表现形式 function？


## Dynamic Directed Graph

control flow
DS


## Modulirity and Netsting

itself a 


