Autonomous agents, powered by modern large language models, represent a major advance in the AI industry. However, when these "AI agents" are deployed in real-world production environments, many workflows remain deterministic and are not all that "agentic". Therefore, the community has largely reached a consensus to use the term "agentic system" to refer to the entire spectrum of AI systems, from deterministic workflows to fully autonomous agents.

Bridgic provides a consistent and unified approach for developing all types of agentic systems—no matter how agentic they are. This is precisely why the name "Bridgic" was chosen—**it bridges logic and magic!** The following sections will introduce and explain the fundamental concepts that form the foundation of Bridgic.

## Worker and Automa

Bridgic has two core concepts:

* **Worker**: the basic execution unit in Bridgic.
* **Automa**: an entity that manages and orchestrates a group of workers. An automa itself is also a worker, which enables the nesting of automa instances within each other.

A worker is an entity that actually performs tasks. In real-world systems, a worker can represent a precise execution logic, such as a function or an API call, or it can be something highly autonomous, like an agent. In other words, a worker can be any entity capable of carrying out actions, regardless of its level of autonomy.

In Bridgic, the [`Worker`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker) class is defined in Python as follows:

```python
class Worker:
    async def arun(self, *args, **kwargs) -> Any:
        ...
```

The [`arun`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker.arun) method of the [`Worker`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker) class is called to execute a task. You can pass any required arguments to `arun`, and it will return a value as the result of the task.

!!! hint "Tips"
    In fact, in addition to the `arun` method, a `Worker` also has a [`run`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker.run) method. This relates to Bridgic's [concurrency mode](../../tutorials/items/core_mechanism/concurrency_mode/). Please refer to the relevant sections for more details.

Besides worker, "automa" is another core concept in Bridgic. An automa acts as a container for a group of workers. Instead of performing tasks by itself, an automa schedules and orchestrates the workers it contains, running them according to a predefined or dynamic execution flow in order to accomplish the overall task.

<div style="text-align: center;">
<img src="../imgs/automa_workers_basic.png" alt="An automa contains several workers" width="512">
</div>

In Bridgic, the [`Automa`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa) class is defined in Python as follows:

```python
class Automa(Worker):
        ...
```

Note that the `Automa` class inherits from `Worker`. This means that every automa is also a worker and can be seamlessly nested within another automa. This design abstraction allows for powerful, modular programming by enabling automa to be composed layer by layer. We will elaborate on this later.

## GraphAutoma

[`GraphAutoma`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma) is a concrete implementation of the automa concept, where workers are organized into a directed graph (DG) with workers as nodes and their relationships as edges.

### Predefined Dependencies

<div style="text-align: center;">
<img src="../imgs/automa_dg.png" alt="An automa directed graph" width="512">
</div>

As shown in the diagram above, the execution order of workers primarily depends on two factors:

* The designation of one or more start workers.
* The predefined dependencies established between workers. They are represented by the solid arrows in the diagram.

The execution flow starts with the start worker (i.e., `worker_1`), followed by the concurrent execution of `worker_2` and `worker_3`, and finally `worker_4` is executed.

Notably, `worker_4` will be triggered only after both `worker_2` and `worker_3` have completed their execution. In other words, within a `GraphAutoma`, if a worker has multiple predecessor workers, it adopts an "AND" execution logic—meaning that the worker will not start until all its direct predecessors have finished.

The execution flow shown above can be implemented with the following code:

```python
from bridgic.core.automa import GraphAutoma, worker

class MyFlow(GraphAutoma):
    @worker(is_start=True)
    async def worker_1(self):
        ...

    @worker(dependencies=["worker_1"])
    async def worker_2(self):
        ...

    @worker(dependencies=["worker_1"])
    async def worker_3(self):
        ...

    @worker(dependencies=["worker_2", "worker_3"])
    async def worker_4(self):
        ...
```

This code uses the [`@worker` decorator](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.worker._worker_decorator.worker) to transform a regular Python method into a worker, and uses the `dependencies` parameter to define the partial order among these workers.

### Dynamic Routing

To support the development of autonomous agents, in addition to the predefined dependencies mentioned above, Bridgic also provides an easy-to-use [`ferry_to()`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.ferry_to) API for implementing dynamic routing.

<div style="text-align: center;">
<img src="../imgs/automa_dg_ferry_to.png" alt="Dynamic Routing based ferry_to" width="512">
</div>

As illustrated in the diagram above, the dashed arrows represent potential execution paths. After `worker_1` finishes execution, the orchestrator of `GraphAutoma` determines at runtime—based on parameters or context—whether to proceed with `worker_2`, `worker_3`, or both.

Here is the code:

```python
from bridgic.core.automa import GraphAutoma, worker

class MyFlow(GraphAutoma):
    @worker(is_start=True)
    async def worker_1(self, x):
        ...
        if x > 0:
            self.ferry_to("worker_2", y=1)
        else
            self.ferry_to("worker_3", z=2)

    @worker()
    async def worker_2(self, y):
        ...

    @worker()
    async def worker_3(self, z):
        ...
```

With the [`ferry_to()`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.ferry_to) mechanism, you can implement dynamic routing in a natural, intuitive way—just like calling a regular function.

!!! hint "Tips"
    The name of the `ferry_to` API comes from the idea of "ferrying" between various "islands" in a directed graph.

With the help of the dynamic routing mechanism provided by `ferry_to`, you can easily implement looping logic in your workflow. The diagram below illustrates how this works:

<div style="text-align: center;">
<img src="../imgs/automa_dg_loop.png" alt="Looping demonstration" width="400">
</div>

Code:

```python
from bridgic.core.automa import GraphAutoma, worker

class MyFlow(GraphAutoma):
    @worker(is_start=True)
    async def worker_1(self, x):
        return x

    @worker(dependencies=["worker_1"])
    async def worker_2(self, x):
        return x + 1

    @worker(dependencies=["worker_2"])
    async def worker_3(self, x):
        if x < 0:
            self.ferry_to("worker_2", x)
        else:
            self.ferry_to("worker_4", x)

    @worker(is_output=True)
    async def worker_4(self, x):
        return x

```

In this code, `worker_3` decides whether to run `worker_2` or `worker_4` next based on different conditions.

It's important to note that when `ferry_to("worker_2", x)` is called, `worker_2` will execute immediately in the next event loop iteration, without waiting for its predefined dependency `worker_1` to complete. This behavior arises from the interplay between dynamic routing and predefined dependencies—an approach that allows Bridgic to seamlessly combine static orchestration with flexible, dynamic control flow.

### API

`GraphAutoma` provides two types of APIs that are used to manage the graph topology:

* **The core API**: [`add_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_worker), [`add_func_as_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_func_as_worker), [`remove_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.remove_worker), and [`add_dependency`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_dependency).

* **The declarative API**: [`@worker` decorator](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.worker._worker_decorator.worker).

For more code examples, please refer to the [Tutorials](../../tutorials/items/quick_start/quick_start.ipynb) section.

## Dynamic Directed Graph

`GraphAutoma` implements a **Dynamic Directed Graph** (DDG) that orchestrates the execution of its internal workers in an asynchronous and dynamic manner using asyncio.

A DDG is a directed graph whose topology can be changed at runtime. Its scheduler divides the orchestration process into several dynamic steps (DS), each executed in a single event loop iteration. At the end of each DS, the scheduler prepares the next set of workers to run based on the predefined dependencies or the dynamic `ferry_to` calls. Any topology changes triggered by `add_worker` or `remove_worker` takes effect in the next DS.

<div style="text-align: center;">
<img src="../imgs/automa_dg.png" alt="An automa directed graph" width="512">
</div>

Taking the above diagram as an example, the entire execution is divided into three dynamic steps:

* DS 1: `worker_1` is executed.
* DS 2: both `worker_2` and `worker_3` are executed.
* DS 3: `worker_4` is executed.

<!-- multip start workers -->

## Modulirity and Netsting

In Bridgic, an automa itself is also a worker, allowing one automa to be added into another. This design enables the construction of complex agentic systems by reusing components through hierarchical nesting, introducing a new paradigm of modular and component-based programming in agent-based development.

For more code examples on modularity, please refer to the "[Modularity](../../tutorials/items/core_mechanism/modularity/)" section in the Tutorials.

