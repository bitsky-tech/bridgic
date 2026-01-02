自主代理，由现代大型语言模型驱动，代表了人工智能行业的一项重大进展。然而，当这些“AI代理”在现实世界的生产环境中部署时，许多工作流程仍然是确定性的，并且并不完全“具有代理性”。因此，社区在很大程度上达成了一致，使用“代理系统”一词来指代从确定性工作流程到完全自主代理的整个AI系统范围。

Bridgic 提供了一种一致且统一的方法来开发所有类型的代理系统——无论它们的代理性如何。这正是选择“Bridgic”这个名称的原因——**它连接了逻辑与魔法！** 接下来的部分将介绍和解释构成 Bridgic 基础的基本概念。

## Worker 和 Automa

Bridgic 有两个核心概念：

* **Worker**：Bridgic 中的基本执行单元。
* **Automa**：管理和协调一组 workers 的实体。一个 automa 本身也是一个 worker，这使得 automa 实例可以相互嵌套。

worker 是实际执行任务的实体。在现实世界的系统中，worker 可以表示精确的执行逻辑，例如一个函数或一个 API 调用，或者它可以是高度自主的东西，比如一个代理。换句话说，worker 可以是任何能够执行操作的实体，无论其自主性水平如何。

在 Bridgic 中，[`Worker`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker) 类在 Python 中定义如下：

```python
class Worker:
    async def arun(self, *args, **kwargs) -> Any:
        ...
```

[`Worker`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker) 类的 [`arun`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker.arun) 方法被调用以执行任务。您可以将任何所需的参数传递给 `arun`，它将返回作为任务结果的值。

!!! hint "提示"
    实际上，除了 `arun` 方法，`Worker` 还有一个 [`run`](../../reference/bridgic-core/bridgic/core/automa/worker/#bridgic.core.automa.worker.Worker.run) 方法。这与 Bridgic 的 [并发模式](../../../../tutorials/items/core_mechanism/concurrency_mode/) 相关。有关更多详细信息，请参阅相关部分。

除了 worker，“automa” 是 Bridgic 中的另一个核心概念。一个 automa 作为一组 workers 的容器。它不是自己执行任务，而是调度和协调其包含的 workers，根据预定义或动态执行流程运行它们，以完成整体任务。

<div style="text-align: center;">
<img src="../imgs/automa_workers_basic.png" alt="An automa contains several workers" width="512">
</div>

在 Bridgic 中，[`Automa`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.Automa) 类在 Python 中定义如下：```python
class Automa(Worker):
        ...
```

请注意，`Automa` 类继承自 `Worker`。这意味着每个 automa 也是一个 worker，并且可以无缝嵌套在另一个 automa 中。这种设计抽象允许通过逐层组合 automa 实现强大的模块化编程。我们将在后面详细阐述这一点。

## GraphAutoma

[`GraphAutoma`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma) 是 automa 概念的具体实现，其中 workers 被组织成一个有向图 (DG)，workers 作为节点，它们之间的关系作为边。

### 预定义依赖

<div style="text-align: center;">
<img src="../imgs/automa_dg.png" alt="An automa directed graph" width="512">
</div>

如上图所示，workers 的执行顺序主要取决于两个因素：

* 一个或多个起始 workers 的指定。
* workers 之间建立的预定义依赖关系。它们在图中用实线箭头表示。

执行流程从起始 worker（即 `worker_1`）开始，随后并发执行 `worker_2` 和 `worker_3`，最后执行 `worker_4`。

值得注意的是，`worker_4` 仅在 `worker_2` 和 `worker_3` 完成执行后才会被触发。换句话说，在 `GraphAutoma` 中，如果一个 worker 有多个前驱 workers，它采用“与”执行逻辑——这意味着该 worker 只有在所有直接前驱完成后才会开始执行。

上述执行流程可以通过以下代码实现：

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

这段代码使用 [`@worker` 装饰器](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.worker._worker_decorator.worker) 将一个普通的 Python 方法转换为 worker，并使用 `dependencies` 参数定义这些 workers 之间的部分顺序。

### 动态路由

为了支持自主代理的开发，除了上述提到的预定义依赖关系，Bridgic 还提供了一个易于使用的 [`ferry_to()`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.ferry_to) API 来实现动态路由。

<div style="text-align: center;">
<img src="../imgs/automa_dg_ferry_to.png" alt="Dynamic Routing based ferry_to" width="512">
</div>如上图所示，虚线箭头表示潜在的执行路径。在 `worker_1` 完成执行后，`GraphAutoma` 的调度器在运行时根据参数或上下文决定是继续执行 `worker_2`、`worker_3` 还是两者都执行。

以下是代码：

```python
from bridgic.core.automa import GraphAutoma, worker

class MyFlow(GraphAutoma):
    @worker(is_start=True)
    async def worker_1(self, x):
        ...
        if x > 0:
            self.ferry_to("worker_2", y=1)
        else:
            self.ferry_to("worker_3", z=2)

    @worker()
    async def worker_2(self, y):
        ...

    @worker()
    async def worker_3(self, z):
        ...
```

通过 [`ferry_to()`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.ferry_to) 机制，您可以以自然、直观的方式实现动态路由——就像调用常规函数一样。

!!! hint "提示"
    `ferry_to` API 的名称源于在有向图中“渡送”不同“岛屿”的概念。

借助 `ferry_to` 提供的动态路由机制，您可以轻松地在工作流中实现循环逻辑。下图说明了其工作原理：

<div style="text-align: center;">
<img src="../imgs/automa_dg_loop.png" alt="循环演示" width="400">
</div>

代码：

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

在这段代码中，`worker_3` 根据不同的条件决定是运行 `worker_2` 还是 `worker_4`。

需要注意的是，当调用 `ferry_to("worker_2", x)` 时，`worker_2` 将在下一个事件循环迭代中立即执行，而无需等待其预定义的依赖 `worker_1` 完成。这种行为源于动态路由与预定义依赖之间的相互作用——这种方法使 Bridgic 能够无缝地将静态调度与灵活的动态控制流结合起来。

### API

`GraphAutoma` 提供两种类型的 API，用于管理图的拓扑：* **核心 API**: [`add_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_worker), [`add_func_as_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_func_as_worker), [`remove_worker`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.remove_worker), 和 [`add_dependency`](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.GraphAutoma.add_dependency).

* **声明式 API**: [`@worker` 装饰器](../../reference/bridgic-core/bridgic/core/automa/#bridgic.core.automa.worker._worker_decorator.worker).

有关更多代码示例，请参阅 [教程](../tutorials/items/quick_start/quick_start.ipynb) 部分。

## 动态有向图

`GraphAutoma` 实现了一个 **动态有向图** (DDG)，使用 asyncio 以异步和动态的方式协调其内部工作者的执行。

DDG 是一个其拓扑可以在运行时更改的有向图。其调度器将协调过程划分为几个动态步骤 (DS)，每个步骤在单个事件循环迭代中执行。在每个 DS 结束时，调度器根据预定义的依赖关系或动态 `ferry_to` 调用准备下一组要运行的工作者。由 `add_worker` 或 `remove_worker` 触发的任何拓扑更改将在下一个 DS 中生效。

<div style="text-align: center;">
<img src="../imgs/automa_dg.png" alt="An automa directed graph" width="512">
</div>

以上述图为例，整个执行被划分为三个动态步骤：

* DS 1: `worker_1` 被执行。
* DS 2: `worker_2` 和 `worker_3` 同时被执行。
* DS 3: `worker_4` 被执行。

<!-- multip start workers -->

## 模块化与嵌套

在 Bridgic 中，一个 automa 本身也是一个工作者，允许一个 automa 被添加到另一个 automa 中。这种设计通过层次嵌套重用组件，使得构建复杂的代理系统成为可能，引入了一种基于组件的模块化编程的新范式，适用于基于代理的开发。

有关模块化的更多代码示例，请参阅教程中的 "[模块化](../../tutorials/items/core_mechanism/modularity/)" 部分。