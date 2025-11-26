# 核心机制

Bridgic 让您通过将工作流程分解为称为 worker 的模块化构建块来构建智能系统。每个 worker 代表一个特定的任务或行为，使得组织复杂的流程变得简单。

Bridgic 引入了清晰的抽象，用于构建流程、在执行单元之间传递数据、处理并发以及启用动态控制逻辑（例如条件分支和路由）。这种设计使用户能够构建从简单工作流程到复杂智能系统的高效可扩展系统。

主要功能包括：

1. [并发模式](../core_mechanism/concurrency_mode): 系统地和方便地组织您的并发执行单元。
2. [参数解析](../core_mechanism/parameter_resolving.ipynb): 探索三种数据传递方式和两种在执行单元之间调度数据的方式，包括参数绑定的 Arguments Mapping、Arguments Injection 和 Inputs Propagation；包括 Input Dispatching 和 Result Dispatching。
3. [动态路由](../core_mechanism/dynamic_routing): 通过易于使用的 `ferry_to()` API 启用条件分支和决策。
4. [动态拓扑](../core_mechanism/dynamic_topology): 在运行时更改底层图拓扑，以支持高度自主的 AI 应用程序。
5. [人在回路](../core_mechanism/human_in_the_loop): 在工作流/智能体执行期间启用人机交互或外部输入。
6. [Worker 回调](../core_mechanism/worker_callback): 提供非侵入性和多范围的回调机制，允许您在不同级别观察、定制或扩展执行，且对核心逻辑的影响最小。
7. [模块化](../core_mechanism/modularity): 通过将一个 Automata 嵌入到另一个中来重用和组合 Automata，以构建可扩展的智能系统。
8. [模型集成](../model_integration/llm_integration): 将模型纳入构建具有更自主能力的程序。

这个架构基础使 Bridgic 成为构建强大、自适应且易于推理的智能系统的平台，使您能够将逻辑的精确性与 AI 的创造潜力相结合。