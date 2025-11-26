# Opik 集成

## 概述

[Comet Opik](https://www.comet.com/docs/opik/) 是一个为智能系统设计的全面可观测性平台。`bridgic-traces-opik` 包使 Opik 能够无缝集成到基于 Bridgic 的智能工作流中。

此集成主要由 `OpikTraceCallback` 支持，这是一个 [WorkerCallback](../core_mechanism/worker_callback.ipynb) 实现，自动为 Worker 执行添加 Opik 跟踪，从而提供全面的可观测性，具体包括：

* **Worker 执行跟踪**：将每个 Worker 的执行记录为 Opik 中的一个跨度，允许可视化开始/结束时间和持续时间。
* **Worker 执行数据报告**：捕获输入、输出和其他必要信息，然后记录到 Opik 平台。
* **层次化跟踪结构**：以反映 automa 层之间嵌套关系的层次结构组织执行跟踪，使得可以直观地看到顶层 automa 是如何由多个嵌套 Worker 的执行组成的。

## 先决条件

Comet 提供了 Opik 平台的托管版本，或者您可以在本地运行该平台。

- 要使用托管版本，您需要 [创建一个 Comet 账户](https://www.comet.com/signup) 并 [获取您的 API 密钥](https://www.comet.com/account-settings/apiKeys)。
- 要在本地运行 Opik 平台，请参见 [安装指南](https://www.comet.com/docs/opik/self-host/overview/) 以获取更多信息。

## 在 Bridgic 中使用 Opik

### 第一步：安装包

```shell
# 自动安装 Opik 包
pip install bridgic-traces-opik
```

### 第二步：配置 Opik

配置 Python SDK 的推荐方法是使用 opik configure 命令。这将提示您设置 API 密钥和 Opik 实例 URL（如适用），以确保正确的路由和身份验证。所有详细信息将保存到配置文件中。

=== "Opik Cloud"

    如果您使用的是平台的云版本，可以通过运行以下命令配置 SDK：

    ```python
    import opik

    opik.configure(use_local=False)
    ```

    您也可以通过命令行调用 [`configure`](https://www.comet.com/docs/opik/python-sdk-reference/cli.html) 来配置 SDK：

    ```bash
    opik configure
    ``` 
=== "自托管"

    如果您自托管该平台，可以通过运行以下命令配置 SDK：

    ```python
    import opik

    opik.configure(use_local=True)
    ```

    或者通过命令行：

    ```bash
    opik configure --use_local
    ```

`configure` 方法将提示您输入必要的信息并将其保存到配置文件 (`~/.opik.config`) 中。当使用命令行版本时，您可以使用 `-y` 或 `--yes` 标志自动批准任何确认提示：

```bash
opik configure --yes
```

### 第三步：注册回调
```您可以在最适合您应用程序的范围内注册 Opik 跟踪。`start_opik_trace` 是最快的路径（通过 `GlobalSetting` 配置全局跟踪的一行代码）。当您想要自定义相同的全局设置或仅针对特定的 automa 时，Bridgic 为这两种用例提供了直接的钩子。

#### 方法 1：全应用程序注册（助手或手动）

选择以下两个选项之一——它们产生完全相同的运行时行为：

=== "start_opik_trace"

    ```python
    from bridgic.traces.opik import start_opik_trace
    start_opik_trace(project_name="bridgic-integration-demo")
    ```

=== "GlobalSetting"

    ```python
    from bridgic.core.automa.worker import WorkerCallbackBuilder
    from bridgic.core.config import GlobalSetting
    from bridgic.traces.opik import OpikTraceCallback

    GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(
        OpikTraceCallback,
        init_kwargs={"project_name": "bridgic-integration-demo"}
    )])
    ```

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.traces.opik import start_opik_trace

class DataAnalysisAutoma(GraphAutoma):
    @worker(is_start=True)
    async def collect_data(self, topic: str) -> dict:
        """收集给定主题的数据。"""
        # 模拟数据收集
        return {
            "topic": topic,
            "data_points": ["point1", "point2", "point3"],
            "timestamp": "2024-01-01"
        }

    @worker(dependencies=["collect_data"])
    async def analyze_trends(self, data: dict) -> dict:
        """分析收集的数据中的趋势。"""
        # 模拟趋势分析
        return {
            "trends": ["trend1", "trend2"],
            "confidence": 0.85,
            "source_data": data
        }

    @worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """生成最终报告。"""
        return f"报告：发现 {len(analysis['trends'])} 个趋势，置信度为 {analysis['confidence']}."

async def automa_arun():
    # 在启动时调用 start_opik_trace(...) 或 GlobalSetting.set(...) 一次
    start_opik_trace(project_name="bridgic-integration-demo")
    automa = DataAnalysisAutoma()
    result = await automa.arun(topic="市场分析")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

#### 方法 2：使用 `RunningOptions` 的每个 automa 范围

当仅需要特定 automa 的跟踪时，通过 `RunningOptions` 配置回调。每个 automa 获取自己的回调实例，其他 automa 不受影响。

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.traces.opik import OpikTraceCallback
```class DataAnalysisAutoma(GraphAutoma):
    @worker(is_start=True)
    async def collect_data(self, topic: str) -> dict:
        """收集给定主题的数据。"""
        # 模拟数据收集
        return {
            "topic": topic,
            "data_points": ["point1", "point2", "point3"],
            "timestamp": "2024-01-01"
        }

    @worker(dependencies=["collect_data"])
    async def analyze_trends(self, data: dict) -> dict:
        """分析收集到的数据中的趋势。"""
        # 模拟趋势分析
        return {
            "trends": ["trend1", "trend2"],
            "confidence": 0.85,
            "source_data": data
        }

    @worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """生成最终报告。"""
        return f"报告: 发现 {len(analysis['trends'])} 个趋势，置信度为 {analysis['confidence']}。"

async def automa_arun():
    builder = WorkerCallbackBuilder(OpikTraceCallback, init_kwargs={"project_name": "bridgic-integration-demo"})
    running_options = RunningOptions(callback_builders=[builder])
    automa = DataAnalysisAutoma(running_options=running_options)
    result = await automa.arun(topic="市场分析")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())

一旦您的 Bridgic 应用程序完成运行，您的终端可能会显示以下消息：

```shell
$ python bridgic-demo/demo.py 
OPIK: 开始将跟踪记录到 "bridgic-integration-demo" 项目，地址为 http://localhost:5173/api/v1/session/redirect/projects/?trace_id=019a9709-e437-7b30-861e-76006b75e969&path=aHR0cDovL2xvY2FsaG9zdDo1MTczL2FwaS8=
报告: 发现 2 个趋势，置信度为 0.85。
```

您可以深入 Opik 应用程序，探索丰富的可视化洞察和详细的工作流程跟踪。

<div style="text-align: center;">
<img src="../../../imgs/bridgic-integration-opik-demo.png" alt="bridgic 集成 opik 演示" width="auto">
</div>