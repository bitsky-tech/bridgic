# LangWatch 集成

## 概述

[LangWatch](https://langwatch.ai/) 是一个为 LLM 应用设计的全面可观测性平台。`bridgic-traces-langwatch` 包使 LangWatch 能够无缝集成到您的基于 Bridgic 的代理工作流中。

此集成主要由 `LangWatchTraceCallback` 支持，这是一个 [WorkerCallback](../core_mechanism/worker_callback.ipynb) 实现，自动为工作者执行添加 LangWatch 跟踪，从而提供全面的可观测性，具体包括：

* **工作者执行跟踪**：将每个工作者的执行记录为 LangWatch 中的一个跨度，允许可视化开始/结束时间和持续时间。
* **工作者执行数据报告**：捕获输入、输出和其他必要信息，然后记录到 LangWatch 平台。
* **层次跟踪结构**：以反映 automa 层之间嵌套关系的层次组织执行跟踪，使得可以直观地看到顶层 automa 是如何由多个嵌套工作者的执行组成的。

## 先决条件

LangWatch 提供了一个托管版本的平台，或者您可以在本地运行该平台。

- 要使用托管版本，您需要 [创建一个 LangWatch 账户](https://langwatch.ai/) 并从仪表板获取您的 API Key。
- 要在本地运行 LangWatch，请参阅 [自托管指南](https://docs.langwatch.ai/self-hosting/overview) 以获取更多信息。

<div style="text-align: center;">
<img src="../../../imgs/langwatch-api-key.png" alt="LangWatch api key" width="auto">
</div>

## 在 Bridgic 中使用 LangWatch

### 第一步：安装包

```shell
# 安装 LangWatch 跟踪包
pip install bridgic-traces-langwatch
```

### 第二步：配置 LangWatch

使用环境变量

设置以下环境变量：

```bash
export LANGWATCH_API_KEY="your-api-key-here"
export LANGWATCH_ENDPOINT="https://app.langwatch.ai"  # 可选，默认为 https://app.langwatch.ai
```

### 第三步：注册回调

您可以通过一次辅助调用（推荐）全局启用 LangWatch 跟踪，或者在需要自定义行为时手动连接回调。Bridgic 还允许您通过 `RunningOptions` 将跟踪范围限制到单个 automa。

#### 方法 1：应用范围注册（辅助或手动）

选择适合您设置的代码片段——它们产生相同的效果。

=== "start_langwatch_trace"

    ```python
    from bridgic.traces.langwatch import start_langwatch_trace

    start_langwatch_trace(
        api_key=None,            # 默认为 LANGWATCH_API_KEY 环境变量
        endpoint_url=None,       # 默认为 LANGWATCH_ENDPOINT 或 https://app.langwatch.ai
        base_attributes=None     # 可选：应用于每个跟踪的共享属性
    )
    ```

=== "GlobalSetting"

    ```python
    from bridgic.core.automa.worker import WorkerCallbackBuilder
    from bridgic.core.config import GlobalSetting
    from bridgic.traces.langwatch import LangWatchTraceCallback
```GlobalSetting.set(callback_builders=[WorkerCallbackBuilder(
        LangWatchTraceCallback,
        init_kwargs={
            "api_key": None,
            "endpoint_url": None,
            "base_attributes": None
        }
    )])

```python
from bridgic.core.automa import GraphAutoma, worker

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
    # 在启动时调用 start_langwatch_trace(...) 或配置 GlobalSetting(...)
    from bridgic.traces.langwatch import start_langwatch_trace

    start_langwatch_trace(
        api_key=None,
        endpoint_url=None,
        base_attributes=None
    )

    automa = DataAnalysisAutoma()
    result = await automa.arun(topic="市场分析")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())
```

#### 方法 2：使用 `RunningOptions` 的每个 automa 范围

当仅需要特定 automa 进行 LangWatch 跟踪时，通过 `RunningOptions` 配置回调。

```python
from bridgic.core.automa import GraphAutoma, RunningOptions, worker
from bridgic.core.automa.worker import WorkerCallbackBuilder
from bridgic.traces.langwatch import LangWatchTraceCallback

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
```@worker(dependencies=["analyze_trends"], is_output=True)
    async def generate_report(self, analysis: dict) -> str:
        """生成最终报告。"""
        return f"报告：发现 {len(analysis['trends'])} 个趋势，置信度为 {analysis['confidence']}。"

async def automa_arun():
    builder = WorkerCallbackBuilder(LangWatchTraceCallback, init_kwargs={
        "api_key": None,
        "endpoint_url": None,
        "base_attributes": None
    })
    running_options = RunningOptions(callback_builders=[builder])
    automa = DataAnalysisAutoma(running_options=running_options)
    result = await automa.arun(topic="市场分析")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(automa_arun())

一旦您的 Bridgic 应用程序完成运行，跟踪信息将自动发送到 LangWatch。您可以在 LangWatch 仪表板中查看它们，以探索丰富的可视化洞察和您工作流程的详细跟踪信息。

<div style="text-align: center;">
<img src="../../../imgs/bridgic-integration-langwatch-demo.png" alt="bridgic 集成 langwatch 演示" width="auto">
</div>