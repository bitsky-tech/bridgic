# Agent Tracer

一个框架无关的AI Agent和LLM应用追踪库。支持与任何Agent框架无缝集成，同时提供强大的多后端可观测性。

## 特性

- 🔌 **框架无关**: 适用于任何Python Agent框架（LangChain、CrewAI、AutoGPT、自定义框架等）
- 🎯 **多后端支持**: 支持LangSmith、LangFuse、LangWatch、Arize Phoenix、Opik和Traceloop
- 🚀 **易于集成**: 简单的API，无需框架特定知识
- 🔄 **异步支持**: 为现代Python应用构建的async/await支持
- 📊 **丰富上下文**: 捕获输入、输出、元数据、日志和错误
- 🛡️ **隐私优先**: 自动屏蔽敏感数据（API密钥、密码等）
- 🔗 **层次化追踪**: 支持嵌套追踪（工作流 → Agent → 工具）

## 安装

### 基础安装

```bash
pip install agent-tracer
```

### 安装特定追踪器

```bash
# 安装LangSmith支持
pip install agent-tracer[langsmith]

# 安装LangFuse支持
pip install agent-tracer[langfuse]

# 安装所有追踪器
pip install agent-tracer[all]
```

## 快速开始

### 基础用法

```python
from agent_tracer import TracingService
import asyncio

# 初始化追踪服务
tracer = TracingService()

async def main():
    # 开始追踪工作流
    await tracer.start_trace(
        trace_id="workflow-123",
        trace_name="客服机器人",
        project_name="我的项目"
    )
    
    # 追踪一个步骤
    async with tracer.trace_step(
        step_name="query_analysis",
        inputs={"query": "今天天气怎么样？"}
    ):
        # 你的Agent逻辑
        result = analyze_query("今天天气怎么样？")
        
        # 设置输出
        tracer.set_outputs(
            step_name="query_analysis",
            outputs={"intent": "weather_query", "entities": ["weather"]}
        )
    
    # 结束追踪
    await tracer.end_trace(outputs={"response": "今天天气晴朗"})

asyncio.run(main())
```

### 配置

通过环境变量配置追踪器：

```bash
# LangSmith
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_PROJECT="your-project"

# LangFuse
export LANGFUSE_SECRET_KEY="your-secret"
export LANGFUSE_PUBLIC_KEY="your-public"
export LANGFUSE_HOST="https://cloud.langfuse.com"

# LangWatch
export LANGWATCH_API_KEY="your-key"

# Arize Phoenix
export ARIZE_API_KEY="your-key"
export ARIZE_SPACE_ID="your-space-id"

# Opik
export OPIK_API_KEY="your-key"
export OPIK_WORKSPACE="your-workspace"

# Traceloop
export TRACELOOP_API_KEY="your-key"
```

## 框架集成示例

### LangChain集成

```python
from agent_tracer import TracingService
from langchain.agents import AgentExecutor

tracer = TracingService()

async def run_langchain_agent(query: str):
    await tracer.start_trace(
        trace_id="lc-agent-run",
        trace_name="LangChain Agent",
        project_name="我的项目"
    )
    
    # 获取LangChain回调
    callbacks = tracer.get_langchain_callbacks()
    
    # 使用回调运行Agent
    result = agent_executor.invoke(
        {"input": query},
        config={"callbacks": callbacks}
    )
    
    await tracer.end_trace(outputs={"result": result})
    return result
```

### 自定义框架集成

```python
from agent_tracer import TracingService

class MyCustomAgent:
    def __init__(self):
        self.tracer = TracingService()
    
    async def run(self, task: str):
        # 开始工作流追踪
        await self.tracer.start_trace(
            trace_id="custom-agent",
            trace_name="自定义Agent工作流",
            project_name="自定义框架"
        )
        
        try:
            # 追踪规划阶段
            async with self.tracer.trace_step(
                step_name="planning",
                inputs={"task": task}
            ):
                plan = await self.plan(task)
                self.tracer.set_outputs("planning", {"plan": plan})
            
            # 追踪执行阶段
            async with self.tracer.trace_step(
                step_name="execution",
                inputs={"plan": plan}
            ):
                result = await self.execute(plan)
                self.tracer.set_outputs("execution", {"result": result})
            
            await self.tracer.end_trace(outputs={"final_result": result})
            return result
            
        except Exception as e:
            await self.tracer.end_trace(error=e)
            raise
```

## API参考

### TracingService

#### 方法

- `start_trace(trace_id, trace_name, project_name, user_id=None, session_id=None)`: 开始新的追踪
- `trace_step(step_name, inputs, metadata=None)`: 追踪步骤/组件的上下文管理器
- `set_outputs(step_name, outputs, metadata=None)`: 设置当前步骤的输出
- `add_log(step_name, log)`: 添加日志条目到当前步骤
- `end_trace(outputs=None, error=None)`: 结束当前追踪
- `get_langchain_callbacks()`: 获取LangChain兼容的回调

## 高级用法

### 嵌套追踪

```python
async def complex_workflow():
    await tracer.start_trace(
        trace_id="main-workflow",
        trace_name="多Agent系统"
    )
    
    # 父任务
    async with tracer.trace_step("coordinator", {"task": "分析数据"}):
        
        # 子任务1
        async with tracer.trace_step("data_fetcher", {"source": "db"}):
            data = fetch_data()
            tracer.set_outputs("data_fetcher", {"data": data})
        
        # 子任务2
        async with tracer.trace_step("analyzer", {"data": data}):
            analysis = analyze(data)
            tracer.set_outputs("analyzer", {"analysis": analysis})
        
        tracer.set_outputs("coordinator", {"complete": True})
    
    await tracer.end_trace(outputs={"status": "success"})
```

### 添加日志

```python
async with tracer.trace_step("processing", {"input": data}):
    tracer.add_log("processing", {
        "name": "debug_info",
        "message": "处理开始",
        "type": "info"
    })
    
    result = process(data)
    
    tracer.add_log("processing", {
        "name": "performance",
        "message": f"处理耗时 {elapsed}ms",
        "type": "metric"
    })
    
    tracer.set_outputs("processing", {"result": result})
```

## 支持的追踪器

| 追踪器 | 状态 | 功能 |
|--------|------|------|
| LangSmith | ✅ 完全支持 | LangChain回调，嵌套追踪 |
| LangFuse | ✅ 完全支持 | 用户/会话追踪，层次化span |
| LangWatch | ✅ 完全支持 | 线程追踪，组件追踪 |
| Arize Phoenix | ✅ 完全支持 | OpenTelemetry，会话追踪 |
| Opik | ✅ 完全支持 | 线程/用户追踪，元数据 |
| Traceloop | ✅ 完全支持 | OpenTelemetry，自定义属性 |

## 设计原则

1. **框架无关**: 不依赖特定的Agent框架
2. **最小依赖**: 核心库依赖最少；追踪器特定依赖可选
3. **类型安全**: 完整的类型提示，更好的IDE支持
4. **异步优先**: 为现代异步Python应用构建
5. **隐私聚焦**: 自动敏感数据屏蔽
6. **可扩展**: 易于添加新的追踪器后端

## 架构

```
agent-tracer/
├── src/
│   └── agent_tracer/
│       ├── __init__.py          # 公共API
│       ├── base.py              # BaseTracer抽象类
│       ├── schema.py            # 数据模式
│       ├── service.py           # TracingService主API
│       ├── utils.py             # 工具函数
│       └── tracers/             # 追踪器实现
│           ├── langsmith.py
│           ├── langfuse.py
│           ├── langwatch.py
│           ├── arize_phoenix.py
│           ├── opik.py
│           └── traceloop.py
├── tests/                       # 测试
├── examples/                    # 示例
└── docs/                        # 文档
```

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

## 许可证

MIT许可证 - 详见 [LICENSE](LICENSE)

## 致谢

最初从 [Langflow](https://github.com/logspace-ai/langflow) 提取并改编为框架无关版本。

## 更多信息

- 📖 完整文档: [README.md](README.md)
- 🚀 快速开始: [QUICKSTART.md](QUICKSTART.md)
- 🏗️ 架构文档: [ARCHITECTURE.md](ARCHITECTURE.md)
- 💡 示例代码: [examples/](examples/)

