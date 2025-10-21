**Bridgic** is an innovative programming framework designed to create agentic systems, ranging from deterministic workflows to autonomous agents. It introduces a new paradigm that simplifies the development of agentic systems.

> ✨ The name "**Bridgic**" is inspired by the idea of *"Bridging Logic and Magic"*. It means seamlessly uniting the precision of *logic* (deterministic execution flows) with the creativity of *magic* (highly autonomous AI).


## 🔗 Features

* **Orchestration**: Bridgic helps manage the execution flow of your AI applications by leveraging both predefined dependencies and dynamic routing.
* **Parameter Binding**: There are three ways to pass data among workers—Arguments Mapping, Arguments Injection, and Inputs Propagation—thereby eliminating the complexity of global state management.
* **Dynamic Routing**: Bridgic enables conditional branching and intelligent decision-making through an easy-to-use `ferry_to()` API that adapts to runtime dynamics.
* **Dynamic Topology**: The topology can be changed at runtime in Bridgic to support highly autonomous AI applications.
* **Modularity**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Human-in-the-Loop**: A Bridgic-style agentic system can request feedback from human whenever needed to dynamically adjust its execution logic.
* **Serialization**: Bridgic employs a scalable serialization and deserialization mechanism to achieve state persistence and recovery, enabling human-in-the-loop in long-running AI systems.
* **Systematic Integration**: A wide range of tools and LLMs can be seamlessly integrated into the Bridgic world, in a systematic way.
* **Customization**: What Bridgic provides is not a "black box" approach. You have full control over every aspect of your AI applications, such as prompts, context windows, the control flow, and more.


## 📦 Installation

Python 3.9 or higher version is required.

```bash
pip install bridgic
```


## 🚀 Development Paradigm

Bridgic provides several core mechanisms for building agentic systems.

Here are simple examples demonstrating each key feature:

### 1. Static Declaration

The `@worker` decorator defines workers with dependencies at the same time as the definition of the subclass of `GraphAutoma`. This declarative approach allows you to express complex workflows as a graph of interconnected tasks, where each worker represents a specific processing step and dependencies define the execution order automatically.

```python
from bridgic.core.automa import GraphAutoma, worker

class TextProcessor(GraphAutoma):
    @worker(is_start=True)
    async def load_text(self, text: str) -> str:
        return text.strip()

    @worker(dependencies=["load_text"])
    async def count_words(self, text: str) -> int:
        return len(text.split())

    @worker(dependencies=["count_words"], is_output=True)
    async def generate_summary(self, word_count: int) -> dict:
        return {"word_count": word_count}

async def main():
    """Run the text processor example."""
    processor = TextProcessor()

    sample_text = "Hello world. This is a sample text. It contains multiple sentences."
    result = await processor.arun(text=sample_text)

    print(f"Final result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
Final result: {'word_count': 11}
```

In this example, the text processing workflow demonstrates how static dependencies create a clear data flow: `load_text` → `count_words` → `generate_summary`. The framework automatically handles the execution order based on the declared dependencies, making it easy to reason about the workflow logic.

### 2. Dynamic Routing

Use `ferry_to()` for conditional routing based on runtime conditions. This enables intelligent decision-making where the system can adapt its behavior based on the content or context of incoming requests, creating truly autonomous agents that can choose different processing paths.

```python
from bridgic.core.automa import GraphAutoma, worker

class SmartRouter(GraphAutoma):
    @worker(is_start=True)
    async def analyze_request(self, request: str) -> str:
        print(f"Analyzing request: {request}")
        if "?" in request:
            print("Detected question - routing to Q&A handler")
            self.ferry_to("handle_question", question=request)
        else:
            print("Standard request - routing to general handler")
            self.ferry_to("handle_general", question=request)

    @worker()
    async def handle_question(self, question: str) -> str:
        print("❓ QUESTION: Processing question")
        response = f"ANSWER: Based on your question '{question}', here's what I think..."
        print(f"Response: {response}")
        return response

    @worker()
    async def handle_general(self, question: str) -> str:
        print("📝 GENERAL: Processing standard question")
        response = f"ACKNOWLEDGED: {question}"
        print(f"Response: {response}")
        return response

async def main():
    """Run the smart router example."""
    router = SmartRouter()
    test_requests = [
        "What is the weather like today?",
        "Create a poeom about love."
    ]
    for request in test_requests:
        print(f"\n--- Processing: {request} ---")
        await router.arun(request=request)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
--- Processing: What is the weather like today? ---
Analyzing request: What is the weather like today?
Detected question - routing to Q&A handler
❓ QUESTION: Processing question
Response: ANSWER: Based on your question 'What is the weather like today?', here's what I think...

--- Processing: Create a poeom about love. ---
Analyzing request: Create a poeom about love.
Standard request - routing to general handler
📝 GENERAL: Processing standard question
Response: ACKNOWLEDGED: Create a poeom about love.
```

The smart router example showcases how `ferry_to()` enables conditional execution paths. The system analyzes each request and dynamically chooses the appropriate handler, demonstrating how agents can make intelligent routing decisions based on the nature of incoming data.

### 3. Parameter Binding

Control how data flows between workers using different mapping rules. This flexible parameter binding mechanism allows you to precisely control how data is passed between workers, supporting various data transformation patterns and enabling complex data processing pipelines.

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.automa.args import ArgsMappingRule, From

class DataProcessor(GraphAutoma):
    @worker(is_start=True)
    async def generate_data(self, bound: int) -> list:
        return list(range(1, bound + 1))

    @worker(dependencies=["generate_data"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def calculate(self, data: list) -> dict:
        return {"sum": sum(data), "count": len(data)}

    @worker(dependencies=["calculate"], args_mapping_rule=ArgsMappingRule.UNPACK, is_output=True)
    async def output(self, bound: From(key="generate_data"), sum: int, count: int) -> str:
        return f"The bound is {bound}. The length of data is {count} and their sum is {sum}."

async def main():
    """Run the data processor example."""
    processor = DataProcessor()

    result = await processor.arun(bound=5)
    print(f"Final result: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
Final result: The bound is 5. The length of data is 5 and their sum is 15.
```

This parameter binding example demonstrates the power of flexible data flow control. The `AS_IS` rule passes data as-is, while `UNPACK` spreads list elements as individual arguments, and `From()` allows accessing data from any worker without direct dependencies. This enables sophisticated data processing patterns where workers can access information from multiple sources.


## 🤖 Building Complex Agentic System

By combining these features, you can build a Bridgic-style agentic system that can:

- **Execute clear workflows** through declarative dependencies
- **Adapt intelligently** to different situations according to runtime conditions
- **Process complex data** across multiple steps, supporting transformations and aggregations

Whether you're building simple automation scripts or complex autonomous agents, Bridgic provides the tools to express your logic clearly while maintaining the flexibility needed for intelligent, adaptive behavior.

More examples will be added in the near future. :)

## 📚 Documents

For more about development skills of Bridgic, see:

- [Tutorials](https://docs.bridgic.ai/latest/tutorials/)
- [Understanding](https://docs.bridgic.ai/latest/home/introduction/)


## 📄 License

This repository is licensed under the [MIT License](/LICENSE).


## 🤝 Contributing

For contribution guidelines and instructions, please see [CONTRIBUTING](/CONTRIBUTING.md).
