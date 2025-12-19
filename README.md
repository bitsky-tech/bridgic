**Bridgic** is an agentic programming framework built around a novel **dynamic topology** orchestration model and a **component-oriented paradigm** that is realized through **ASL (Agent Structure Language)**—a powerful declarative DSL for composing, reusing, and nesting agentic structures. Together, these elements make it possible to develop the entire spectrum of agentic systems, ranging from deterministic workflows to autonomous agents.

> ✨ The name "**Bridgic**" is inspired by the idea of *"Bridging Logic and Magic"*. It means seamlessly uniting the precision of *logic* (deterministic execution flows) with the creativity of *magic* (highly autonomous AI).


## 🔗 Features

* **Orchestration**: Bridgic introduces a novel orchestration model based on DDG (Dynamic Directed Graph).
* **Dynamic Routing**: Bridgic enables conditional branching and dynamic orchestration through an easy-to-use `ferry_to()` API.
* **Dynamic Topology**: The DDG-based orchestration topology can be changed at runtime in Bridgic to support highly autonomous AI applications.
* **ASL**: ASL (Agent Structure Language) is a powerful declarative DSL that embodies a component-oriented paradigm and is even capable of supporting dynamic topologies.
* **Modularity & Componentization**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Parameter Resolving**: Two mechanisms are designed to pass data among workers/automas—thereby eliminating the complexity of global state management whenever necessary.
* **Human-in-the-Loop**: A Bridgic-style agentic system can request feedback from human whenever needed to dynamically adjust its execution logic.
* **Serialization**: Bridgic employs a scalable serialization and deserialization mechanism to achieve state persistence and recovery, enabling human-in-the-loop in long-running AI systems.
* **Systematic Integration**: A wide range of tools, LLMs and tracing functionalities can be seamlessly integrated into the Bridgic world, in a systematic way.
* **Customization**: What Bridgic provides is not a "black box" approach. You have full control over every aspect of your AI applications, such as prompts, context windows, the control flow, and more.


## 📦 Installation

Python 3.9 or higher version is required.

```bash
pip install -U bridgic
```

To run the following examples, the `bridgic-llms-openai` module also needs to be installed.

```bash
pip install -U bridgic-llms-openai
```

## 🚀 Code Examples

Here are simple examples demonstrating each key feature.

### 0. LLM Setup

First of all, create a LLM instance for later use.

```python
import os
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration
from bridgic.core.model.types import Message

# Get the API key and model name from environment variables.
_api_key = os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("OPENAI_MODEL_NAME")

llm = OpenAILlm(
    api_key=_api_key,
    timeout=5,
    configuration=OpenAIConfiguration(model=_model_name),
)
```

### 1. ASL (Agent Structure Language)

Let's start with a simple workflow example that demonstrates the use of [ASL](https://docs.bridgic.ai/latest/tutorials/items/asl/quick_start/): a text generation agent that breaks down a user query into multiple sub-queries and generates answers for each one.

Prepare two Python functions that respectively execute each step:

```python
from typing import List, Dict

# Break down the query into a list of sub-queries.
async def break_down_query(user_input: str) -> List[str]:
    llm_response = await llm.achat(
        messages=[
            Message.from_text(text="Break down the query into multiple sub-queries and only return the sub-queries", role="system"),
            Message.from_text(text=user_input, role="user"),
        ]
    )
    return [item.strip() for item in llm_response.message.content.split("\n") if item.strip()]

# Generate answers for each sub-query.
async def query_answer(queries: List[str]) -> Dict[str, str]:
    answers = []
    for query in queries:
        response = await llm.achat(
            messages=[
                Message.from_text(text="Answer the given query briefly", role="system"),
                Message.from_text(text=query, role="user"),
            ]
        )
        answers.append(response.message.content)
    
    res = {
        query: answer
        for query, answer in zip(queries, answers)
    }
    return res
```

Then, use ASL to orchestrate this workflow:

```python
from bridgic.asl import ASLAutoma, graph

class SplitSolveAgent(ASLAutoma):
    with graph as g:
        a = break_down_query
        b = query_answer

        +a >> ~b
```

**Key points:**

- **`with graph as g:`** - Opens a graph context.
- **`a = break_down_query`** - Declares a worker named `a` that corresponds to the `break_down_query` function.
- **`b = query_answer`** - Declares a worker named `b` that corresponds to the `query_answer` function.
- **`a >> b`** - Defines a dependency: `b` depends on `a`.
- **`+a`** - Marks `a` as a start worker.
- **`~b`** - Marks `b` as an output worker.

Create an instance of `SplitSolveAgent` and run it:

```python
async def main():
    text_generation_agent = SplitSolveAgent()
    query = "When and where was Einstein born?"
    sub_qas = await text_generation_agent.arun(query)
    print(sub_qas)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
{'1. What is the date of birth of Albert Einstein?': 'Albert Einstein was born on March 14, 1879.', '2. In which city was Albert Einstein born?': 'Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire.', '3. In which country was Albert Einstein born?': 'Albert Einstein was born in Germany.'}
```

Suppose we are going to develop a chatbot that merges these individual answers into a unified response. `SplitSolveAgent` can be reused in ASL:

```python
async def merge_answers(qa_pairs: Dict[str, str], user_input: str) -> str:
    answers = "\n".join([v for v in qa_pairs.values()])
    llm_response = await llm.achat(
        messages=[
            Message.from_text(text=f"Merge the given answers into a unified response to the original question", role="system"),
            Message.from_text(text=f"Query: {user_input}\nAnswers: {answers}", role="user"),
        ]
    )
    return llm_response.message.content

# Define the Chatbot agent, reuse `SplitSolveAgent` in a component-oriented fashion.
class Chatbot(ASLAutoma):
    with graph as g:
        a = SplitSolveAgent()
        b = merge_answers

        +a >> ~b
```

Create an instance of `Chatbot` and run it:

```python
async def main():
    chatbot = Chatbot()
    query = "When and where was Einstein born?"
    answer = await chatbot.arun(query)
    print(answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
Albert Einstein was born on March 14, 1879, in Ulm, which is located in the Kingdom of Württemberg, within the German Empire.
```

Note that the preceding `Chatbot` agent can alternatively be expressed in a nested form in ASL:

```python
class Chatbot(ASLAutoma):
    with graph as g:
        # Define the `split_solve` sub-graph
        with graph as split_solve:
            a = break_down_query
            b = query_answer    
            +a >> ~b

        end = merge_answers
        +split_solve >> ~end
```

The code examples above use ASL to define agents. It is worth noting that, besides ASL, Bridgic provides multiple types of programming APIs. An overview of Bridgic’s API hierarchy is illustrated in the figure below. Developers can invoke the core API directly, use the declarative API built on top of it, or create agents using ASL.

<div align="center">
    <img src="docs/images/bridgic_api_hierarchy.png" alt="Bridgic API Hierarchy Overview" width="50%"/>
</div>


In all subsequent examples in this README, we consistently use ASL to present the code. If you are interested in defining these examples using the declarative API, please refer to [here](docs/auxiliaries/declarative_api_examples.md).

### 2. Dynamic Routing

The `ferry_to()` API enables an automa to dynamically decide which worker should run next, allowing the workflow to adapt its execution path based on runtime conditions. This capability works hand in hand with static dependency declarations, making the execution process much more adaptive and intelligent. With [dynamic routing](https://docs.bridgic.ai/latest/tutorials/items/core_mechanism/dynamic_routing/) powered by `ferry_to()`, you can easily build agentic systems that adjust their behavior at runtime.

```python
from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.args import System
from bridgic.asl import ASLAutoma, graph

async def routing_request(
    request: str,
    automa: GraphAutoma = System("automa"),
) -> str:
    print(f"Routing request: {request}")
    if "?" in request:  # Route using a simple rule that checks for "?"
        automa.ferry_to("hq", question=request)
    else:
        automa.ferry_to("hg", question=request)

async def handle_question(question: str) -> str:
    print("❓ QUESTION: Processing question")
    llm_response = await llm.achat(
        messages=[
            Message.from_text(text="You are a helpful assistant", role="system"),
            Message.from_text(text=question, role="user"),
        ]
    )
    return llm_response.message.content

async def handle_general(question: str) -> str:
    print("📝 GENERAL: Processing general input")
    llm_response = await llm.achat(
        messages=[
            Message.from_text(text="Carry out the user's instructions faithfully and briefly", role="system"),
            Message.from_text(text=question, role="user"),
        ]
    )
    return llm_response.message.content

class SimpleRouter(ASLAutoma):
    with graph as g:
        start = routing_request
        hq = handle_question
        hg = handle_general

        +start, ~hq, ~hg
```

Create an instance of `SimpleRouter` and run it:

```python
async def main():
    router = SimpleRouter()
    test_requests = [
        "When and where was Einstein born?",
        "Create a poem about love."
    ]
    for request in test_requests:
        print(f"\n--- Processing: {request} ---")
        response = await router.arun(request=request)
        print(f"--- Response: \n{response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
--- Processing: When and where was Einstein born? ---
Routing request: When and where was Einstein born?
❓ QUESTION: Processing question
--- Response: 
Albert Einstein was born on March 14, 1879, in the city of Ulm, in the Kingdom of Württemberg in the German Empire.

--- Processing: Create a poem about love. ---
Routing request: Create a poem about love.
📝 GENERAL: Processing general input
--- Response: 
In whispers soft as twilight's breeze,  
Two souls entwined, a dance with ease.  
Hearts beat in rhythm, a timeless song,  
In love's embrace, where we belong.  
...
```

The smart router example showcases how `ferry_to()` enables conditional execution paths. The system analyzes each request and dynamically chooses the appropriate handler, demonstrating how agents can make dynamic routing decisions based on the nature of incoming data.

### 3. Dynamic Topology

Bridgic introduces a novel orchestration model built on a DDG ([Dynamic Directed Graph](https://docs.bridgic.ai/latest/tutorials/items/core_mechanism/dynamic_topology/)), in which the graph topology can be modified at runtime. A typical use case is dynamically instantiating workers based on the number of items in a list returned by a previous task. Each item requires its own handler, but the number of required handlers is not known until runtime.

ASL provides the ability to declare such dynamic behaviors using lambda functions. Here's an example:

```python
from typing import List
from bridgic.core.automa.args import ResultDispatchingRule
from bridgic.asl import ASLAutoma, graph, concurrent, Settings, ASLField


async def produce_task(user_input: int) -> List[int]:
    tasks = [i for i in range(user_input)]
    return tasks

async def task_handler(sub_task: int) -> int:
    res = sub_task + 1
    return res


class DynamicGraph(ASLAutoma):
    with graph(user_input=ASLField(type=int)) as g:
        producer = produce_task

        with concurrent(tasks = ASLField(type=list, dispatching_rule=ResultDispatchingRule.IN_ORDER)) as c:
            dynamic_handler = lambda tasks: (
                task_handler *Settings(key=f"handler_{task}")
                for task in tasks
            )

        +producer >> ~c
```

In this example, the `producer` worker generates a dynamic list based on `user_input`. Each element in the list is assigned to a `task_handler` worker for processing. A `concurrent` container is used to represent a graph structure in which all internal workers execute concurrently.

<div align="center">
    <img src="docs/docs/tutorials/imgs/dynamic_topo.png" alt="Dynamic Topology Example" width="35%"/>
</div>

Create an instance of `DynamicGraph` and run it:

```python
async def main():
    dynamic_graph = DynamicGraph()
    result = await dynamic_graph.arun(user_input=3)
    print(f"--- Result: \n{result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
--- Result: 
[1, 2, 3]
```

### 4. Parameter Resolving

The following example demonstrates the capability of [parameter resolving](https://docs.bridgic.ai/latest/tutorials/items/core_mechanism/parameter_resolving/). Suppose we are building a RAG-based question-answering system: the user input is processed through two concurrent retrieval paths—keyword search and semantic search. Each path retrieves a set of chunks, which are then merged and used to generate a retrieval-augmented response.

```python

from typing import List, Tuple
from bridgic.asl import ASLAutoma, graph, Settings
from bridgic.core.automa.args import ArgsMappingRule, From

async def pre_process(user_input: str) -> str:
    return user_input.strip()

async def keyword_search(query: str) -> List[str]:
    # Simulate keyword search by returning a fixed list of chunks.
    chunks = [
        "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg, Germany  (now simply part of modern Germany).",
        "Einstein was born into a secular Jewish family Biography.",
        "Einstein had one sister, Maja, who was born two years after him.",
    ]
    return chunks

async def semantic_search(query: str) -> List[str]:
    # Simulate semantic search by returning a fixed list of chunks.
    chunks = [
        "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire (now part of Germany).",
        "Shortly after his birth, his family moved to Munich, where he spent most of his childhood.",
        "Einstein excelled at physics and mathematics from an early age, teaching himself algebra, calculus, and Euclidean geometry by age twelve.",
    ]
    return chunks

async def synthesize_response(
    search_results: Tuple[List[str], List[str]], 
    query: str = From("pre_process")
) -> str:
    chunks_by_keyword, chunks_by_semantic = search_results
    all_chunks = chunks_by_keyword + chunks_by_semantic
    prompt = f"{query}\n---\nAnswer the above question based on the following references.\n{all_chunks}"
    print(f"{prompt}\n------------------\n")
    llm_response = await llm.achat(
        messages=[
            Message.from_text(text="You are a helpful assistant", role="system"),
            Message.from_text(text=prompt, role="user"),
        ]
    )
    return llm_response.message.content

class RAGProcessor(ASLAutoma):
    with graph as g:
        pre_process = pre_process
        k = keyword_search
        s = semantic_search
        output = synthesize_response *Settings(args_mapping_rule=ArgsMappingRule.MERGE)
        
        +pre_process >> (k & s) >> ~output
```

<div align="center">
    <img src="docs/images/parameter_resolving_demo.png" alt="Bridgic Prameter Resolving Example" width="50%"/>
</div>


**Key points:**

- **`query` argument** - The `query` arguments of `keyword_search` and `semantic_search` are received from the result of `pre_process` through the **Arguments Mapping** mechanism.
- **`*Settings(args_mapping_rule=ArgsMappingRule.MERGE)`** - the `MERGE` mode of the **Arguments Mapping** rule is specified, which makes the results of `keyword_search` and `semantic_search` merged into the `search_results` argument of `synthesize_response`.
- **`From("pre_process")`** - Injects the result of `pre_process` into the `query` argument of `synthesize_response`.

Create an instance of `RAGProcessor` and run it:

```python
async def main():
    rag = RAGProcessor()
    result = await rag.arun(user_input="When and where was Einstein born?")
    print(f"Final response: \n{result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
When and where was Einstein born?
---
Answer the above question based on the following references.
['Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg, Germany  (now simply part of modern Germany).', 'Einstein was born into a secular Jewish family Biography.', 'Einstein had one sister, Maja, who was born two years after him.', 'Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg in the German Empire (now part of Germany).', 'Shortly after his birth, his family moved to Munich, where he spent most of his childhood.', 'Einstein excelled at physics and mathematics from an early age, teaching himself algebra, calculus, and Euclidean geometry by age twelve.']
------------------

Final response: 
Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg, Germany (now part of modern Germany).
```

### 5. ReAct in Bridgic

```python
from bridgic.core.agentic import ReActAutoma

async def get_weather(
    city: str,
) -> str:
    """
    Retrieves current weather for the given city.

    Parameters
    ----------
    city : str
        The city to get the weather of, e.g. New York.
    
    Returns
    -------
    str
        The weather for the given city.
    """
    # Mock the weather API call.
    return f"The weather in {city} is sunny today and the temperature is 20 degrees Celsius."

async def main():
    react = ReActAutoma(
        llm=llm,
        tools=[get_weather],
        system_prompt="You are a weatherman that is good at forecasting weather by using tools.",
    )
    result = await react.arun(user_msg="What is the weather in Tokyo?")
    print(f"Final response: \n{result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
Final response: 
The weather in Tokyo is sunny today, with a temperature of 20 degrees Celsius.
```

In Bridgic, an automa can be resued as a tool by [`ReActAutoma`](https://docs.bridgic.ai/latest/reference/bridgic-core/bridgic/core/agentic/#bridgic.core.agentic.ReActAutoma), in a component-oriented fashion.

```python
from bridgic.asl import ASLAutoma, graph
from bridgic.core.agentic.tool_specs import as_tool
from bridgic.core.agentic import ReActAutoma

def multiply(x: int, y: int) -> int:
    """
    This function is used to multiply two numbers.

    Parameters
    ----------
    x : int
        The first number to multiply
    y : int
        The second number to multiply

    Returns
    -------
    int
        The product of the two numbers
    """
    return x * y

@as_tool(multiply)
class MultiplyAutoma(ASLAutoma):
    with graph as g:
        start = multiply
        +start, ~start

async def main():
    react = ReActAutoma(
        llm=llm,
        system_prompt="You are a helpful assistant that is good at calculating by using tools.",
    )
    result = await react.arun(
        user_msg="What is 235 * 4689?",
        chat_history=[
            {
                "role": "user",
                "content": "Could you help me to do some calculations?",
            },
            {
                "role": "assistant",
                "content": "Of course, I can help you with that.",
            }
        ],
        # tools may be provided at runtime in Bridgic `ReActAutoma`.
        tools=[MultiplyAutoma],
    )
    print(f"Final response: \n{result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```
Final response: 
The result of multiplying 235 by 4689 is 1,101,915.
```

**Key points:**

- **`tools=[MultiplyAutoma]`** - **Automa as a tool!**
- **`tools=[MultiplyAutoma]`** - The `tools` argument can be passed either to `react.arun` at runtime or during the initialization of `ReActAutoma`.

## 🤖 Building Complex Agentic System

By combining these features, you can build a Bridgic-style agentic system that can:

- **Execute well-defined workflows** through static dependencies (ASL or `@worker`);
- **Adapt intelligently** to different situations according to runtime conditions (`ferry_to` or Dynamic Topology);
- **Process complex data** across multiple steps.

Whether you're building simple workflows or complex autonomous agents, Bridgic provides the dev tools to define your logic clearly while retaining the flexibility required for intelligent, adaptive behavior.

More features will be added in the near future. :)

## 📚 Documents

For more about development skills of Bridgic, see:

- [Tutorials](https://docs.bridgic.ai/latest/tutorials/)
- [Understanding](https://docs.bridgic.ai/latest/home/introduction/)
- [ASL Syntax Learning](https://docs.bridgic.ai/latest/tutorials/items/asl/quick_start/)
- [Model Integration](https://docs.bridgic.ai/latest/tutorials/items/model_integration/)
- [Observability](https://docs.bridgic.ai/latest/tutorials/items/observability/)

## 📄 License

This repository is licensed under the [MIT License](/LICENSE).


## 🤝 Contributing

For contribution guidelines and instructions, please see [CONTRIBUTING](/CONTRIBUTING.md).
