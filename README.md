**Bridgic** is an innovative programming framework designed to create agentic systems. From simple workflows to complex autonomous agents, Bridgic provides a new paradigm to design your agentic system with ease.

> ✨ The name "**Bridgic**" is inspired by the idea of *"Bridging Logic and Magic"*. It means seamlessly uniting the precision of *logic* (deterministic execution flows) with the creativity of *magic* (highly autonomous AI).


## Features

* **Orchestration**: Bridgic helps you express your program logic in the form of Worker-Automa choreography, making asynchronous programming easy and flexible.
* **Parameter Binding**: There are three ways to pass data among workers, including Arguments Mapping, Arguments Injection, and Inputs Propagation, avoiding the complexity of unmanageable global state.
* **Dynamic Routing**: Bridgic supports you to dynamically route to different workers based on runtime conditions using the `ferry_to()` method, enabling intelligent branching and conditional execution flows that adapt to real-time data and context.
* **Dynamic Topology**: Bridgic allows you to modify the topology at runtime, enabling your workflows to evolve and adapt without restarting the entire system.
* **Modularity**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Human-in-the-Loop**: An Bridgic style agentic system can request feedback from human whenever needed to affect its own execution.
* **Serialization**: Bridgic use a complete set of serialization and deserialization solutions to achieve state persistence and recovery to support a larger human-in-the-loop system.
* **Systematic Integration**: A wide range of tools and LLMs can be seamlessly integrated into the Bridgic world, in a systematic way.
* **Customization**: What Bridgic provides is not a "black box" approach. You have full control over every aspect of your AI applications, such as prompts, context windows, the control flow, and more.

## Install

Python 3.9 or higher version is required.

```bash
pip install bridgic
```

## Example Code

Initialize the running environment for LLM:

```python
import os
from bridgic.llms.openai.openai_llm import OpenAILlm, OpenAIConfiguration

_api_key = os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("OPENAI_MODEL_NAME")

llm = OpenAILlm(
    api_key=_api_key,
    configuration=OpenAIConfiguration(model=_model_name),
)
```

Then, create a `word learning assistant` with code:

```python
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.model.types import Message, Role

class WordLearningAssistant(GraphAutoma):
    @worker(is_start=True)
    async def generate_derivatives(self, word: str):
        response = await llm.achat(
            model=_model_name,
            messages=[
                Message.from_text(text="You are a word learning assistant. Generate derivatives of the input word in a list.", role=Role.SYSTEM),
                Message.from_text(text=word, role=Role.USER),
            ]
        )
        return response.message.content

    @worker(dependencies=["generate_derivatives"], is_output=True)
    async def make_sentences(self, derivatives):
        response = await llm.achat(
            model=_model_name,
            messages=[
                Message.from_text(text="You are a word learning assistant. Make sentences with the input derivatives in a list.", role=Role.SYSTEM),
                Message.from_text(text=derivatives, role=Role.USER),
            ]
        )
        return response.message.content
```

Let's run it:

```python
word_learning_assistant = WordLearningAssistant()
res = await word_learning_assistant.arun(word="happy")
print(res)
```

For more information and examples, see the [Tutorials](https://docs.bridgic.ai/tutorials/).

## Understanding

See [Understanding Bridgic](https://docs.bridgic.ai/home/introduction/).

## License

This repo is available under the [MIT license](/LICENSE).