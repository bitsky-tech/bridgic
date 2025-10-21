**Bridgic** is an innovative programming framework designed to create agentic systems, ranging from deterministic workflows to autonomous agents. It introduces a new paradigm that simplifies the development of agentic systems.

> ✨ The name "**Bridgic**" is inspired by the idea of *"Bridging Logic and Magic"*. It means seamlessly uniting the precision of *logic* (deterministic execution flows) with the creativity of *magic* (highly autonomous AI).


## Features

* **Orchestration**: Bridgic helps manage the execution flow of your AI applications by leveraging both predefined dependencies and dynamic routing.
* **Parameter Binding**: There are three ways to pass data among workers—Arguments Mapping, Arguments Injection, and Inputs Propagation—thereby eliminating the complexity of global state management.
* **Dynamic Routing**: Bridgic enables conditional branching and intelligent decision-making through an easy-to-use `ferry_to()` API that adapts to runtime dynamics.
* **Dynamic Topology**: The topology can be changed at runtime in Bridgic to support highly autonomous AI applications.
* **Modularity**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Human-in-the-Loop**: A Bridgic-style agentic system can request feedback from human whenever needed to dynamically adjust its execution logic.
* **Serialization**: Bridgic employs a scalable serialization and deserialization mechanism to achieve state persistence and recovery, enabling human-in-the-loop in long-running AI systems.
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