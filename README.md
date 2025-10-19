Bridgic is an innovative programming framework designed to create agentic systems, from simple workflows to fully autonomous agents. Its APIs are thoughtfully crafted to be both simple and powerful.

## Core Features

* **Orchestration**: Bridgic helps to manage the control flow of your AI applications asynchronously.
* **Dynamic Control Flow**: Bridgic supports dynamic routing based on input data, and even allows workers to be added or removed at runtime.
* **Modularization**: In Bridgic, a complex agentic system can be composed by reusing components through hierarchical nesting.
* **Human-in-the-Loop**: A workflow or an agent built with Bridgic can request feedback from humans whenever needed.
* **Serialization**: Bridgic includes serialization, deserialization, and resuming capabilities to support human-in-the-loop.
* **Parameter Binding**: There are three ways to pass data among workers, including Arguments Mapping, Arguments Injection, and Inputs Propagation.
* **Systematic Integration**: A wide range of tools and LLMs can be seamlessly integrated into the Bridgic world, in a systematic way.
* **Customization**: What Bridgic provides is not a "black box" approach. You have full control over every aspect of your AI applications, such as prompts, context windows, the control flow, and more.

## Install Bridgic

Requires Python 3.9 or higher.

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

For more information and examples, see the [Tutorials](http://127.0.0.1:8000/tutorials/).

## Understanding

See [Understanding Bridgic](http://127.0.0.1:8000/home/introduction/).

## License

This repo is available under the [MIT license](/LICENSE).