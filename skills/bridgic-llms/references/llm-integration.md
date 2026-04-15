# LLM Integration Reference

Complete API reference for bridgic LLM providers.

## Initialization

### OpenAILlm (recommended)

```python
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration

llm = OpenAILlm(
    api_key="your-api-key",
    api_base="https://api.openai.com/v1",  # or custom endpoint
    configuration=OpenAIConfiguration(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2000,
    ),
    timeout=30.0,
)
```

Supports: `BaseLlm`, `StructuredOutput`, `ToolSelection`.

### OpenAILikeLlm (basic only)

```python
from bridgic.llms.openai_like import OpenAILikeLlm

llm = OpenAILikeLlm(
    api_key="your-api-key",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

Supports: `BaseLlm` only. No structured output, no tool selection.

### VllmServerLlm (self-hosted)

```python
from bridgic.llms.vllm import VllmServerLlm

llm = VllmServerLlm(
    api_key="token",
    api_base="http://localhost:8000/v1",
)
```

Supports: `BaseLlm`, `StructuredOutput`, `ToolSelection`.

---

## Message Types

```python
from bridgic.core.model.types import Message, Role

# Create messages
system_msg = Message.from_text("You are helpful.", role=Role.SYSTEM)
user_msg = Message.from_text("Hello!", role=Role.USER)
assistant_msg = Message.from_text("Hi there!", role=Role.ASSISTANT)
```

Roles: `Role.SYSTEM`, `Role.USER`, `Role.ASSISTANT`.

---

## Chat Interface

```python
response = llm.chat(
    messages=[system_msg, user_msg],
    model="gpt-4o",
    temperature=0.7,
)
print(response.message.content)  # str
```

## Stream Interface

```python
for chunk in llm.stream(messages=[system_msg, user_msg], model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

---

## Structured Output Protocol

Requires `bridgic-llms-openai` or `bridgic-llms-vllm`.

### With Pydantic Model

```python
from pydantic import BaseModel, Field
from bridgic.core.model.protocols import PydanticModel

class MathSolution(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning")
    answer: int = Field(description="Final numerical answer")

solution = llm.structured_output(
    messages=[Message.from_text("What is 15 * 23?", role=Role.USER)],
    constraint=PydanticModel(model=MathSolution),
    model="gpt-4o",
)
print(solution.reasoning)  # str
print(solution.answer)     # int
```

### With JSON Schema

```python
from bridgic.core.model.protocols import JsonSchema

result = llm.structured_output(
    messages=messages,
    constraint=JsonSchema(schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }),
    model="gpt-4o",
)
```

---

## Tool Selection Protocol

Requires `bridgic-llms-openai` or `bridgic-llms-vllm`.

```python
from bridgic.core.model.types import Tool

# Define tools
tools = [
    Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'San Francisco, CA'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    ),
]

# Model selects tool
tool_calls, content = llm.select_tool(
    messages=[Message.from_text("What's the weather in Paris?", role=Role.USER)],
    tools=tools,
    model="gpt-4o",
    tool_choice="auto",
)

for tool_call in tool_calls:
    print(f"Tool: {tool_call.name}")        # str
    print(f"Arguments: {tool_call.arguments}")  # dict
    print(f"Call ID: {tool_call.id}")        # str
```

---

## Common Pitfalls

1. **Do NOT use `OpenAILikeLlm` for structured output or tool selection** — it only implements `BaseLlm`. Use `OpenAILlm` instead.
2. **Do NOT use `OpenAILikeLlm` in bridgic-amphibious projects** — the agent framework requires `StructuredOutput` and `ToolSelection` for `on_agent` mode.
3. **Always load API credentials from environment** — use `dotenv` + `os.environ.get()`, never hardcode.
4. **Set `temperature=0.0` for deterministic workflows** — higher temperatures introduce randomness in `on_workflow` step execution.
