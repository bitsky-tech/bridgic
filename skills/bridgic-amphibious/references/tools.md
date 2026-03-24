# Tools — `FunctionToolSpec`

## What tools are

Tools are the agent's capabilities — async Python functions that the LLM can invoke during the Act phase of the OTA cycle. The framework wraps them in `FunctionToolSpec` to extract metadata for the LLM prompt.

## How `FunctionToolSpec.from_raw()` works

```python
from bridgic.core.agentic.tool_specs import FunctionToolSpec

async def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities.

    Parameters
    ----------
    origin : str
        Departure city name.
    destination : str
        Arrival city name.
    date : str
        Travel date in YYYY-MM-DD format.
    """
    return f"Found 3 flights from {origin} to {destination} on {date}"

tool = FunctionToolSpec.from_raw(search_flights)
```

**What gets extracted automatically:**
- **Tool name** ← function name (`search_flights`)
- **Description** ← docstring first paragraph (`"Search for available flights..."`) — this is what the LLM reads to decide when to use the tool
- **Parameters** ← type hints + docstring parameter descriptions → JSON Schema — this tells the LLM what arguments to provide and their types

**The LLM sees** the tool name, description, and parameter schema in its prompt. Good docstrings directly improve tool selection accuracy.

## Requirements

| Requirement | Why |
|------------|-----|
| `async def` | Tools execute concurrently via `asyncio.gather()` by default |
| Type hints on all parameters | Extracted into JSON Schema for LLM parameter generation |
| Docstring | Becomes the tool description the LLM uses for tool selection |
| Return `str` | Tool results are injected back into the prompt as text |

## Best practices

- **Descriptive docstrings** — The LLM chooses tools based on descriptions. Be specific: "Search for flights" not "Do a search"
- **Parameter descriptions** — Add parameter docs (NumPy or Google style). The LLM needs to know what each parameter means
- **Return meaningful strings** — The LLM reads the return value. Format results clearly: `"Found 3 flights: ..."` not `"OK"`
- **One capability per tool** — Don't combine unrelated actions. `search_flights` and `book_flight` should be separate tools
- **Error messages in return** — Return error info as strings rather than raising exceptions, so the LLM can self-correct: `"Error: invalid date format, use YYYY-MM-DD"`

## Overriding extracted metadata

```python
# Override name and/or description
tool = FunctionToolSpec.from_raw(
    my_func,
    tool_name="custom_name",
    tool_description="Custom description for the LLM",
)
```
