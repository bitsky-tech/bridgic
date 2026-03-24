# Execution Tracing — `AgentTrace`

Record every observe-think-act cycle for debugging, replay, or analysis.

## Enabling

```python
result = await agent.arun(context=ctx, trace_running=True)
```

## Accessing the trace

```python
trace = agent._agent_trace          # AgentTrace instance
data  = trace.build()               # {"steps": [TraceStep, ...], "metadata": {}}
data  = trace.build(metadata={"run": "v1"})  # with custom metadata
```

## Saving / loading

```python
agent._agent_trace.save("trace.json")
agent._agent_trace.save("trace.json", metadata={"experiment": "A"})

loaded = AgentTrace.load("trace.json")  # returns dict
```

## `TraceStep` fields

Each step records one OTA cycle:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Think-unit name (e.g. `"planner"`) |
| `step_content` | `str` | Raw LLM output text |
| `tool_calls` | `List[RecordedToolCall]` | Tool invocations in this step |
| `observation` | `Optional[str]` | Observation fed to the LLM |
| `observation_hash` | `Optional[str]` | Hash for deduplication |
| `output_type` | `StepOutputType` | `TOOL_CALLS`, `STRUCTURED`, or `CONTENT_ONLY` |
| `structured_output` | `Optional[Dict]` | Parsed output when `output_schema` is used |
| `structured_output_class` | `Optional[str]` | Class name of the output schema |

## `RecordedToolCall` fields

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool called |
| `tool_arguments` | `Dict[str, Any]` | Arguments passed |
| `tool_result` | `Any` | Return value |
| `success` | `bool` | Whether the call succeeded |
| `error` | `Optional[str]` | Error message if failed |

## Imports

```python
from bridgic.amphibious import AgentTrace, TraceStep, RecordedToolCall, StepOutputType
```
