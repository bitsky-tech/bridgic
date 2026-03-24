# Key imports

```python
# Core classes
from bridgic.amphibious import (
    AmphibiousAutoma,
    CognitiveWorker,
    CognitiveContext,
    think_unit,
    step,
    AgentTrace,
)

# Context layer
from bridgic.amphibious import (
    Context, Exposure, LayeredExposure, EntireExposure,
    CognitiveTools, CognitiveSkills, CognitiveHistory,
    Step, Skill,
)

# Types
from bridgic.amphibious import (
    RunMode, ErrorStrategy,
    WorkflowStep, AgentFallback, ToolResult,
    StepToolCall, ToolArgument,
    TraceStep, RecordedToolCall, StepOutputType,
)

# Sentinel
from bridgic.amphibious import _DELEGATE

# Tools
from bridgic.core.agentic.tool_specs import FunctionToolSpec
```
