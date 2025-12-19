import pytest
from enum import Enum

from bridgic.core.agentic.tool_specs import FunctionToolSpec
from bridgic.core.automa.worker import CallableWorker
from bridgic.core.utils._msgpackx import dump_bytes, load_bytes

# Test FunctionToolSpec

class WeatherUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

async def get_weather(
    location: str,
    unit: WeatherUnit,
) -> str:
    """
    Retrieves current weather for the given location.

    Parameters
    ----------
    location : str
        City and country e.g. Bogotá, Colombia.
    unit : WeatherUnit
        Units the temperature will be returned in.
    
    Returns
    -------
    str
        The weather for the given location.
    """
    ...

@pytest.fixture
def function_tool_spec():
    return FunctionToolSpec.from_raw(get_weather)

def test_function_tool_spec(function_tool_spec):
    tool = function_tool_spec.to_tool()
    assert tool.name == "get_weather"
    assert tool.description == "Retrieves current weather for the given location."
    assert tool.parameters["properties"]["location"]["type"] == "string"
    assert tool.parameters["properties"]["location"]["description"] == "City and country e.g. Bogotá, Colombia."
    assert tool.parameters["$defs"]["WeatherUnit"]["enum"] == ['celsius', 'fahrenheit']
    assert tool.parameters["properties"]["unit"]["description"] == "Units the temperature will be returned in."

    my_worker = function_tool_spec.create_worker() 
    assert isinstance(my_worker, CallableWorker)

def test_function_tool_spec_deserialization(function_tool_spec):
    data = dump_bytes(function_tool_spec)
    assert type(data) is bytes
    obj = load_bytes(data)
    assert type(obj) is FunctionToolSpec

    test_function_tool_spec(obj)

