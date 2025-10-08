import pytest
from enum import Enum
from typing import cast

from bridgic.core.intelligence import FunctionToolSpec, AutomaToolSpec
from bridgic.core.automa.worker import CallableWorker
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.utils import msgpackx

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
    data = msgpackx.dump_bytes(function_tool_spec)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is FunctionToolSpec

    test_function_tool_spec(obj)

# Test AutomaToolSpec

class MultiplyAutoma(GraphAutoma):
    @worker(is_start=True, is_output=True)
    async def multiply(self, x: int, y: int):
        return x * y

@pytest.fixture
def automa_tool_spec():
    return AutomaToolSpec.from_raw(
        MultiplyAutoma,
        tool_name="multiply",
        tool_description="Multiply two numbers",
        tool_parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            }
        }
    )

@pytest.mark.asyncio
async def test_automa_tool_spec(automa_tool_spec):
    tool = automa_tool_spec.to_tool()
    assert tool.name == "multiply"
    assert tool.description == "Multiply two numbers"
    assert tool.parameters["properties"]["x"]["type"] == "number"
    assert tool.parameters["properties"]["x"]["description"] == "The first number to multiply"
    assert tool.parameters["properties"]["y"]["type"] == "number"
    assert tool.parameters["properties"]["y"]["description"] == "The second number to multiply"

    my_worker = automa_tool_spec.create_worker()
    assert isinstance(my_worker, MultiplyAutoma)
    multiply_automa = cast(MultiplyAutoma, my_worker)
    result = await multiply_automa.arun(x=2, y=5)
    assert result == 10

@pytest.mark.asyncio
async def test_automa_tool_spec_deserialization(automa_tool_spec):
    data = msgpackx.dump_bytes(automa_tool_spec)
    assert type(data) is bytes
    obj = msgpackx.load_bytes(data)
    assert type(obj) is AutomaToolSpec
    await test_automa_tool_spec(obj)