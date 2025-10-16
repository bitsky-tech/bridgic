import pytest
from enum import Enum
from typing import cast

from bridgic.core.agentic.tool_specs import FunctionToolSpec, AutomaToolSpec, as_tool
from bridgic.core.automa.worker import CallableWorker
from bridgic.core.automa import GraphAutoma, worker
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

# Test AutomaToolSpec with customized tool name, description and parameters

class MultiplyAutoma(GraphAutoma):
    @worker(is_start=True, is_output=True)
    async def multiply(self, x: float, y: float):
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
    data = dump_bytes(automa_tool_spec)
    assert type(data) is bytes
    obj = load_bytes(data)
    assert type(obj) is AutomaToolSpec
    await test_automa_tool_spec(obj)

# Test AutomaToolSpec with decorated automa class

def add(x: int, y: int):
    """
    Add two numbers

    Parameters
    ----------
    x : int
        The first number to add
    y : int
        The second number to add
    """
    # Note: this function need not to be implemented.
    ...

@as_tool(add)
class AddAutoma(GraphAutoma):
    @worker(is_start=True, is_output=True)
    async def add(self, x: int, y: int):
        return x + y

@pytest.fixture
def decorated_automa_tool_spec():
    return AutomaToolSpec.from_raw(AddAutoma)

@pytest.mark.asyncio
async def test_decorated_automa_tool_spec(decorated_automa_tool_spec):
    tool = decorated_automa_tool_spec.to_tool()
    assert tool.name == "add"
    assert tool.description == "Add two numbers"
    assert tool.parameters["properties"]["x"]["type"] == "integer"
    assert tool.parameters["properties"]["x"]["description"] == "The first number to add"
    assert tool.parameters["properties"]["y"]["type"] == "integer"
    assert tool.parameters["properties"]["y"]["description"] == "The second number to add"

    my_worker = decorated_automa_tool_spec.create_worker()
    assert isinstance(my_worker, AddAutoma)
    add_automa = cast(AddAutoma, my_worker)
    result = await add_automa.arun(x=2, y=5)
    assert result == 7

@pytest.mark.asyncio
async def test_decorated_automa_tool_spec_deserialization(decorated_automa_tool_spec):
    data = dump_bytes(decorated_automa_tool_spec)
    assert type(data) is bytes
    obj = load_bytes(data)
    assert type(obj) is AutomaToolSpec
    await test_decorated_automa_tool_spec(obj)
