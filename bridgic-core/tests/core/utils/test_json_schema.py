from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema
from pydantic.json_schema import SkipJsonSchema
from typing import Any, List, Dict, Set, Tuple, Optional, Union, Annotated
from enum import Enum
from datetime import datetime, date, time
import json

from bridgic.core.automa.args._args_descriptor import From, System
from bridgic.core.utils._json_schema import create_func_params_json_schema

# Test case 1: the simplest case
def f1():
    ...

def test_func_params_json_schema_case_1():
    json_schema = create_func_params_json_schema(f1)
    # json_schema should be like:
    # 
    # {
    #   "properties": {},
    #   "title": "f1",
    #   "type": "object"
    # }    

    class F1Model(BaseModel):
        model_config = ConfigDict(title="f1")
    assert json_schema == F1Model.model_json_schema()

# Test case 2: parameters with type hints and defaults
def f2(a, b: str, c: int = 5, d: float = 6):
    ...

def test_func_params_json_schema_case_2():
    json_schema = create_func_params_json_schema(f2)
    # json_schema should be like:
    # 
    # {
    #   "properties": {
    #     "a": {
    #       "title": "A"
    #     },
    #     "b": {
    #       "title": "B",
    #       "type": "string"
    #     },
    #     "c": {
    #       "default": 5,
    #       "title": "C",
    #       "type": "integer"
    #     },
    #     "d": {
    #       "default": 6,
    #       "title": "D",
    #       "type": "number"
    #     }
    #   },
    #   "required": [
    #     "a",
    #     "b"
    #   ],
    #   "title": "f2",
    #   "type": "object"
    # }
    class F2Model(BaseModel):
        a: Any
        b: str
        c: int = 5
        d: float = 6
        model_config = ConfigDict(title="f2")
    assert json_schema == F2Model.model_json_schema()

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# Test case 3: parameters of list, dict, set, tuple, enum.
def f3(
    a: List[int],
    b: Optional[Dict[str, int]],
    c: List[str] = [],
    d: Dict[str, int] = {"x": 1, "y": 2},
    e: Color = Color.RED,
    f: Optional[Set[int]] = None,
    g: Tuple[int, str] = (1, "one"),
):
    ...

def test_func_params_json_schema_case_3():
    json_schema = create_func_params_json_schema(f3)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "Color": {
    #       "enum": [
    #         "red",
    #         "green",
    #         "blue"
    #       ],
    #       "title": "Color",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "a": {
    #       "items": {
    #         "type": "integer"
    #       },
    #       "title": "A",
    #       "type": "array"
    #     },
    #     "b": {
    #       "anyOf": [
    #         {
    #           "additionalProperties": {
    #             "type": "integer"
    #           },
    #           "type": "object"
    #         },
    #         {
    #           "type": "null"
    #         }
    #       ],
    #       "title": "B"
    #     },
    #     "c": {
    #       "default": [],
    #       "items": {
    #         "type": "string"
    #       },
    #       "title": "C",
    #       "type": "array"
    #     },
    #     "d": {
    #       "additionalProperties": {
    #         "type": "integer"
    #       },
    #       "default": {
    #         "x": 1,
    #         "y": 2
    #       },
    #       "title": "D",
    #       "type": "object"
    #     },
    #     "e": {
    #       "$ref": "#/$defs/Color",
    #       "default": "red"
    #     },
    #     "f": {
    #       "anyOf": [
    #         {
    #           "items": {
    #             "type": "integer"
    #           },
    #           "type": "array",
    #           "uniqueItems": true
    #         },
    #         {
    #           "type": "null"
    #         }
    #       ],
    #       "default": null,
    #       "title": "F"
    #     },
    #     "g": {
    #       "default": [
    #         1,
    #         "one"
    #       ],
    #       "maxItems": 2,
    #       "minItems": 2,
    #       "prefixItems": [
    #         {
    #           "type": "integer"
    #         },
    #         {
    #           "type": "string"
    #         }
    #       ],
    #       "title": "G",
    #       "type": "array"
    #     }
    #   },
    #   "required": [
    #     "a",
    #     "b"
    #   ],
    #   "title": "f3",
    #   "type": "object"
    # }
    class F3Model(BaseModel):
        a: List[int]
        b: Optional[Dict[str, int]]
        c: List[str] = []
        d: Dict[str, int] = {"x": 1, "y": 2}
        e: Color = Color.RED
        f: Optional[Set[int]] = None
        g: Tuple[int, str] = (1, "one")
        model_config = ConfigDict(title="f3")
    assert json_schema == F3Model.model_json_schema()

# Test case 4: parameters of datetime, date, time.
_current_datetime = datetime.now()
_current_date = date.today()
_current_time = datetime.now().time()

def f4(
    a: datetime,
    b: date,
    c: time,
    d: datetime = _current_datetime,
    e: date = _current_date,
    f: time = _current_time,
):
    ...

def test_func_params_json_schema_case_4():
    json_schema = create_func_params_json_schema(f4)
    # json_schema should be like:
    # 
    # {
    #   "properties": {
    #     "a": {
    #       "format": "date-time",
    #       "title": "A",
    #       "type": "string"
    #     },
    #     "b": {
    #       "format": "date",
    #       "title": "B",
    #       "type": "string"
    #     },
    #     "c": {
    #       "format": "time",
    #       "title": "C",
    #       "type": "string"
    #     },
    #     "d": {
    #       "default": "2025-10-04T16:33:15.918430",
    #       "format": "date-time",
    #       "title": "D",
    #       "type": "string"
    #     },
    #     "e": {
    #       "default": "2025-10-04",
    #       "format": "date",
    #       "title": "E",
    #       "type": "string"
    #     },
    #     "f": {
    #       "default": "16:33:15.918438",
    #       "format": "time",
    #       "title": "F",
    #       "type": "string"
    #     }
    #   },
    #   "required": [
    #     "a",
    #     "b",
    #     "c"
    #   ],
    #   "title": "f4",
    #   "type": "object"
    # }
    class F4Model(BaseModel):
        a: datetime
        b: date
        c: time
        d: datetime = _current_datetime
        e: date = _current_date
        f: time = _current_time
        model_config = ConfigDict(title="f4")
    assert json_schema == F4Model.model_json_schema()

# Test case 5: parameters with Field defaults, or annotated with str / Field / WithJsonSchema.
def f5(
    a: Annotated[float, "Description for a"],
    b: Annotated[str, Field(description="Description for b")],
    c: Annotated[str, Field(default="c-default", description="Description for c")],
    # d: Annotated[int, WithJsonSchema({"description": "Description for d"})],
    # !!! Note: there may be a bug in pydantic. `type` field will be missing when `WithJsonSchema` is used. Therefore, `WithJsonSchema` is not recommended to be used.
    e: str = Field(default="e-default", description="Description for e"),
    f: str = Field(description="Description for f"),
    g: Annotated[Color, "Description for g"] = Color.GREEN,
):
    ...

def test_func_params_json_schema_case_5():
    json_schema = create_func_params_json_schema(f5)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "Color": {
    #       "enum": [
    #         "red",
    #         "green",
    #         "blue"
    #       ],
    #       "title": "Color",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "a": {
    #       "description": "Description for a",
    #       "title": "A",
    #       "type": "number"
    #     },
    #     "b": {
    #       "description": "Description for b",
    #       "title": "B",
    #       "type": "string"
    #     },
    #     "c": {
    #       "default": "c-default",
    #       "description": "Description for c",
    #       "title": "C",
    #       "type": "string"
    #     },
    #     "d": {
    #       "description": "Description for d",
    #       "title": "D"
    #     },
    #     "e": {
    #       "default": "e-default",
    #       "description": "Description for e",
    #       "title": "E",
    #       "type": "string"
    #     },
    #     "f": {
    #       "description": "Description for f",
    #       "title": "F",
    #       "type": "string"
    #     },
    #     "g": {
    #       "$ref": "#/$defs/Color",
    #       "default": "green",
    #       "description": "Description for g"
    #     }
    #   },
    #   "required": [
    #     "a",
    #     "b",
    #     "d",
    #     "f"
    #   ],
    #   "title": "f5",
    #   "type": "object"
    # }
    class F5Model(BaseModel):
        a: Annotated[float, Field(description="Description for a")]
        b: Annotated[str, Field(description="Description for b")]
        c: Annotated[str, Field(default="c-default", description="Description for c")]
        # d: Annotated[int, WithJsonSchema({"description": "Description for d"})]
        e: str = Field(default="e-default", description="Description for e")
        f: str = Field(description="Description for f")
        g: Annotated[Color, Field(description="Description for g")] = Color.GREEN
        model_config = ConfigDict(title="f5")
    assert json_schema == F5Model.model_json_schema()

# Test case 6: conflict test for Field defaults, annotated str / Field, etc.
def f6(
    a: Annotated[str, "Description for a annotated"] = Field(description="Description for a in field default"),
    b: Annotated[str, "Description for b annotated"] = Field(default="b-default"),
    c: Annotated[str, Field(description="Description for c annotated field", default="c-default annotated")] = "c-default value",
    d: Annotated[str, Field(description="Description for d annotated field", default="d-default annotated")] = Field(default="d-default field", description="Description for d in field default"),
    e: Annotated[str, Field(description="Description for e annotated field", default="e-default annotated")] = Field(default="e-default field"),
):
    ...

def test_func_params_json_schema_case_6():
    json_schema = create_func_params_json_schema(f6)
    # json_schema should be like:
    # 
    # {
    #   "properties": {
    #     "a": {
    #       "description": "Description for a in field default",
    #       "title": "A",
    #       "type": "string"
    #     },
    #     "b": {
    #       "default": "b-default",
    #       "description": "Description for b annotated",
    #       "title": "B",
    #       "type": "string"
    #     },
    #     "c": {
    #       "default": "c-default value",
    #       "description": "Description for c annotated field",
    #       "title": "C",
    #       "type": "string"
    #     },
    #     "d": {
    #       "default": "d-default field",
    #       "description": "Description for d in field default",
    #       "title": "D",
    #       "type": "string"
    #     },
    #     "e": {
    #       "default": "e-default field",
    #       "description": "Description for e annotated field",
    #       "title": "E",
    #       "type": "string"
    #     }
    #   },
    #   "required": [
    #     "a"
    #   ],
    #   "title": "f6",
    #   "type": "object"
    # }


    class F6Model(BaseModel):
        a: str = Field(description="Description for a in field default")
        b: str = Field(default="b-default", description="Description for b annotated")
        c: str = Field(description="Description for c annotated field", default="c-default value")
        d: str = Field(default="d-default field", description="Description for d in field default")
        e: str = Field(default="e-default field", description="Description for e annotated field")
        model_config = ConfigDict(title="f6")
    assert json_schema == F6Model.model_json_schema()

# Test case 7: parameters with mixed types and SkipJsonSchema.
def f7(
    a: Union[str, Color] = Field(description="Description for a"),
    b: SkipJsonSchema[Union[str, Color]] = Field(description="Description for b"),
    c: Union[str, SkipJsonSchema[Color]] = Field(default=None, description="Description for c"),
):
    ...

def test_func_params_json_schema_case_7():
    json_schema = create_func_params_json_schema(f7)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "Color": {
    #       "enum": [
    #         "red",
    #         "green",
    #         "blue"
    #       ],
    #       "title": "Color",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "a": {
    #       "anyOf": [
    #         {
    #           "type": "string"
    #         },
    #         {
    #           "$ref": "#/$defs/Color"
    #         }
    #       ],
    #       "description": "Description for a",
    #       "title": "A"
    #     },
    #     "c": {
    #       "default": null,
    #       "description": "Description for c",
    #       "title": "C",
    #       "type": "string"
    #     }
    #   },
    #   "required": [
    #     "a"
    #   ],
    #   "title": "f7",
    #   "type": "object"
    # }

    class F7Model(BaseModel):
        a: Union[str, Color] = Field(description="Description for a")
        c: str = Field(default=None, description="Description for c")
        model_config = ConfigDict(title="f7")
    assert json_schema == F7Model.model_json_schema()

# Test case 8: parameters with description in Numpydoc-style docstring.

class WeatherUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

def get_weather_v1(
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

def test_func_params_json_schema_case_8():
    json_schema = create_func_params_json_schema(get_weather_v1)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "WeatherUnit": {
    #       "enum": [
    #         "celsius",
    #         "fahrenheit"
    #       ],
    #       "title": "WeatherUnit",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "location": {
    #       "description": "City and country e.g. Bogot\u00e1, Colombia.",
    #       "title": "Location",
    #       "type": "string"
    #     },
    #     "unit": {
    #       "$ref": "#/$defs/WeatherUnit",
    #       "description": "Units the temperature will be returned in."
    #     }
    #   },
    #   "required": [
    #     "location",
    #     "unit"
    #   ],
    #   "title": "get_weather_v1",
    #   "type": "object"
    # }

    class F8Model(BaseModel):
        location: str = Field(description="City and country e.g. Bogotá, Colombia.")
        unit: WeatherUnit = Field(description="Units the temperature will be returned in.")
        model_config = ConfigDict(title="get_weather_v1")
    assert json_schema == F8Model.model_json_schema()

# Test case 9: parameters with description in Google-style docstring.
def get_weather_v2(
    location: str,
    unit: WeatherUnit,
) -> str:
    """
    Retrieves current weather for the given location.

    Args:
        location (str): City and country e.g. Bogotá, Colombia.
        unit (WeatherUnit): Units the temperature will be returned in.

    Returns:
        str: The weather for the given location.
    """
    ...

def test_func_params_json_schema_case_9():
    json_schema = create_func_params_json_schema(get_weather_v2)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "WeatherUnit": {
    #       "enum": [
    #         "celsius",
    #         "fahrenheit"
    #       ],
    #       "title": "WeatherUnit",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "location": {
    #       "description": "City and country e.g. Bogot\u00e1, Colombia.",
    #       "title": "Location",
    #       "type": "string"
    #     },
    #     "unit": {
    #       "$ref": "#/$defs/WeatherUnit",
    #       "description": "Units the temperature will be returned in."
    #     }
    #   },
    #   "required": [
    #     "location",
    #     "unit"
    #   ],
    #   "title": "get_weather_v2",
    #   "type": "object"
    # }

    class F9Model(BaseModel):
        location: str = Field(description="City and country e.g. Bogotá, Colombia.")
        unit: WeatherUnit = Field(description="Units the temperature will be returned in.")
        model_config = ConfigDict(title="get_weather_v2")
    assert json_schema == F9Model.model_json_schema()

# Test case 10: parameters with nested BaseModels + reST-style docstring.

class Country(BaseModel):
    code: str          # Country code, e.g. "US", "CN"
    name: str          # Full country name, e.g. "United States"
    region: Optional[str] = None  # Region, e.g. "North America"

class Location(BaseModel):
    city: str
    country: Country  

def get_weather_v3(
    location: Location,
    unit: WeatherUnit,
) -> str:
    """
    Retrieves current weather for the given location.

    :param location: City and country e.g. Bogotá, Colombia.
    :type location: Location
    :param unit: Units the temperature will be returned in.
    :type unit: WeatherUnit
    :returns: The weather for the given location.
    :rtype: str
    """
    ...

def test_func_params_json_schema_case_10():
    json_schema = create_func_params_json_schema(get_weather_v3)
    # json_schema should be like:
    # 
    # {
    #   "$defs": {
    #     "Country": {
    #       "properties": {
    #         "code": {
    #           "title": "Code",
    #           "type": "string"
    #         },
    #         "name": {
    #           "title": "Name",
    #           "type": "string"
    #         },
    #         "region": {
    #           "anyOf": [
    #             {
    #               "type": "string"
    #             },
    #             {
    #               "type": "null"
    #             }
    #           ],
    #           "default": null,
    #           "title": "Region"
    #         }
    #       },
    #       "required": [
    #         "code",
    #         "name"
    #       ],
    #       "title": "Country",
    #       "type": "object"
    #     },
    #     "Location": {
    #       "properties": {
    #         "city": {
    #           "title": "City",
    #           "type": "string"
    #         },
    #         "country": {
    #           "$ref": "#/$defs/Country"
    #         }
    #       },
    #       "required": [
    #         "city",
    #         "country"
    #       ],
    #       "title": "Location",
    #       "type": "object"
    #     },
    #     "WeatherUnit": {
    #       "enum": [
    #         "celsius",
    #         "fahrenheit"
    #       ],
    #       "title": "WeatherUnit",
    #       "type": "string"
    #     }
    #   },
    #   "properties": {
    #     "location": {
    #       "$ref": "#/$defs/Location",
    #       "description": "City and country e.g. Bogot\u00e1, Colombia."
    #     },
    #     "unit": {
    #       "$ref": "#/$defs/WeatherUnit",
    #       "description": "Units the temperature will be returned in."
    #     }
    #   },
    #   "required": [
    #     "location",
    #     "unit"
    #   ],
    #   "title": "get_weather_v3",
    #   "type": "object"
    # }

    class F10Model(BaseModel):
        location: Location = Field(description="City and country e.g. Bogotá, Colombia.")
        unit: WeatherUnit = Field(description="Units the temperature will be returned in.")
        model_config = ConfigDict(title="get_weather_v3")
    assert json_schema == F10Model.model_json_schema()

# Test case 11: parameters with custom class + Epytext docstring (i.e.,  Epydoc-style, a javadoc like style).

from pydantic_core import core_schema as cs
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue

class MyLocation:
    _country: str
    _city: str

    def __init__(self, country: str, city: str):
        self._country = country
        self._city = city

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> cs.CoreSchema:
        return cs.typed_dict_schema(
            {
                'country': cs.typed_dict_field(cs.str_schema()),
                'city': cs.typed_dict_field(cs.str_schema()),
            },
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        # Customize json schema here if needed!
        json_schema['examples'] = [
            {
                'country': 'United States',
                'city': 'New York',
            }
        ]
        json_schema['title'] = 'MyLocation'
        return json_schema

def get_weather_v4(
    location: MyLocation,
    unit: WeatherUnit,
) -> str:
    """
    Retrieves current weather for the given location.

    @param location: City and country e.g. Bogotá, Colombia.
    @type location: MyLocation
    @param unit: Units the temperature will be returned in.
    @type unit: WeatherUnit
    @return: The weather for the given location.
    @rtype: str
    """
    ...

def test_func_params_json_schema_case_11():
    json_schema = create_func_params_json_schema(get_weather_v4)
    expected_json_schema = {
      "$defs": {
        "WeatherUnit": {
          "enum": [
            "celsius",
            "fahrenheit"
          ],
          "title": "WeatherUnit",
          "type": "string"
        }
      },
      "properties": {
        "location": {
          "description": "City and country e.g. Bogot\u00e1, Colombia.",
          "examples": [
            {
              "city": "New York",
              "country": "United States"
            }
          ],
          "properties": {
            "country": {
              "title": "Country",
              "type": "string"
            },
            "city": {
              "title": "City",
              "type": "string"
            }
          },
          "required": [
            "country",
            "city"
          ],
          "title": "MyLocation",
          "type": "object"
        },
        "unit": {
          "$ref": "#/$defs/WeatherUnit",
          "description": "Units the temperature will be returned in."
        }
      },
      "required": [
        "location",
        "unit"
      ],
      "title": "get_weather_v4",
      "type": "object"
    }
    
    assert json_schema == expected_json_schema


def get_weather_v5(
    location: MyLocation,
    unit: WeatherUnit,
    rtx = System("runtime_context"),
    from_value = From("from_value"),
    current_automa =System("automa"),
    default_value = 1,
) -> str:
    """
    Retrieves current weather for the given location.

    @param location: City and country e.g. Bogotá, Colombia.
    @type location: MyLocation
    @param unit: Units the temperature will be returned in.
    @type unit: WeatherUnit
    @param default_value: Default value.
    @type default_value: int
    @return: The weather for the given location.
    @rtype: str
    """
    ...

def test_func_params_json_schema_case_12():
    json_schema = create_func_params_json_schema(get_weather_v5)
    expected_json_schema = {
      "$defs": {
        "WeatherUnit": {
          "enum": [
            "celsius",
            "fahrenheit"
          ],
          "title": "WeatherUnit",
          "type": "string"
        }
      },
      "properties": {
        'default_value': {'default': 1, 'description': 'Default value.', 'title': 'Default Value'},
        "location": {
          "description": "City and country e.g. Bogot\u00e1, Colombia.",
          "examples": [
            {
              "city": "New York",
              "country": "United States"
            }
          ],
          "properties": {
            "country": {
              "title": "Country",
              "type": "string"
            },
            "city": {
              "title": "City",
              "type": "string"
            }
          },
          "required": [
            "country",
            "city"
          ],
          "title": "MyLocation",
          "type": "object"
        },
        "unit": {
          "$ref": "#/$defs/WeatherUnit",
          "description": "Units the temperature will be returned in."
        },
      },
      "required": [
        "location",
        "unit"
      ],
      "title": "get_weather_v5",
      "type": "object"
    }
    
    assert json_schema == expected_json_schema