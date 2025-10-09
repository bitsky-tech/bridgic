from typing import List, Protocol, Any, Dict, Type, Literal, Union, Optional, ClassVar, Tuple
from pydantic import BaseModel, Field

from bridgic.core.intelligence.base_llm import Message

###########################################################
# Structures and protocols for constraint generation.
###########################################################

class PydanticModel(BaseModel):
    constraint_type: Literal["pydantic_model"] = "pydantic_model"
    model: Type[BaseModel] = Field(..., description="Model type of the PydanticModel constraint.")

class JsonSchema(BaseModel):
    constraint_type: Literal["json_schema"] = "json_schema"
    name: str = Field(..., description="Name of the JsonSchema constraint.")
    schema_dict: Dict[str, Any] = Field(..., description="Schema of the JsonSchema constraint.")

class Regex(BaseModel):
    constraint_type: Literal["regex"] = "regex"
    pattern: str = Field(..., description="Pattern of the Regex constraint.")
    description: str = Field(..., description="Description of the Regex constraint.")

class RegexPattern:
    INTEGER: ClassVar[Regex] = Regex(pattern=r"-?\d+", description="A regex for matching integers.")
    FLOAT = Regex(pattern=r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)([eE][-+]?\d+)?", description="A regex for matching floats.")
    DATE: ClassVar[Regex] = Regex(pattern=r"\d{4}-\d{2}-\d{2}", description="A regex for matching dates.")
    TIME: ClassVar[Regex] = Regex(pattern=r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?", description="A regex for matching times.")
    DATE_TIME_ISO_8601: ClassVar[Regex] = Regex(pattern=rf"{DATE.pattern}T{TIME.pattern}(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?", description="A regex for matching date-time in ISO 8601 format.")
    IP_V4_ADDRESS: ClassVar[Regex] = Regex(pattern=r"(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)", description="A regex for matching IPv4 addresses.")
    IP_V6_ADDRESS: ClassVar[Regex] = Regex(pattern=r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}", description="A regex for matching IPv6 addresses.")
    EMAIL: ClassVar[Regex] = Regex(pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", description="A regex for matching email addresses.")

class Choice(BaseModel):
    constraint_type: Literal["choice"] = "choice"
    choices: List[str] = Field(..., description="Choices of the choice constraint.")

class EbnfGrammar(BaseModel):
    constraint_type: Literal["ebnf_grammar"] = "ebnf_grammar"
    syntax: str = Field(..., description="Syntax of the EBNF grammar constraint.")
    description: str = Field(..., description="Description of the EBNF grammar constraint.")

class LarkGrammar(BaseModel):
    constraint_type: Literal["lark_grammar"] = "lark_grammar"
    syntax: str = Field(..., description="Syntax of the Lark grammar constraint.")
    description: str = Field(..., description="Description of the Lark grammar constraint.")

Constraint = Union[PydanticModel, JsonSchema, EbnfGrammar, LarkGrammar, Regex, Choice]

class StructuredOutput(Protocol):
    """
    StructuredOutput is a protocol that defines the interface for LLM providers that can output 
    in specified format. The actual formats supported for structured output depend on the real 
    capabilities of the model provider.
    """

    def structured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        **kwargs,
    ) -> Any: ...

    async def astructured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        **kwargs,
    ) -> Any: ...

###########################################################
# Structures and protocols for tool use.
###########################################################

class Tool(BaseModel):
    name: str = Field(..., description="Name of the tool.")
    description: str = Field(..., description="Description of the tool.")
    parameters: Dict[str, Any] = Field(..., description="JSON schema object that describes the parameters of the tool.")

class ToolCall(BaseModel):
    id: Optional[str] = Field(..., description="ID of the tool call.")
    name: str = Field(..., description="Name of the tool.")
    arguments: Dict[str, Any] = Field(..., default_factory=dict, description="Real arguments that are used to call the tool.")

class ToolSelect(Protocol):
    """
    ToolSelect is a protocol that defines the interface for LLM providers that can select tools 
    to use and decide their specific parameters.
    """

    def tool_select(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Union[str, None]]: ...

    async def atool_select(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> Tuple[List[ToolCall], Union[str, None]]: ...