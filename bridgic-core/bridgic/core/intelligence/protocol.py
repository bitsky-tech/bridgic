from typing import List, Protocol, Any, Dict, Type, Literal, Union, Optional
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
    schema: Dict[str, Any] = Field(..., description="Schema of the JsonSchema constraint.")

class Regex(BaseModel):
    constraint_type: Literal["regex"] = "regex"
    pattern: str = Field(..., description="Pattern of the Regex constraint.")

class EbnfGrammar(BaseModel):
    constraint_type: Literal["ebnf_grammar"] = "ebnf_grammar"
    syntax: str = Field(..., description="Syntax of the EBNF grammar constraint.")

Constraint = Union[PydanticModel, JsonSchema, EbnfGrammar, Regex]

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
    parameters: Dict[str, Any] = Field(..., default_factory=dict, description="Real parameters that are used to call the tool.")

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
    ) -> List[ToolCall]: ...

    async def atool_select(
        self,
        messages: List[Message],
        tools: List[Tool],
        **kwargs,
    ) -> List[ToolCall]: ...