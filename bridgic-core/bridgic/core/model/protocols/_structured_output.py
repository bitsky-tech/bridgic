from typing import List, Any, Protocol, Literal, Union, Dict, Type, ClassVar
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message

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

class RegexPattern:
    INTEGER: ClassVar[Regex] = Regex(pattern=r"-?\d+")
    FLOAT = Regex(pattern=r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)([eE][-+]?\d+)?")
    DATE: ClassVar[Regex] = Regex(pattern=r"\d{4}-\d{2}-\d{2}")
    TIME: ClassVar[Regex] = Regex(pattern=r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?")
    DATE_TIME_ISO_8601: ClassVar[Regex] = Regex(pattern=rf"{DATE.pattern}T{TIME.pattern}(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?")
    IP_V4_ADDRESS: ClassVar[Regex] = Regex(pattern=r"(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)")
    IP_V6_ADDRESS: ClassVar[Regex] = Regex(pattern=r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}")
    EMAIL: ClassVar[Regex] = Regex(pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

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

Constraint = Union[PydanticModel, JsonSchema, EbnfGrammar, LarkGrammar, Regex, Choice]

class StructuredOutput(Protocol):
    """
    Protocol for LLM providers that support structured output generation.

    StructuredOutput defines the interface for language models that can generate 
    responses in specific formats according to given constraints. This protocol 
    enables controlled output generation for various data structures and formats.

    Methods
    -------
    structured_output
        Synchronous method for generating structured output based on constraints.
    astructured_output
        Asynchronous method for generating structured output based on constraints.

    Note
    ----
    1. Both synchronous and asynchronous methods must be implemented
    2. Supported constraint types depend on the specific LLM provider implementation
    3. Output format is determined by the constraint type provided
    4. Common constraint types include PydanticModel, JsonSchema, Regex, Choice, etc.
    """

    def structured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        **kwargs,
    ) -> Any:
        """
        Generate structured output based on conversation context and constraints.

        Parameters
        ----------
        messages : List[Message]
            The conversation history and current context.
        constraint : Constraint
            The output format constraint. Supported types:
            - PydanticModel: Output as Pydantic model instance
            - JsonSchema: Output as JSON matching the schema
            - Regex: Output matching the regex pattern
            - Choice: Output from predefined choices
            - EbnfGrammar: Output following EBNF grammar rules
            - LarkGrammar: Output following Lark grammar rules
        **kwargs
            Additional keyword arguments for output generation configuration.

        Returns
        -------
        Any
            The structured output matching the specified constraint format.
        """
        ...

    async def astructured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        **kwargs,
    ) -> Any:
        """
        Asynchronously generate structured output based on conversation context and constraints.

        Parameters
        ----------
        messages : List[Message]
            The conversation history and current context.
        constraint : Constraint
            The output format constraint. Supported types:
            - PydanticModel: Output as Pydantic model instance
            - JsonSchema: Output as JSON matching the schema
            - Regex: Output matching the regex pattern
            - Choice: Output from predefined choices
            - EbnfGrammar: Output following EBNF grammar rules
            - LarkGrammar: Output following Lark grammar rules
        **kwargs
            Additional keyword arguments for output generation configuration.

        Returns
        -------
        Any
            The structured output matching the specified constraint format.
        """
        ...