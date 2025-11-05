from typing import List, Any, Protocol, Literal, Union, Dict, Type, ClassVar
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message

class PydanticModel(BaseModel):
    """
    A constraint defined as a Pydantic model for structured LLM output.
    """
    constraint_type: Literal["pydantic_model"] = "pydantic_model"
    """The type of the constraint, in this case `pydantic_model`."""
    model: Type[BaseModel] = Field(..., description="Model type of the PydanticModel constraint.")
    """The Pydantic model type of the constraint."""

class JsonSchema(BaseModel):
    """
    A constraint defined as a JSON schema for structured LLM output.
    """
    constraint_type: Literal["json_schema"] = "json_schema"
    """The type of the constraint, in this case `json_schema`."""
    schema_dict: Dict[str, Any] = Field(..., description="Schema of the JsonSchema constraint.")
    """The JSON schema of the constraint."""

class Regex(BaseModel):
    """
    A constraint defined as a regular expression for structured LLM output.
    """
    constraint_type: Literal["regex"] = "regex"
    """The type of the constraint, in this case `regex`."""
    pattern: str = Field(..., description="Pattern of the Regex constraint.")
    """The regular expression of the constraint."""

class RegexPattern:
    """Constants that define some common regular expressions for structured LLM output."""
    INTEGER: ClassVar[Regex] = Regex(pattern=r"-?\d+")
    """A regular expression for integers."""
    FLOAT = Regex(pattern=r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)([eE][-+]?\d+)?")
    """A regular expression for floats."""
    DATE: ClassVar[Regex] = Regex(pattern=r"\d{4}-\d{2}-\d{2}")
    """A regular expression for dates."""
    TIME: ClassVar[Regex] = Regex(pattern=r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?")
    """A regular expression for times."""
    DATE_TIME_ISO_8601: ClassVar[Regex] = Regex(pattern=rf"{DATE.pattern}T{TIME.pattern}(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?")
    """A regular expression for date-time in ISO 8601 format."""
    IP_V4_ADDRESS: ClassVar[Regex] = Regex(pattern=r"(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)")
    """A regular expression for IPv4 addresses."""
    IP_V6_ADDRESS: ClassVar[Regex] = Regex(pattern=r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}")
    """A regular expression for IPv6 addresses."""
    EMAIL: ClassVar[Regex] = Regex(pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    """A regular expression for email addresses."""

class Choice(BaseModel):
    """
    A constraint defined as a predefined set of choices for structured LLM output.
    """
    constraint_type: Literal["choice"] = "choice"
    """The type of the constraint, in this case `choice`."""
    choices: List[str] = Field(..., description="Choices of the choice constraint.")
    """The choices of the constraint."""

class EbnfGrammar(BaseModel):
    """
    A constraint defined as an EBNF grammar for structured LLM output.
    """
    constraint_type: Literal["ebnf_grammar"] = "ebnf_grammar"
    """The type of the constraint, in this case `ebnf_grammar`."""
    syntax: str = Field(..., description="Syntax of the EBNF grammar constraint.")
    """The syntax of the EBNF grammar constraint."""

class LarkGrammar(BaseModel):
    """
    A constraint defined as a Lark grammar for structured LLM output.
    """
    constraint_type: Literal["lark_grammar"] = "lark_grammar"
    """The type of the constraint, in this case `lark_grammar`."""
    syntax: str = Field(..., description="Syntax of the Lark grammar constraint.")
    """The syntax of the Lark grammar constraint."""

Constraint = Union[PydanticModel, JsonSchema, EbnfGrammar, LarkGrammar, Regex, Choice]
"""The constraint type for structured LLM output."""

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

    Notes
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