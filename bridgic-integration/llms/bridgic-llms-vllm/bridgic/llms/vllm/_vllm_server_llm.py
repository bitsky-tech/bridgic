import httpx
import json

from typing import List, Union, Optional, Dict, Any, Tuple, Literal
from typing_extensions import override, overload
from pydantic import BaseModel
from openai.types.chat import ChatCompletionNamedToolChoiceParam, ChatCompletionMessageFunctionToolCall

from bridgic.core.model import BaseLlm
from bridgic.core.model.types import *
from bridgic.core.model.protocols import StructuredOutput, ToolSelection, PydanticModel, JsonSchema, Constraint, EbnfGrammar, Regex, Choice
from bridgic.llms.openai_like import OpenAILikeLlm, OpenAILikeConfiguration
from bridgic.core.utils._console import printer
from bridgic.core.utils._collection import validate_required_params, merge_dict, filter_dict

class VllmServerConfiguration(OpenAILikeConfiguration):
    """
    Configuration for the vLLM server.
    """
    pass

class VllmServerLlm(OpenAILikeLlm, StructuredOutput, ToolSelection):
    """
    VllmServerLlm is a wrapper around the vLLM server, providing common calling interfaces for 
    self-hosted LLM service, such as chat, stream, as well as with encapsulation of common 
    seen high-level functionality.

    Parameters
    ----------
    api_base: str
        The base URL of the LLM provider.
    api_key: str
        The API key of the LLM provider.
    timeout: Optional[float]
        The timeout in seconds.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        configuration: Optional[VllmServerConfiguration] = VllmServerConfiguration(),
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            configuration=configuration,
            timeout=timeout,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return super().dump_to_dict()

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)

    @overload
    def structured_output(
        self,
        messages: List[Message],
        constraint: PydanticModel,
        model: Optional[str] = None,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        presence_penalty: Optional[float] = ...,
        frequency_penalty: Optional[float] = ...,
        extra_body: Optional[Dict[str, Any]] = ...,
        **kwargs,
    ) -> BaseModel: ...

    @overload
    def structured_output(
        self,
        messages: List[Message],
        constraint: JsonSchema,
        model: Optional[str] = None,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        presence_penalty: Optional[float] = ...,
        frequency_penalty: Optional[float] = ...,
        extra_body: Optional[Dict[str, Any]] = ...,
        **kwargs,
    ) -> Dict[str, Any]: ...

    @overload
    def structured_output(
        self,
        messages: List[Message],
        constraint: Choice,
        model: Optional[str] = None,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        presence_penalty: Optional[float] = ...,
        frequency_penalty: Optional[float] = ...,
        extra_body: Optional[Dict[str, Any]] = ...,
        **kwargs,
    ) -> str: ...

    def structured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[BaseModel, Dict[str, Any], str]:
        '''
        Structured output in a specified format. This part of the functionality is provided based on the 
        capabilities of [vLLM Structured Output](https://docs.vllm.ai/en/latest/features/structured_outputs.html).

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        constraint: Constraint
            The constraint to use for the structured output.
        model: Optional[str]
            The model to use for the structured output.
        temperature: Optional[float]
            The temperature to use for the structured output.
        top_p: Optional[float]
            The top_p to use for the structured output.
        presence_penalty: Optional[float]
            The presence_penalty to use for the structured output.
        frequency_penalty: Optional[float]
            The frequency_penalty to use for the structured output.
        extra_body: Optional[Dict[str, Any]]
            The extra_body to use for the structured output.
        **kwargs: Any
            The kwargs to use for the structured output.

        Returns
        -------
        Union[BaseModel, Dict[str, Any], str]
            The return type is based on the constraint type:
            * If the constraint is PydanticModel, return an instance of the corresponding Pydantic model.
            * If the constraint is JsonSchema, return a Dict[str, Any] that is the parsed JSON.
            * Otherwise, return a str.
        '''
        response = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=self._convert_constraint(constraint, extra_body),
            **kwargs,
        )
        return self._convert_response(constraint, response)

    async def astructured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[BaseModel, Dict[str, Any], str]:
        '''
        Structured output in a specified format. This part of the functionality is provided based on the 
        capabilities of [vLLM Structured Output](https://docs.vllm.ai/en/latest/features/structured_outputs.html).

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        constraint: Constraint
            The constraint to use for the structured output.
        model: Optional[str]
            The model to use for the structured output.
        temperature: Optional[float]
            The temperature to use for the structured output.
        top_p: Optional[float]
            The top_p to use for the structured output.
        presence_penalty: Optional[float]
            The presence_penalty to use for the structured output.
        frequency_penalty: Optional[float]
            The frequency_penalty to use for the structured output.
        extra_body: Optional[Dict[str, Any]]
            The extra_body to use for the structured output.
        **kwargs: Any
            The kwargs to use for the structured output.

        Returns
        -------
        Union[BaseModel, Dict[str, Any], str]
            The return type is based on the constraint type:
            * If the constraint is PydanticModel, return an instance of the corresponding Pydantic model.
            * If the constraint is JsonSchema, return a Dict[str, Any] that is the parsed JSON.
            * Otherwise, return a str.
        '''
        response = await self.achat(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=self._convert_constraint(constraint, extra_body),
            **kwargs,
        )
        return self._convert_response(constraint, response)

    def _convert_constraint(
        self,
        constraint: Constraint,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        extra_body = {} if extra_body is None else extra_body

        if isinstance(constraint, PydanticModel):
            extra_body["guided_json"] = constraint.model.model_json_schema()
        elif isinstance(constraint, JsonSchema):
            extra_body["guided_json"] = constraint.schema_dict
        elif isinstance(constraint, Regex):
            extra_body["guided_regex"] = constraint.pattern
        elif isinstance(constraint, Choice):
            extra_body["guided_choice"] = constraint.choices
        elif isinstance(constraint, EbnfGrammar):
            extra_body["guided_grammar"] = constraint.syntax
        else:
            raise ValueError(f"Invalid constraint: {constraint}")

        return extra_body

    def _convert_response(
        self,
        constraint: Constraint,
        response: Response,
    ) -> Union[BaseModel, Dict[str, Any], str]:
        content = response.message.content

        if isinstance(constraint, PydanticModel):
            return constraint.model.model_validate_json(content)
        elif isinstance(constraint, JsonSchema):
            return json.loads(content)
        return content

    def select_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        tool_choice: Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam] = "auto",
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[Dict]]:
        """
        Select tools from a specified list of tools.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: List[Tool]
            The tools to use for the tool select.
        model: Optional[str]
            The model to use for the tool select.
        temperature: Optional[float]
            The temperature to use for the tool select.
        top_p: Optional[float]
            The top_p to use for the tool select.
        presence_penalty: Optional[float]
            The presence_penalty to use for the tool select.
        frequency_penalty: Optional[float]
            The frequency_penalty to use for the tool select.
        extra_body: Optional[Dict[str, Any]]
            The extra_body to use for the tool select.
        tool_choice : Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam]
            Tool choice mode for tool calling. There are 4 choices that are supported:
            - `auto` means the model can pick between generating a message or calling one or more tools.
            To enable this feature, you should set the tags `--enable-auto-tool-choice` and `--tool-call-parser` 
            when starting the vLLM server.
            - `required` means the model must generate one or more tool calls based on the specified tool list 
            in the `tools` parameter. The number of tool calls depends on the user's query.
            - `none` means the model will not call any tool and instead generates a message. When tools are 
            specified in the request, vLLM includes tool definitions in the prompt by default, regardless 
            of the tool_choice setting. To exclude tool definitions when tool_choice='none', use the 
            `--exclude-tools-when-tool-choice-none` option when starting the vLLM server.
            - You can also specify a particular function using named function calling by setting `tool_choice` 
            parameter to a json object, like `tool_choice={"type": "function", "function": {"name": "get_weather"}}`.

        **kwargs: Any
            The kwargs to use for the tool select.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            A list that contains the selected tools and their arguments.

        Notes
        -----
        See more on [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling.html).
        """
        # Build parameters dictionary for validation
        params = filter_dict(merge_dict(self.configuration.model_dump(), {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "extra_body": extra_body,
            **kwargs,
        }))
        
        # Validate required parameters for tool selection
        validate_required_params(params, ["model"])
        
        input_messages = [self._convert_message(message=msg, strict=True) for msg in messages]
        input_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            } for tool in tools
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=input_messages,
            tools=input_tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        tool_calls = response.choices[0].message.tool_calls

        output_content = ""
        if response.choices[0].message.content:
            output_content = response.choices[0].message.content

        output_tool_calls = []
        if tool_calls:
            output_tool_calls = self._convert_tool_calls(tool_calls)

        return (output_tool_calls, output_content)

    async def aselect_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        tool_choice: Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam] = "auto",
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Select tools from a specified list of tools.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: List[Tool]
            The tools to use for the tool select.
        model: Optional[str]
            The model to use for the tool select.
        temperature: Optional[float]
            The temperature to use for the tool select.
        top_p: Optional[float]
            The top_p to use for the tool select.
        presence_penalty: Optional[float]
            The presence_penalty to use for the tool select.
        frequency_penalty: Optional[float]
            The frequency_penalty to use for the tool select.
        extra_body: Optional[Dict[str, Any]]
            The extra_body to use for the tool select.
        tool_choice : Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam]
            Tool choice mode for tool calling. There are 4 choices that are supported:
            - `auto` means the model can pick between generating a message or calling one or more tools.
            To enable this feature, you should set the tags `--enable-auto-tool-choice` and `--tool-call-parser` 
            when starting the vLLM server.
            - `required` means the model must generate one or more tool calls based on the specified tool list 
            in the `tools` parameter. The number of tool calls depends on the user's query.
            - `none` means the model will not call any tool and instead generates a message. When tools are 
            specified in the request, vLLM includes tool definitions in the prompt by default, regardless 
            of the tool_choice setting. To exclude tool definitions when tool_choice='none', use the 
            `--exclude-tools-when-tool-choice-none` option when starting the vLLM server.
            - You can also specify a particular function using named function calling by setting `tool_choice` 
            parameter to a json object, like `tool_choice={"type": "function", "function": {"name": "get_weather"}}`.

        **kwargs: Any
            The kwargs to use for the tool select.

        Returns
        -------
        Tuple[List[ToolCall], Optional[str]]
            A list that contains the selected tools and their arguments.

        Notes
        -----
        See more on [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling.html).
        """
        # Build parameters dictionary for validation
        params = filter_dict(merge_dict(self.configuration.model_dump(), {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "extra_body": extra_body,
            **kwargs,
        }))
        
        # Validate required parameters for tool selection
        validate_required_params(params, ["model"])
        
        input_messages = [self._convert_message(message=msg, strict=True) for msg in messages]
        input_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            } for tool in tools
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=input_messages,
            tools=input_tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        tool_calls = response.choices[0].message.tool_calls

        output_content = ""
        if response.choices[0].message.content:
            output_content = response.choices[0].message.content

        output_tool_calls = []
        if tool_calls:
            output_tool_calls = self._convert_tool_calls(tool_calls)

        return (output_tool_calls, output_content)

    def _convert_tool_calls(self, tool_calls: List[ChatCompletionMessageFunctionToolCall]) -> List[ToolCall]:
        return [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            ) for tool_call in tool_calls
        ]

