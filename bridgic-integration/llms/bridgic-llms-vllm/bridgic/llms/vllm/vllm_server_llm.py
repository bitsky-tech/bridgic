import httpx
import warnings
import json
import uuid
import copy

from typing import List, Optional
from typing_extensions import override
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.resources.chat.completions.completions import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam

from bridgic.core.intelligence.base_llm import *
from bridgic.core.intelligence.content import *
from bridgic.core.intelligence.protocol import *
from bridgic.llms.openai_like.openai_like_llm import OpenAILikeLlm
from bridgic.core.utils.console import printer
from bridgic.core.utils.collection import filter_dict

class VllmServerConfiguration(BaseModel):
    """
    Configuration class for vLLM server LLM providers.
    
    This class defines the default parameters that can be set for vLLM server
    LLM providers, allowing for consistent configuration across different providers.
    
    Parameters
    ----------
    model : Optional[str]
        The model identifier to use by default.
    temperature : Optional[float]
        Sampling temperature between 0 and 2. Higher values make output more random.
    top_p : Optional[float]
        Nucleus sampling parameter. Alternative to temperature.
    presence_penalty : Optional[float]
        Penalty for new tokens based on whether they appear in the text so far.
    frequency_penalty : Optional[float]
        Penalty for new tokens based on their frequency in the text so far.
    max_tokens : Optional[int]
        Maximum number of tokens to generate.
    stop : Optional[List[str]]
        List of sequences where the API will stop generating further tokens.
    min_tools : Optional[int]
        Minimum number of tools to select in tool selection.
    max_tools : Optional[int]
        Maximum number of tools to select in tool selection.
    """
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    min_tools: Optional[int] = None
    max_tools: Optional[int] = None

class VllmServerLlm(OpenAILikeLlm, StructuredOutput, ToolSelection):
    """
    VllmServerLlm is a wrapper around vLLM server that provides OpenAI-compatible API.
    It extends OpenAILikeLlm with additional vLLM-specific features like structured output
    and tool selection capabilities.

    Parameters
    ----------
    api_base: str
        The base URL of the vLLM server.
    api_key: str
        The API key for the vLLM server.
    configuration: Optional[VllmServerConfiguration]
        The configuration for the vLLM server. If None, uses the default configuration.
    timeout: Optional[float]
        The timeout in seconds.
    http_client: Optional[httpx.Client]
        Custom synchronous HTTP client for requests. If None, creates a default client.
    http_async_client: Optional[httpx.AsyncClient]
        Custom asynchronous HTTP client for requests. If None, creates a default client.

    Attributes
    ----------
    client : openai.OpenAI
        The synchronous OpenAI client instance.
    async_client : openai.AsyncOpenAI
        The asynchronous OpenAI client instance.
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
        # Store vLLM-specific configuration
        self.vllm_configuration = configuration
        
        # Initialize parent with OpenAILikeConfiguration
        from bridgic.llms.openai_like.openai_like_llm import OpenAILikeConfiguration
        openai_like_config = OpenAILikeConfiguration(
            model=configuration.model,
            temperature=configuration.temperature,
            top_p=configuration.top_p,
            presence_penalty=configuration.presence_penalty,
            frequency_penalty=configuration.frequency_penalty,
            max_tokens=configuration.max_tokens,
            stop=configuration.stop,
        )
        
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            configuration=openai_like_config,
            timeout=timeout,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["vllm_configuration"] = self.vllm_configuration.model_dump()
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.vllm_configuration = VllmServerConfiguration(**state_dict["vllm_configuration"])

    def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Response:
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        
        # Build parameters dictionary and filter out None values
        # The priority order is as follows: configuration passed through the interface > configuration of the instance itself.
        params = filter_dict({
            **self.vllm_configuration.model_dump(),
            "messages": msgs,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "extra_body": extra_body,
            **kwargs,
        }, exclude_none=True)
        
        response = self.client.chat.completions.create(**params)
        openai_message: ChatCompletionMessage = response.choices[0].message
        text: str = openai_message.content if openai_message.content else ""

        if openai_message.refusal:
            warnings.warn(openai_message.refusal, RuntimeWarning)

        return Response(
            message=Message.from_text(text, role=Role.AI),
            raw=response,
        )

    def stream(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> StreamResponse:
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        
        # Build parameters dictionary and filter out None values
        params = filter_dict({
            **self.vllm_configuration.model_dump(),
            "messages": msgs,
            "model": model,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "extra_body": extra_body,
            **kwargs,
        }, exclude_none=True)
        
        response = self.client.chat.completions.create(**params)
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            delta_content = delta_content if delta_content else ""
            yield MessageChunk(delta=delta_content, raw=chunk)

    async def achat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Response:
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        
        # Build parameters dictionary and filter out None values
        params = filter_dict({
            **self.vllm_configuration.model_dump(),
            "messages": msgs,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "extra_body": extra_body,
            **kwargs,
        }, exclude_none=True)
        
        response = await self.async_client.chat.completions.create(**params)
        openai_message: ChatCompletionMessage = response.choices[0].message
        text: str = openai_message.content if openai_message.content else ""

        if openai_message.refusal:
            warnings.warn(openai_message.refusal, RuntimeWarning)

        return Response(
            message=Message.from_text(text, role=Role.AI),
            raw=response,
        )

    async def astream(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncStreamResponse:
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        
        # Build parameters dictionary and filter out None values
        params = filter_dict({
            **self.vllm_configuration.model_dump(),
            "messages": msgs,
            "model": model,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "extra_body": extra_body,
            **kwargs,
        }, exclude_none=True)
        
        response = await self.async_client.chat.completions.create(**params)
        async for chunk in response:
            delta_content = chunk.choices[0].delta.content
            delta_content = delta_content if delta_content else ""
            yield MessageChunk(delta=delta_content, raw=chunk)

    def _convert_message(self, message: Message) -> ChatCompletionMessageParam:
        content_list = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                content_list.append(block.text)
        content_txt = "\n\n".join(content_list)

        if message.role == Role.SYSTEM:
            return ChatCompletionSystemMessageParam(content=content_txt, role="system")
        elif message.role == Role.USER:
            return ChatCompletionUserMessageParam(content=content_txt, role="user")
        elif message.role == Role.AI:
            return ChatCompletionAssistantMessageParam(content=content_txt, role="assistant")
        elif message.role == Role.TOOL:
            return ChatCompletionToolMessageParam(content=content_txt, role="tool")
        else:
            raise ValueError(f"Invalid role: {message.role}")

    from typing import overload, Literal

    @overload
    def structured_output(
        self,
        messages: List[Message],
        constraint: PydanticModel,
        model: str,
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
        model: str,
        temperature: Optional[float] = ...,
        top_p: Optional[float] = ...,
        presence_penalty: Optional[float] = ...,
        frequency_penalty: Optional[float] = ...,
        extra_body: Optional[Dict[str, Any]] = ...,
        **kwargs,
    ) -> Dict[str, Any]: ...

    def structured_output(
        self,
        messages: List[Message],
        constraint: Constraint,
        model: str,
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
        model: str
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
        model: str,
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
        model: str
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
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        min_tools: Optional[int] = None,
        max_tools: Optional[int] = None,
        **kwargs,
    ) -> List[ToolCall]:
        """
        Select tools from a specified list of tools.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: List[Tool]
            The tools to use for the tool select.
        model: str
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
        min_tools: Optional[int]
            The minimum number of tools to select.
        max_tools: Optional[int]
            The maximum number of tools to select.
        **kwargs: Any
            The kwargs to use for the tool select.

        Returns
        -------
        List[ToolCall]
            A list that contains the selected tools and their arguments.
        """
        # Use configuration defaults if not provided
        min_tools = min_tools if min_tools is not None else self.vllm_configuration.min_tools
        max_tools = max_tools if max_tools is not None else self.vllm_configuration.max_tools
        
        schema = self._convert_select_tool_schema(tools, min_tools, max_tools)

        response: Dict[str, Any] = self.structured_output(
            model=model,
            constraint=JsonSchema(schema_dict=schema, name=""),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=extra_body,
            **kwargs,
        )
        tool_calls = response["tool_calls"]
        return self._convert_tool_calls(tool_calls)

    async def aselect_tool(
        self,
        messages: List[Message],
        tools: List[Tool],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        min_tools: Optional[int] = None,
        max_tools: Optional[int] = None,
        **kwargs,
    ) -> List[ToolCall]:
        """
        Select tools from a specified list of tools.

        Parameters
        ----------
        messages: List[Message]
            The messages to send to the LLM.
        tools: List[Tool]
            The tools to use for the tool select.
        model: str
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
        min_tools: Optional[int]
            The minimum number of tools to select.
        max_tools: Optional[int]
            The maximum number of tools to select.
        **kwargs: Any
            The kwargs to use for the tool select.

        Returns
        -------
        List[ToolCall]
            A list that contains the selected tools and their arguments.
        """
        # Use configuration defaults if not provided
        min_tools = min_tools if min_tools is not None else self.vllm_configuration.min_tools
        max_tools = max_tools if max_tools is not None else self.vllm_configuration.max_tools
        
        schema = self._convert_select_tool_schema(tools, min_tools, max_tools)

        response: Dict[str, Any] = await self.astructured_output(
            model=model,
            constraint=JsonSchema(schema_dict=schema, name=""),
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=extra_body,
            **kwargs,
        )
        tool_calls = response["tool_calls"]
        return self._convert_tool_calls(tool_calls)

    def _convert_select_tool_schema(
        self,
        tools: List[Tool] = [],
        min_tools: Optional[int] = None,
        max_tools: Optional[int] = None,
    ) -> Dict[str, Any]:
        if min_tools is not None and min_tools < 0:
            raise ValueError("min_tools must be greater than or equal to 0.")
        if max_tools is not None and max_tools < 0:
            raise ValueError("max_tools must be greater than or equal to 0.")
        if min_tools is not None and max_tools is not None and min_tools > max_tools:
            raise ValueError("min_tools must be less than or equal to max_tools.")

        schema = {
            "$defs": {},
            "properties": {
                "tool_calls": {
                    "items": {
                        "anyOf": []
                    },
                    "type": "array",
                    "title": "ToolCallList",
                },
            },
        }

        if min_tools is not None:
            schema["properties"]["tool_calls"]["minItems"] = min_tools
        if max_tools is not None:
            schema["properties"]["tool_calls"]["maxItems"] = max_tools

        for tool in tools:
            if tool.name in schema["$defs"]:
                raise ValueError(f"Tool name {tool.name} is duplicated.")

            schema["$defs"][tool.name] = {
                "type": "object",
                "title": tool.name,
                "properties": {
                    "name": {
                        "type": "string",
                        "const": tool.name,
                        "title": "Name",
                    },
                    "arguments": tool.parameters,
                },
                "required": ["name", "arguments"],
            }
            schema["properties"]["tool_calls"]["items"]["anyOf"].append({
                "$ref": f"#/$defs/{tool.name}",
            })

        return schema

    def _convert_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        return [
            ToolCall(
                id=str(uuid.uuid4()),
                name=tool_call["name"],
                arguments=tool_call["arguments"],
            ) for tool_call in tool_calls
        ]
