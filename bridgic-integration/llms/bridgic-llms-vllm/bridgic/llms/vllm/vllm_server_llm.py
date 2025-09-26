import httpx
import warnings
import json
import uuid
import copy

from typing import List
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

class VllmServerLlm(OpenAILikeLlm, StructuredOutput, ToolSelect):
    """
    OpenAILikeLlm is a thin wrapper around the LLM providers that makes it compatible with the 
    services that provide OpenAI compatible API. To support the widest range of model providers, 
    this wrapper only supports text-modal usage.

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
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(
            api_base=api_base,
            api_key=api_key,
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
        response = self.client.chat.completions.create(
            messages=msgs,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )
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
        response = self.client.chat.completions.create(
            messages=msgs,
            model=model,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )
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
        response = await self.async_client.chat.completions.create(
            messages=msgs,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )
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
        response = await self.async_client.chat.completions.create(
            messages=msgs,
            model=model,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            **kwargs,
        )
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
            extra_body["guided_json"] = constraint.schema
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

    def tool_select(
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
        schema = self._convert_tool_select_schema(tools, min_tools, max_tools)

        response: Dict[str, Any] = self.structured_output(
            model=model,
            constraint=JsonSchema(schema=schema, name=""),
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

    async def atool_select(
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
        schema = self._convert_tool_select_schema(tools, min_tools, max_tools)

        response: Dict[str, Any] = await self.astructured_output(
            model=model,
            constraint=JsonSchema(schema=schema, name=""),
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

    def _convert_tool_select_schema(
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
