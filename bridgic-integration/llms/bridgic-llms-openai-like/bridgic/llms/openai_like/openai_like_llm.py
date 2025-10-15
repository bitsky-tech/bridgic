import httpx
import warnings
import json

from typing import List, Dict, Any, Optional
from typing_extensions import override
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.resources.chat.completions.completions import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_message_function_tool_call_param import ChatCompletionMessageFunctionToolCallParam
from pydantic import BaseModel

from bridgic.core.model.types import *
from bridgic.core.model.types import *
from bridgic.core.utils.collection import filter_dict, merge_dict, validate_required_params

class OpenAILikeConfiguration(BaseModel):
    """
    Default configuration for OpenAI-compatible chat completions.

    Parameters
    ----
    model : str, optional
        Default model to use when a call-time `model` is not provided.
    temperature : float, optional
        Sampling temperature in [0, 2]. Higher is more random, lower is more deterministic.
    top_p : float, optional
        Nucleus sampling probability mass in (0, 1]. Alternative to temperature.
    presence_penalty : float, optional
        Penalize new tokens based on whether they appear so far. [-2.0, 2.0].
    frequency_penalty : float, optional
        Penalize new tokens based on their frequency so far. [-2.0, 2.0].
    max_tokens : int, optional
        Maximum number of tokens to generate for the completion.
    stop : list[str], optional
        Up to 4 sequences where generation will stop.
    """
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


class OpenAILikeLlm(BaseLlm):
    """
    OpenAILikeLlm is a thin wrapper around the LLM providers that makes it compatible with the 
    services that provide OpenAI compatible API. To support the widest range of model providers, 
    this wrapper only supports text-modal usage.

    Parameters
    ----
    api_base: str
        The base URL of the LLM provider.
    api_key: str
        The API key of the LLM provider.
    timeout: Optional[float]
        The timeout in seconds.
    """

    api_base: str
    api_key: str
    configuration: OpenAILikeConfiguration
    timeout: float
    http_client: httpx.Client
    http_async_client: httpx.AsyncClient

    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        api_base: str,
        api_key: str,
        configuration: Optional[OpenAILikeConfiguration] = OpenAILikeConfiguration(),
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
    ):
        # Record for serialization / deserialization.
        self.api_base = api_base
        self.api_key = api_key
        self.configuration = configuration
        self.timeout = timeout
        self.http_client = http_client
        self.http_async_client = http_async_client

        # Initialize clients.
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout, http_client=http_client)
        self.async_client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout, http_client=http_async_client)

    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Response:
        """
        Send a synchronous chat completion request to an OpenAI-compatible provider.

        Parameters
        ----
        messages : list[Message]
            Conversation messages.
        model : str, optional
            Model ID to use. Required unless provided in `configuration.model`.
        temperature : float, optional
            Sampling temperature in [0, 2]. Higher is more random, lower is more deterministic.
        top_p : float, optional
            Nucleus sampling probability mass in (0, 1]. Alternative to temperature.
        presence_penalty : float, optional
            Penalize new tokens based on whether they appear so far. [-2.0, 2.0].
        frequency_penalty : float, optional
            Penalize new tokens based on their frequency so far. [-2.0, 2.0].
        max_tokens : int, optional
            Maximum tokens to generate for completion.
        stop : list[str], optional
            Up to 4 sequences where generation will stop.
        extra_body : dict, optional
            Extra JSON payload sent to the provider.
        **kwargs
            Additional provider-specific arguments.

        Returns
        ----
        Response
            Bridgic response containing the generated message and raw API response.

        Notes
        ----
        - Required parameter validation ensures `messages` and final `model` are present
          (from either the call or `configuration`).
        """
        params = self._build_parameters(
            messages=messages,
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
        validate_required_params(params, ["messages", "model"])
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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> StreamResponse:
        """
        Stream a chat completion response incrementally.

        Parameters
        ----
        messages : list[Message]
            Conversation messages.
        model : str, optional
            Model ID to use. Required unless provided in `configuration.model`.
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop, extra_body
            See `chat` for details.
        **kwargs
            Additional provider-specific arguments.

        Yields
        ----
        MessageChunk
            Delta chunks as they arrive from the provider.

        Notes
        ----
        - Validates `messages`, final `model`, and `stream=True`.
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            stream=True,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model", "stream"])
        response = self.client.chat.completions.create(**params)
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            delta_content = delta_content if delta_content else ""
            yield MessageChunk(delta=delta_content, raw=chunk)

    async def achat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Response:
        """
        Asynchronously send a chat completion request to an OpenAI-compatible provider.

        Parameters
        ----
        messages : list[Message]
            Conversation messages.
        model : str, optional
            Model ID to use. Required unless provided in `configuration.model`.
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop, extra_body
            See `chat` for details.
        **kwargs
            Additional provider-specific arguments.

        Returns
        ----
        Response
            Bridgic response containing the generated message and raw API response.
        """
        params = self._build_parameters(
            messages=messages,
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
        validate_required_params(params, ["messages", "model"])
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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncStreamResponse:
        """
        Asynchronously stream a chat completion response incrementally.

        Parameters
        ----
        messages : list[Message]
            Conversation messages.
        model : str, optional
            Model ID to use. Required unless provided in `configuration.model`.
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop, extra_body
            See `chat` for details.
        **kwargs
            Additional provider-specific arguments.

        Yields
        ----
        MessageChunk
            Delta chunks as they arrive from the provider.

        Notes
        ----
        - Validates `messages`, final `model`, and `stream=True`.
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
            stream=True,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model", "stream"])
        response = await self.async_client.chat.completions.create(**params)
        async for chunk in response:
            delta_content = chunk.choices[0].delta.content
            delta_content = delta_content if delta_content else ""
            yield MessageChunk(delta=delta_content, raw=chunk)

    def _build_parameters(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge configuration defaults with per-call parameters and remove None values.

        Parameters
        ----
        messages : list[Message]
            Conversation messages to send.
        model : str, optional
            Model identifier. May be omitted if `configuration.model` is set.
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop, extra_body, stream
            Standard OpenAI chat parameters.
        **kwargs
            Additional provider-specific parameters.

        Returns
        ----
        dict
            Final parameter dictionary for the OpenAI-compatible API.
        """
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        merge_params = merge_dict(self.configuration.model_dump(), {
            "messages": msgs,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "extra_body": extra_body,
            "stream": stream,
            **kwargs,
        })
        return filter_dict(merge_params, exclude_none=True)

    def _convert_message(self, message: Message, strict: bool = False) -> ChatCompletionMessageParam:
        if strict:
            return self._convert_message_strict(message)
        else:
            return self._convert_message_normal(message)

    def _convert_message_normal(self, message: Message) -> ChatCompletionMessageParam:
        content_list = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                content_list.append(block.text)
            if isinstance(block, ToolCallBlock):
                content_list.append(
                    f"Tool call:\n"
                    f"- id: {block.id}\n"
                    f"- name: {block.name}\n"
                    f"- arguments: {block.arguments}"
                )
            if isinstance(block, ToolResultBlock):
                content_list.append(f"Tool result: {block.content}")
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

    def _convert_message_strict(self, message: Message) -> ChatCompletionMessageParam:
        content_list = []
        tool_call_list = []
        tool_result = ""
        tool_result_call_id = None

        for block in message.blocks:
            if isinstance(block, TextBlock):
                content_list.append(block.text)
            if isinstance(block, ToolCallBlock):
                tool_call: ChatCompletionMessageFunctionToolCallParam = {
                    "type": "function",
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    },
                }
                tool_call_list.append(tool_call)
            if isinstance(block, ToolResultBlock):
                tool_result = block.content
                tool_result_call_id = block.id

        content_txt = "\n\n".join(content_list)

        if message.role == Role.SYSTEM:
            return ChatCompletionSystemMessageParam(content=content_txt, role="system")
        elif message.role == Role.USER:
            return ChatCompletionUserMessageParam(content=content_txt, role="user")
        elif message.role == Role.AI:
            return ChatCompletionAssistantMessageParam(content=content_txt, tool_calls=tool_call_list, role="assistant")
        elif message.role == Role.TOOL:
            content_txt = "\n\n".join([content_txt, tool_result])
            return ChatCompletionToolMessageParam(content=content_txt, tool_call_id=tool_result_call_id, role="tool")
        else:
            raise ValueError(f"Invalid role: {message.role}")

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "configuration": self.configuration.model_dump(),
        }
        if self.http_client:
            warnings.warn(
                "httpx.Client is not serializable, so it will be set to None in the deserialization.",
                RuntimeWarning,
            )
        if self.http_async_client:
            warnings.warn(
                "httpx.AsyncClient is not serializable, so it will be set to None in the deserialization.",
                RuntimeWarning,
            )
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.api_base = state_dict["api_base"]
        self.api_key = state_dict["api_key"]
        self.timeout = state_dict["timeout"]
        self.configuration = OpenAILikeConfiguration(**state_dict.get("configuration", {}))

        self.http_client = None
        self.http_async_client = None

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=self.http_client,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            http_client=self.http_async_client,
        )
