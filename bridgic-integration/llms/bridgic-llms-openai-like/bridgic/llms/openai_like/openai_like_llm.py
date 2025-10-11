import httpx
import warnings

from typing import List, Dict, Any, Optional
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
from bridgic.core.utils.collection import filter_dict

class OpenAILikeConfiguration(BaseModel):
    """
    Configuration class for OpenAI-like LLM providers.
    
    This class defines the default parameters that can be set for OpenAI-compatible
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
    ----------
    api_base: str
        The base URL of the LLM provider.
    api_key: str
        The API key of the LLM provider.
    configuration: Optional[OpenAILikeConfiguration]
        The configuration for the LLM provider. If None, uses the default configuration.
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
            **self.configuration.model_dump(),
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
            **self.configuration.model_dump(),
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
            **self.configuration.model_dump(),
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
            **self.configuration.model_dump(),
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
        self.configuration = OpenAILikeConfiguration(**state_dict["configuration"])

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
