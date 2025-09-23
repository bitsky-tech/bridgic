import httpx
import warnings
import json
import uuid
import copy

from typing import List
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

class OpenAILlm(OpenAILikeLlm, StructuredOutput, ToolSelect):
    """
    Wrapper class for OpenAI, providing common chat and stream calling interfaces for OpenAI model
    and implementing the common protocols in the Bridgic framework.

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
        Structured output in a specified format.

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
        # TODO
        pass

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
        Structured output in a specified format.

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
        # TODO
        pass

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
        parallel_tool_calls: Optional[bool] = True,
        tool_choice: Optional[Literal["auto", "required", "none"]] = "auto",
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
        # TODO
        pass

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
        parallel_tool_calls: Optional[bool] = True,
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto",
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
        # TODO
        pass