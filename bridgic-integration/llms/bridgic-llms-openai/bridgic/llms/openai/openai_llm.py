import json
import httpx
import warnings

from typing import List, overload, Literal
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageFunctionToolCall
from pydantic import BaseModel
from openai import Stream
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
    OpenAI LLM client implementing Bridgic's intelligence protocols.

    This class provides a comprehensive interface to OpenAI's language models,
    supporting chat completions, streaming, structured outputs, and tool calling
    within the Bridgic framework. It inherits from OpenAILikeLlm and implements
    the StructuredOutput and ToolSelect protocols for enhanced functionality.

    Parameters
    ----------
    api_key : str
        The API key for OpenAI services. Required for authentication.
    api_base : Optional[str], default=None
        The base URL for the OpenAI API. If None, uses the default OpenAI endpoint.
    timeout : Optional[float], default=None
        Request timeout in seconds. If None, no timeout is applied.
    http_client : Optional[httpx.Client], default=None
        Custom synchronous HTTP client for requests. If None, creates a default client.
    http_async_client : Optional[httpx.AsyncClient], default=None
        Custom asynchronous HTTP client for requests. If None, creates a default client.

    Attributes
    ----------
    client : openai.OpenAI
        The synchronous OpenAI client instance.
    async_client : openai.AsyncOpenAI
        The asynchronous OpenAI client instance.

    Examples
    --------
    Basic usage for chat completion:

    >>> llm = OpenAILlm(api_key="your-api-key")
    >>> messages = [Message.from_text("Hello!", role=Role.USER)]
    >>> response = llm.chat(messages=messages, model="gpt-4")
    >>> print(response.message.content)

    Structured output with Pydantic model:

    >>> class Answer(BaseModel):
    ...     reasoning: str
    ...     result: int
    >>> constraint = PydanticModel(model=Answer)
    >>> structured_response = llm.structured_output(
    ...     messages=messages,
    ...     constraint=constraint,
    ...     model="gpt-4"
    ... )

    Tool calling:

    >>> tools = [Tool(name="calculator", description="Calculate math", parameters={})]
    >>> tool_calls = llm.tool_select(messages=messages, tools=tools, model="gpt-4")

    Notes
    -----
    - Supports both synchronous and asynchronous operations
    - Handles OpenAI API refusals by issuing warnings
    - Automatically converts between Bridgic and OpenAI message formats
    - Implements structured output with strict JSON schema validation
    - Provides comprehensive tool calling capabilities
    """

    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
        http_async_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize the OpenAI LLM client with configuration parameters.

        Parameters
        ----------
        api_key : str
            The API key for OpenAI services. Required for authentication.
        api_base : Optional[str], default=None
            The base URL for the OpenAI API. If None, uses the default OpenAI endpoint.
        timeout : Optional[float], default=None
            Request timeout in seconds. If None, no timeout is applied.
        http_client : Optional[httpx.Client], default=None
            Custom synchronous HTTP client for requests. If None, creates a default client.
        http_async_client : Optional[httpx.AsyncClient], default=None
            Custom asynchronous HTTP client for requests. If None, creates a default client.

        Notes
        -----
        This class inherits from OpenAILikeLlm and implements StructuredOutput and 
        ToolSelect protocols for enhanced functionality.
        """
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
        """
        Send a synchronous chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int], default=None
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]], default=None
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Response
            A response object containing the generated message and raw API response.

        Notes
        -----
        This method converts Bridgic Message objects to OpenAI format and handles
        potential refusal responses by issuing warnings.
        """
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response: ChatCompletion = self.client.chat.completions.create(
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
        openai_message = response.choices[0].message
        text = openai_message.content if openai_message.content else ""
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
        """
        Send a streaming chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int], default=None
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]], default=None
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Yields
        ------
        MessageChunk
            Individual chunks of the response as they are received from the API.
            Each chunk contains a delta (partial content) and the raw response.

        Notes
        -----
        This method enables real-time streaming of the model's response,
        useful for providing incremental updates to users as the response is generated.
        """
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
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
        """
        Send an asynchronous chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int], default=None
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]], default=None
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Response
            A response object containing the generated message and raw API response.

        Notes
        -----
        This is the asynchronous version of the chat method, suitable for
        concurrent processing and non-blocking I/O operations.
        """
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
        """
        Send an asynchronous streaming chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int], default=None
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]], default=None
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Yields
        ------
        MessageChunk
            Individual chunks of the response as they are received from the API.
            Each chunk contains a delta (partial content) and the raw response.

        Notes
        -----
        This is the asynchronous version of the stream method, suitable for
        concurrent processing and non-blocking streaming operations.
        """
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
        """
        Convert a Bridgic Message object to OpenAI ChatCompletionMessageParam format.

        Parameters
        ----------
        message : Message
            The Bridgic message object containing role, content blocks, and metadata.

        Returns
        -------
        ChatCompletionMessageParam
            A message parameter object compatible with OpenAI's chat completion API.

        Raises
        ------
        ValueError
            If the message role is not supported (SYSTEM, USER, AI, or TOOL).

        Notes
        -----
        This method extracts text blocks from the message and combines them with
        newlines. Only TextBlock instances are processed; other block types are ignored.
        """
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
        """
        Generate structured output in a specified format using OpenAI's structured output API.

        This method leverages OpenAI's structured output capabilities to ensure the model
        response conforms to a specified schema. Recommended for use with GPT-4o and later models.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        constraint : Constraint
            The constraint defining the desired output format (PydanticModel or JsonSchema).
        model : str
            Model ID used to generate the response. Structured outputs work best with GPT-4o and later.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Union[BaseModel, Dict[str, Any], str]
            The structured response in the format specified by the constraint:
            - BaseModel instance if constraint is PydanticModel
            - Dict[str, Any] if constraint is JsonSchema  
            - str for other constraint types

        Examples
        --------
        Using a Pydantic model constraint:

        >>> class Answer(BaseModel):
        ...     reasoning: str
        ...     result: int
        >>> constraint = PydanticModel(model=Answer)
        >>> response = llm.structured_output(
        ...     messages=[Message.from_text("What is 2+2?", role=Role.USER)],
        ...     constraint=constraint,
        ...     model="gpt-4o"
        ... )
        >>> print(response.reasoning, response.result)

        Using a JSON schema constraint:

        >>> schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        >>> constraint = JsonSchema(schema=schema)
        >>> response = llm.structured_output(
        ...     messages=[Message.from_text("Hello", role=Role.USER)],
        ...     constraint=constraint,
        ...     model="gpt-4o"
        ... )
        >>> print(response["answer"])

        Notes
        -----
        - Utilizes OpenAI's native structured output API with strict schema validation
        - All schemas automatically have additionalProperties set to False
        - Best performance achieved with GPT-4o and later models (gpt-4o-mini, gpt-4o-2024-08-06, and later)
        """
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response = self.client.chat.completions.parse(
            messages=msgs,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=self._get_response_format(constraint),
            **kwargs,
        )
        return self._convert_response(constraint, response.choices[0].message)

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
        """
        Asynchronously generate structured output in a specified format using OpenAI's API.

        This is the asynchronous version of structured_output, suitable for concurrent
        processing and non-blocking operations. It leverages OpenAI's structured output
        capabilities to ensure the model response conforms to a specified schema.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        constraint : Constraint
            The constraint defining the desired output format (PydanticModel or JsonSchema).
        model : str
            Model ID used to generate the response. Structured outputs work best with GPT-4o and later.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Union[BaseModel, Dict[str, Any], str]
            The structured response in the format specified by the constraint:
            - BaseModel instance if constraint is PydanticModel
            - Dict[str, Any] if constraint is JsonSchema
            - str for other constraint types

        Examples
        --------
        Using asynchronous structured output:

        >>> async def get_structured_response():
        ...     llm = OpenAILlm(api_key="your-key")
        ...     constraint = PydanticModel(model=Answer)
        ...     response = await llm.astructured_output(
        ...         messages=[Message.from_text("Calculate 5+3", role=Role.USER)],
        ...         constraint=constraint,
        ...         model="gpt-4o"
        ...     )
        ...     return response

        Notes
        -----
        - This is the asynchronous version of structured_output
        - Utilizes OpenAI's native structured output API with strict schema validation
        - Suitable for concurrent processing and high-throughput applications
        - Best performance achieved with GPT-4o and later models (gpt-4o-mini, gpt-4o-2024-08-06, and later)
        """
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response = await self.async_client.chat.completions.parse(
            messages=msgs,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=self._get_response_format(constraint),
            **kwargs,
        )
        return self._convert_response(constraint, response.choices[0].message)

    def _add_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add required properties to JSON schema for OpenAI structured outputs.

        Parameters
        ----------
        schema : Dict[str, Any]
            The original JSON schema dictionary.

        Returns
        -------
        Dict[str, Any]
            The modified schema with additionalProperties set to False.

        Notes
        -----
        OpenAI requires additionalProperties to be set to False for all objects
        in structured output schemas. See:
        [AdditionalProperties False Must Always Be Set in Objects](https://platform.openai.com/docs/guides/structured-outputs?example=moderation#additionalproperties-false-must-always-be-set-in-objects)
        """
        schema["additionalProperties"] = False
        return schema

    def _get_response_format(self, constraint: Constraint) -> Dict[str, Any]:
        """
        Convert a Bridgic constraint to OpenAI response format specification.

        Parameters
        ----------
        constraint : Constraint
            The constraint object specifying the desired output format.
            Can be PydanticModel or JsonSchema.

        Returns
        -------
        Dict[str, Any]
            OpenAI-compatible response format specification with JSON schema.

        Raises
        ------
        ValueError
            If the constraint type is not supported (not PydanticModel or JsonSchema).

        Notes
        -----
        This method converts Bridgic constraint objects to the format expected
        by OpenAI's structured output API. All schemas have strict validation enabled.
        """
        if isinstance(constraint, PydanticModel):
            result = {
                "type": "json_schema",
                "json_schema": {
                    "schema": self._add_schema_properties(constraint.model.model_json_schema()),
                    "name": constraint.__class__.__name__,
                    "strict": True,
                },
            }
            return result
        elif isinstance(constraint, JsonSchema):
            return {
                "type": "json_schema",
                "json_schema": {
                    "schema": self._add_schema_properties(constraint.schema),
                    "name": constraint.__class__.__name__,
                    "strict": True,
                },
            }
        else:
            raise ValueError(f"Invalid constraint: {constraint}")
    
    def _convert_response(
        self,
        constraint: Constraint,
        response: ChatCompletionMessage,
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """
        Convert OpenAI response to the appropriate format based on constraint type.

        Parameters
        ----------
        constraint : Constraint
            The constraint object that was used to request the structured output.
        response : ChatCompletionMessage
            The response message from OpenAI containing the structured content.

        Returns
        -------
        Union[BaseModel, Dict[str, Any], str]
            The parsed response in the format specified by the constraint:
            - BaseModel instance if constraint is PydanticModel
            - Dict if constraint is JsonSchema
            - Raw string content for other constraint types

        Notes
        -----
        This method assumes the response content is valid JSON when dealing with
        PydanticModel or JsonSchema constraints.
        """
        content = response.content
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
        parallel_tool_calls: Optional[bool] = True,
        tool_choice: Optional[Literal["auto", "required", "none"]] = "auto",
        **kwargs,
    ) -> List[ToolCall]:
        """
        Select and invoke tools from a list based on conversation context.

        This method enables the model to intelligently select and call appropriate tools
        from a provided list based on the conversation context. It supports OpenAI's
        function calling capabilities with parallel execution and various control options.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far providing context for tool selection.
        tools : List[Tool]
            A list of tools the model may call.
        model : str
            Model ID used to generate the response. Function calling requires compatible models.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        parallel_tool_calls : Optional[bool], default=True
            Whether to enable parallel function calling during tool use.
        tool_choice : Optional[Literal["auto", "required", "none"]], default="auto"
            Controls which (if any) tool is called by the model. `none` means the model will
            not call any tool and instead generates a message. `auto` means the model can
            pick between generating a message or calling one or more tools. `required` means
            the model must call one or more tools.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        List[ToolCall]
            List of selected tool calls with their IDs, names, and parsed arguments.

        Examples
        --------
        >>> tools = [
        ...     Tool(name="get_weather", description="Get weather info", parameters={}),
        ...     Tool(name="calculator", description="Perform calculations", parameters={})
        ... ]
        >>> messages = [Message.from_text("What's the weather in Tokyo?", role=Role.USER)]
        >>> tool_calls = llm.tool_select(messages=messages, tools=tools, model="gpt-4")
        >>> for call in tool_calls:
        ...     print(f"Tool: {call.name}, Args: {call.arguments}")

        Notes
        -----
        - Requires OpenAI models that support function calling
        - Tool calls are automatically assigned the ID from OpenAI
        - Function arguments are parsed from JSON to Python dictionaries

        More OpenAI information: [function-calling](https://platform.openai.com/docs/guides/function-calling)
        """
        tools = self._convert_tools_2_openai_json(tools)
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response: Dict[str, Any] = self.client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_body=extra_body,
            **kwargs,
        )
        tool_calls = response.choices[0].message.tool_calls
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
        parallel_tool_calls: Optional[bool] = True,
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto",
        **kwargs,
    ) -> List[ToolCall]:
        """
        Asynchronously select and invoke tools from a list based on conversation context.

        This is the asynchronous version of tool_select, suitable for concurrent processing
        and non-blocking operations. It enables the model to intelligently select and call
        appropriate tools from a provided list based on the conversation context.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far providing context for tool selection.
        tools : List[Tool]
            A list of tools the model may call.
        model : str
            Model ID used to generate the response. Function calling requires compatible models.
        temperature : Optional[float], default=None
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float], default=None
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float], default=None
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]], default=None
            Add additional JSON properties to the request.
        parallel_tool_calls : Optional[bool], default=True
            Whether to enable parallel function calling during tool use.
        tool_choice : Optional[Literal["auto", "none", "required"]], default="auto"
            Controls which (if any) tool is called by the model. `none` means the model will
            not call any tool and instead generates a message. `auto` means the model can
            pick between generating a message or calling one or more tools. `required` means
            the model must call one or more tools.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        List[ToolCall]
            List of selected tool calls with their IDs, names, and parsed arguments.

        Examples
        --------
        >>> async def select_tools():
        ...     tools = [Tool(name="search", description="Search web", parameters={})]
        ...     messages = [Message.from_text("Search for Python tutorials", role=Role.USER)]
        ...     tool_calls = await llm.atool_select(messages=messages, tools=tools, model="gpt-4")
        ...     return tool_calls

        Notes
        -----
        - This is the asynchronous version of tool_select
        - Suitable for concurrent processing and high-throughput applications
        - Requires OpenAI models that support function calling
        - Tool calls are automatically assigned the ID from OpenAI
        - Function arguments are parsed from JSON to Python dictionaries

        More OpenAI information: [function-calling](https://platform.openai.com/docs/guides/function-calling)
        """
        json_desc_tools = self._convert_tools_2_openai_json(tools)
        msgs: List[ChatCompletionMessageParam] = [self._convert_message(msg) for msg in messages]
        response: Dict[str, Any] = await self.async_client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tools=json_desc_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            extra_body=extra_body,
            **kwargs,
        )
        tool_calls = response.choices[0].message.tool_calls
        return self._convert_tool_calls(tool_calls)
    
    def _convert_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tool parameters to OpenAI function properties format.

        Parameters
        ----------
        parameters : Dict[str, Any]
            The tool parameters dictionary containing properties and required fields.

        Returns
        -------
        Dict[str, Any]
            OpenAI-compatible properties object with type, properties, required fields,
            and additionalProperties set to False.

        Notes
        -----
        This method ensures the parameters conform to OpenAI's function calling schema
        requirements by adding the necessary object structure and disabling additional properties.
        """
        return {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
            "additionalProperties": False
        }
    
    def _convert_tool_2_json(self, tool: Tool) -> Dict[str, Any]:
        """
        Convert a Bridgic Tool object to OpenAI function calling format.

        Parameters
        ----------
        tool : Tool
            The Bridgic tool object containing name, description, and parameters.

        Returns
        -------
        Dict[str, Any]
            OpenAI-compatible function definition with type, name, description, and parameters.

        Notes
        -----
        This method structures the tool information according to OpenAI's function
        calling API specification, wrapping the tool in a "function" type.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": self._convert_parameters(tool.parameters),
            }
        }
    
    def _convert_tools_2_openai_json(
        self,
        tools: List[Tool] = [],
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of Bridgic Tool objects to OpenAI tools format.

        Parameters
        ----------
        tools : List[Tool], default=[]
            List of Bridgic tool objects to convert.

        Returns
        -------
        List[Dict[str, Any]]
            List of OpenAI-compatible tool definitions.

        Notes
        -----
        This method processes each tool through _convert_tool_2_json to ensure
        proper formatting for OpenAI's function calling API.
        """
        tools = [self._convert_tool_2_json(tool) for tool in tools]
        return tools

    def _convert_tool_calls(self, tool_calls: List[ChatCompletionMessageFunctionToolCall]) -> List[ToolCall]:
        """
        Convert OpenAI tool calls to Bridgic ToolCall objects.

        Parameters
        ----------
        tool_calls : List[ChatCompletionMessageFunctionToolCall]
            List of tool calls returned by OpenAI's function calling API.

        Returns
        -------
        List[ToolCall]
            List of Bridgic ToolCall objects with unique IDs, function names, and parsed arguments.

        Notes
        -----
        Each tool call is assigned the ID from OpenAI, and the function arguments
        are parsed from JSON string format to a Python dictionary.
        """
        return [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            ) for tool_call in tool_calls
        ]