import json
import httpx
import warnings

from typing import List, Dict, Tuple, Optional, overload, Any, Union, Literal
from typing_extensions import override
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageFunctionToolCall, ChatCompletionNamedToolChoiceParam, ChatCompletionToolChoiceOptionParam
from openai.types.chat.chat_completion_message_tool_call_param import ChatCompletionMessageToolCallParam, Function
from pydantic import BaseModel
from openai import Stream, OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.resources.chat.completions.completions import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam

from bridgic.core.model import BaseLlm
from bridgic.core.model.types import *
from bridgic.core.model.protocols import StructuredOutput, ToolSelection, PydanticModel, JsonSchema, Constraint
from bridgic.core.utils._console import printer
from bridgic.core.utils._collection import filter_dict, merge_dict, validate_required_params

class OpenAIConfiguration(BaseModel):
    """Default configuration for OpenAI chat completions.

    model : str
        Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
    temperature : Optional[float]
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
        make the output more random, while lower values like 0.2 will make it more
        focused and deterministic.
    top_p : Optional[float]
        An alternative to sampling with temperature, called nucleus sampling, where the
        model considers the results of the tokens with top_p probability mass.
    presence_penalty : Optional[float]
        Number between -2.0 and 2.0. Positive values penalize new tokens based on
        whether they appear in the text so far, increasing the model's likelihood to
        talk about new topics.
    frequency_penalty : Optional[float]
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their
        existing frequency in the text so far, decreasing the model's likelihood to
        repeat the same line verbatim.
    max_tokens : Optional[int]
        The maximum number of tokens that can be generated in the chat completion.
        This value is now deprecated in favor of `max_completion_tokens`.
    stop : Optional[List[str]]
        Up to 4 sequences where the API will stop generating further tokens.
        Not supported with latest reasoning models `o3` and `o3-mini`.
    """
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None

class OpenAILlm(BaseLlm, StructuredOutput, ToolSelection):
    """
    Wrapper class for OpenAI, providing common chat and stream calling interfaces for OpenAI model
    and implementing the common protocols in the Bridgic framework.

    Parameters
    ----------
    api_key : str
        The API key for OpenAI services. Required for authentication.
    api_base : Optional[str]
        The base URL for the OpenAI API. If None, uses the default OpenAI endpoint.
    timeout : Optional[float]
        Request timeout in seconds. If None, no timeout is applied.
    http_client : Optional[httpx.Client]
        Custom synchronous HTTP client for requests. If None, creates a default client.
    http_async_client : Optional[httpx.AsyncClient]
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

    ```python
    llm = OpenAILlm(api_key="your-api-key")
    messages = [Message.from_text("Hello!", role=Role.USER)]
    response = llm.chat(messages=messages, model="gpt-4")
    ```

    Structured output with Pydantic model:

    ```python
    class Answer(BaseModel):
        reasoning: str
        result: int

    constraint = PydanticModel(model=Answer)
    structured_response = llm.structured_output(
        messages=messages,
        constraint=constraint,
        model="gpt-4"
    )
    ```

    Tool calling:

    ```python
    tools = [Tool(name="calculator", description="Calculate math", parameters={})]
    tool_calls, tool_call_response = llm.select_tool(messages=messages, tools=tools, model="gpt-4")
    ```
    """

    api_base: str
    api_key: str
    configuration: OpenAIConfiguration
    timeout: float
    http_client: httpx.Client
    http_async_client: httpx.AsyncClient

    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        configuration: Optional[OpenAIConfiguration] = OpenAIConfiguration(),
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
        api_base : Optional[str]
            The base URL for the OpenAI API. If None, uses the default OpenAI endpoint.
        configuration : Optional[OpenAIConfiguration]
            The configuration for the OpenAI API. If None, uses the default configuration.
        timeout : Optional[float]
            Request timeout in seconds. If None, no timeout is applied.
        http_client : Optional[httpx.Client]
            Custom synchronous HTTP client for requests. If None, creates a default client.
        http_async_client : Optional[httpx.AsyncClient]
            Custom asynchronous HTTP client for requests. If None, creates a default client.
        """
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
        tools: Optional[List[Tool]] = None,
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
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int]
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]]
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        tools : Optional[List[Tool]]
            A list of tools to use in the chat completion.
        extra_body : Optional[Dict[str, Any]]
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Response
            A response object containing the generated message and raw API response.
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
        # Validate required parameters for non-streaming chat completion
        validate_required_params(params, ["messages", "model"])
        
        response: ChatCompletion = self.client.chat.completions.create(**params)
        return self._handle_chat_response(response)

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
        Send a streaming chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int]
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]]
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]]
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
        # Validate required parameters for streaming chat completion
        validate_required_params(params, ["messages", "model", "stream"])
        
        response: Stream[ChatCompletionChunk] = self.client.chat.completions.create(**params)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
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
        tools: Optional[List[Tool]] = None,
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
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int]
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]]
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        tools : Optional[List[Tool]]
            A list of tools to use in the chat completion.
        extra_body : Optional[Dict[str, Any]]
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
        # Validate required parameters for non-streaming chat completion
        validate_required_params(params, ["messages", "model"])
        
        response = await self.async_client.chat.completions.create(**params)
        return self._handle_chat_response(response)

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
        Send an asynchronous streaming chat completion request to OpenAI.

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far.
        model : str
            Model ID used to generate the response, like `gpt-4o` or `gpt-4`.
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        max_tokens : Optional[int]
            The maximum number of tokens that can be generated in the chat completion.
            This value is now deprecated in favor of `max_completion_tokens`.
        stop : Optional[List[str]]
            Up to 4 sequences where the API will stop generating further tokens.
            Not supported with latest reasoning models `o3` and `o3-mini`.
        extra_body : Optional[Dict[str, Any]]
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
        # Validate required parameters for streaming chat completion
        validate_required_params(params, ["messages", "model", "stream"])
        
        response = await self.async_client.chat.completions.create(**params)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
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
        tools: Optional[List[Tool]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        msgs: List[ChatCompletionMessageParam] = [self._convert_chat_completions_message(msg) for msg in messages]
        
        # Handle tools parameter - convert to list if provided, otherwise use empty list
        json_desc_tools = [self._convert_tool_to_json(tool) for tool in tools] if tools is not None else None
        
        # Build parameters dictionary and filter out None values
        # The priority order is as follows: configuration passed through the interface > configuration of the instance itself.
        merge_params = merge_dict(self.configuration.model_dump(), {
            "messages": msgs,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "tools": json_desc_tools,
            "extra_body": extra_body,
            "stream": stream,
            "response_format": response_format,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            **kwargs,
        })
        
        params = filter_dict(merge_params, exclude_none=True)
        return params

    def _handle_chat_response(self, response: ChatCompletion) -> Response:
        openai_message = response.choices[0].message
        text = openai_message.content if openai_message.content else ""
        
        if openai_message.refusal:
            warnings.warn(openai_message.refusal, RuntimeWarning)

        # Handle tool calls in the response
        # if openai_message.tool_calls:
        #     # Create a message with both text content and tool calls
        #     blocks = []
        #     if text:
        #         blocks.append(TextBlock(text=text))
        #     else:
        #         # Ensure there's always some text content, even if empty
        #         blocks.append(TextBlock(text=""))
            
        #     for tool_call in openai_message.tool_calls:
        #         tool_call_block = ToolCallBlock(
        #             id=tool_call.id,
        #             name=tool_call.function.name,
        #             arguments=json.loads(tool_call.function.arguments)
        #         )
        #         blocks.append(tool_call_block)
            
        #     message = Message(role=Role.AI, blocks=blocks)
        # else:
        #     # Regular text response
        #     message = Message.from_text(text, role=Role.AI)

        return Response(
            message=Message.from_text(text, role=Role.AI),
            raw=response,
        )

    def _convert_chat_completions_message(self, message: Message) -> ChatCompletionMessageParam:
        """
        Convert a Bridgic Message to OpenAI ChatCompletionMessageParam.
        
        This method handles different message types including:
        - Text messages
        - Messages with tool calls (ToolCallBlock)
        - Messages with tool results (ToolResultBlock)
        
        Parameters
        ----
        message : Message
            The Bridgic message to convert
            
        Returns
        ----
        ChatCompletionMessageParam
            The converted OpenAI message parameter
        """
        # Extract text content from TextBlocks and ToolResultBlocks
        content_list = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                content_list.append(block.text)
            elif isinstance(block, ToolResultBlock):
                content_list.append(block.content)
        content_txt = "\n\n".join(content_list) if content_list else ""
        
        # Extract tool calls from ToolCallBlocks
        tool_calls = []
        for block in message.blocks:
            if isinstance(block, ToolCallBlock):
                tool_call = ChatCompletionMessageToolCallParam(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.arguments)
                    )
                )
                tool_calls.append(tool_call)
        
        # Handle different message roles
        if message.role == Role.SYSTEM:
            return ChatCompletionSystemMessageParam(content=content_txt, role="system", **message.extras)
        elif message.role == Role.USER:
            return ChatCompletionUserMessageParam(content=content_txt, role="user", **message.extras)
        elif message.role == Role.AI:
            # For AI messages, include tool calls if present
            if tool_calls:
                return ChatCompletionAssistantMessageParam(
                    content=content_txt, 
                    role="assistant", 
                    tool_calls=tool_calls,
                    **message.extras
                )
            else:
                return ChatCompletionAssistantMessageParam(content=content_txt, role="assistant", **message.extras)
        elif message.role == Role.TOOL:
            # For tool messages, extract tool_call_id from ToolResultBlock
            tool_call_id = None
            for block in message.blocks:
                if isinstance(block, ToolResultBlock):
                    tool_call_id = block.id
                    break
            
            if tool_call_id is None:
                raise ValueError("Tool message must contain a ToolResultBlock with an ID")
            
            return ChatCompletionToolMessageParam(
                content=content_txt, 
                role="tool", 
                tool_call_id=tool_call_id,
                **message.extras
            )
        else:
            raise ValueError(f"Invalid role: {message.role}")
    
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


    def structured_output(
        self,
        messages: List[Message],
        constraint: Union[PydanticModel, JsonSchema],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[BaseModel, Dict[str, Any]]:
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
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]]
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Union[BaseModel, Dict[str, Any]]
            The structured response in the format specified by the constraint:
            - BaseModel instance if constraint is PydanticModel
            - Dict[str, Any] if constraint is JsonSchema

        Examples
        --------
        Using a Pydantic model constraint:

        ```python
        class Answer(BaseModel):
            reasoning: str
            result: int

        constraint = PydanticModel(model=Answer)
        response = llm.structured_output(
            messages=[Message.from_text("What is 2+2?", role=Role.USER)],
            constraint=constraint,
            model="gpt-4o"
        )
        print(response.reasoning, response.result)
        ```

        Using a JSON schema constraint:

        ```python
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        constraint = JsonSchema(schema=schema)
        response = llm.structured_output(
            messages=[Message.from_text("Hello", role=Role.USER)],
            constraint=constraint,
            model="gpt-4o"
        )
        print(response["answer"])
        ```

        Notes
        -----
        - Utilizes OpenAI's native structured output API with strict schema validation
        - All schemas automatically have additionalProperties set to False
        - Best performance achieved with GPT-4o and later models (gpt-4o-mini, gpt-4o-2024-08-06, and later)
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=extra_body,
            response_format=self._get_response_format(constraint),
            **kwargs,
        )
        # Validate required parameters for structured output
        validate_required_params(params, ["messages", "model"])
        
        response = self.client.chat.completions.parse(**params)
        return self._convert_response(constraint, response.choices[0].message.content)

    async def astructured_output(
        self,
        messages: List[Message],
        constraint: Union[PydanticModel, JsonSchema],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[BaseModel, Dict[str, Any]]:
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
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]]
            Add additional JSON properties to the request.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        Union[BaseModel, Dict[str, Any]]
            The structured response in the format specified by the constraint:
            - BaseModel instance if constraint is PydanticModel
            - Dict[str, Any] if constraint is JsonSchema

        Examples
        --------
        Using asynchronous structured output:

        ```python
        async def get_structured_response():
            llm = OpenAILlm(api_key="your-key")
            constraint = PydanticModel(model=Answer)
            response = await llm.astructured_output(
                messages=[Message.from_text("Calculate 5+3", role=Role.USER)],
                constraint=constraint,
                model="gpt-4o"
            )
            return response
        ```

        Notes
        -----
        - This is the asynchronous version of structured_output
        - Utilizes OpenAI's native structured output API with strict schema validation
        - Suitable for concurrent processing and high-throughput applications
        - Best performance achieved with GPT-4o and later models (gpt-4o-mini, gpt-4o-2024-08-06, and later)
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            extra_body=extra_body,
            response_format=self._get_response_format(constraint),
            **kwargs,
        )
        # Validate required parameters for structured output
        validate_required_params(params, ["messages", "model"])
        
        response = await self.async_client.chat.completions.parse(**params)
        return self._convert_response(constraint, response.choices[0].message.content)

    def _add_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI requires additionalProperties to be set to False for all objects
        in structured output schemas. See:
        [AdditionalProperties False Must Always Be Set in Objects](https://platform.openai.com/docs/guides/structured-outputs?example=moderation#additionalproperties-false-must-always-be-set-in-objects)
        """
        schema["additionalProperties"] = False
        return schema
    
    def _get_response_format(self, constraint: Union[PydanticModel, JsonSchema]) -> Dict[str, Any]:
        if isinstance(constraint, PydanticModel):
            result = {
                "type": "json_schema",
                "json_schema": {
                    "schema": self._add_schema_properties(constraint.model.model_json_schema()),
                    "name": constraint.model.__name__,
                    "strict": True,
                },
            }
            return result
        elif isinstance(constraint, JsonSchema):
            return {
                "type": "json_schema",
                "json_schema": {
                    "schema": self._add_schema_properties(constraint.schema_dict),
                    # default name for schema
                    "name": "schema",
                    "strict": True,
                },
            }
        else:
            raise ValueError(f"Unsupported constraint type '{constraint.constraint_type}'. More info about OpenAI structured output: https://platform.openai.com/docs/guides/structured-outputs")

    def _convert_response(
        self,
        constraint: Union[PydanticModel, JsonSchema],
        content: str,
    ) -> Union[BaseModel, Dict[str, Any]]:
        if isinstance(constraint, PydanticModel):
            return constraint.model.model_validate_json(content)
        elif isinstance(constraint, JsonSchema):
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported constraint type '{constraint.constraint_type}'. More info about OpenAI structured output: https://platform.openai.com/docs/guides/structured-outputs")

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
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam] = None,
        **kwargs,
    ) -> Tuple[List[ToolCall], Optional[str]]:
        """
        Select and invoke tools from a list based on conversation context.

        This method enables the model to intelligently select and call appropriate tools
        from a provided list based on the conversation context. It supports OpenAI's
        function calling capabilities with parallel execution and various control options.

        More OpenAI information: [function-calling](https://platform.openai.com/docs/guides/function-calling)

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far providing context for tool selection.
        tools : List[Tool]
            A list of tools the model may call.
        model : str
            Model ID used to generate the response. Function calling requires compatible models.
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]]
            Add additional JSON properties to the request.
        parallel_tool_calls : Optional[bool]
            Whether to enable parallel function calling during tool use.
        tool_choice : Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam]
            Controls which tool, if any, the model may call.
            - `none`: The model will not call any tool and will instead generate a message. This is the default when no tools are provided.
            - `auto`: The model may choose to generate a message or call one or more tools. This is the default when tools are provided.
            - `required`: The model must call one or more tools.
            - To force a specific tool, pass `{"type": "function", "function": {"name": "my_function"}}`.
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        List[ToolCall]
            List of selected tool calls with their IDs, names, and parsed arguments.
        Union[str, None]
            The content of the message from the model.
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
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
        # Validate required parameters for tool selection
        validate_required_params(params, ["messages", "model"])
        
        response: ChatCompletion = self.client.chat.completions.create(**params)
        tool_calls = response.choices[0].message.tool_calls
        content = response.choices[0].message.content
        return (self._convert_tool_calls(tool_calls), content)

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
        parallel_tool_calls: Optional[bool] = None,
        tool_choice: Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam] = None,
        **kwargs,
    )-> Tuple[List[ToolCall], Optional[str]]:
        """
        Select and invoke tools from a list based on conversation context.

        This method enables the model to intelligently select and call appropriate tools
        from a provided list based on the conversation context. It supports OpenAI's
        function calling capabilities with parallel execution and various control options.

        More OpenAI information: [function-calling](https://platform.openai.com/docs/guides/function-calling)

        Parameters
        ----------
        messages : List[Message]
            A list of messages comprising the conversation so far providing context for tool selection.
        tools : List[Tool]
            A list of tools the model may call.
        model : str
            Model ID used to generate the response. Function calling requires compatible models.
        temperature : Optional[float]
            What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it more
            focused and deterministic.
        top_p : Optional[float]
            An alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
        presence_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on
            whether they appear in the text so far, increasing the model's likelihood to
            talk about new topics.
        frequency_penalty : Optional[float]
            Number between -2.0 and 2.0. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's likelihood to
            repeat the same line verbatim.
        extra_body : Optional[Dict[str, Any]]
            Add additional JSON properties to the request.
        parallel_tool_calls : Optional[bool]
            Whether to enable parallel function calling during tool use.
        tool_choice : Union[Literal["auto", "required", "none"], ChatCompletionNamedToolChoiceParam]
            Controls which tool, if any, the model may call.
            - `none`: The model will not call any tool and will instead generate a message. This is the default when no tools are provided.
            - `auto`: The model may choose to generate a message or call one or more tools. This is the default when tools are provided.
            - `required`: The model must call one or more tools.
            - To force a specific tool, pass `{"type": "function", "function": {"name": "my_function"}}`.
        
        **kwargs
            Additional keyword arguments passed to the OpenAI API.

        Returns
        -------
        List[ToolCall]
            List of selected tool calls with their IDs, names, and parsed arguments.
        Union[str, None]
            The content of the message from the model.
        """
        params = self._build_parameters(
            messages=messages,
            model=model,
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
        # Validate required parameters for tool selection
        validate_required_params(params, ["messages", "model"])
        
        response: ChatCompletion = await self.async_client.chat.completions.create(**params)
        tool_calls = response.choices[0].message.tool_calls
        content = response.choices[0].message.content
        return (self._convert_tool_calls(tool_calls), content)

    def _convert_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
            "additionalProperties": False
        }

    def _convert_tool_to_json(self, tool: Tool) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": self._convert_parameters(tool.parameters),
            }
        }

    def _convert_tool_calls(self, tool_calls: List[ChatCompletionMessageFunctionToolCall]) -> List[ToolCall]:
        return [] if tool_calls is None else [
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            ) for tool_call in tool_calls
        ]
    
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
        self.configuration = OpenAIConfiguration(**state_dict.get("configuration", {}))
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