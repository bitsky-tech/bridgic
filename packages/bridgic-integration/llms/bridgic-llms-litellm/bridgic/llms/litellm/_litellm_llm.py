import warnings
from typing import List, Dict, Any, Optional

from typing_extensions import override
from pydantic import BaseModel

from bridgic.core.model import BaseLlm, RetryPolicyConfig, retryable_model_call
from bridgic.core.model.types import *
from bridgic.core.utils._collection import filter_dict, merge_dict, validate_required_params


class LiteLLMConfiguration(BaseModel):
    """
    Configuration for LiteLLM chat completions.

    Provides default values that can be overridden at call time.
    """

    model: Optional[str] = None
    """Default model to use when a call-time ``model`` is not provided."""
    temperature: Optional[float] = None
    """Sampling temperature in [0, 2]. Higher is more random, lower is more deterministic."""
    top_p: Optional[float] = None
    """Nucleus sampling probability mass in (0, 1]. Alternative to temperature."""
    presence_penalty: Optional[float] = None
    """Penalize new tokens based on whether they appear so far. [-2.0, 2.0]."""
    frequency_penalty: Optional[float] = None
    """Penalize new tokens based on their frequency so far. [-2.0, 2.0]."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate for the completion."""
    stop: Optional[List[str]] = None
    """Up to 4 sequences where generation will stop."""


class LiteLLM(BaseLlm):
    """
    LiteLLM integration for Bridgic, providing access to 100+ LLM providers
    (OpenAI, Anthropic, Google, Groq, Together AI, AWS Bedrock, Azure, etc.)
    through a single unified interface.

    Uses provider-prefixed model names, e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``groq/llama-3.3-70b-versatile``.
    See https://docs.litellm.ai/docs/providers for the full provider list.

    API keys are read from environment variables automatically by LiteLLM
    (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``). You can also pass
    ``api_key`` explicitly to override.

    Parameters
    ----------
    api_key : str, optional
        API key for the underlying provider. When ``None``, LiteLLM reads
        the appropriate key from environment variables.
    api_base : str, optional
        Base URL for the API endpoint. Useful for LiteLLM proxy or custom
        endpoints. When ``None``, uses the provider's default endpoint.
    configuration : LiteLLMConfiguration, optional
        Default configuration. If ``None``, uses ``LiteLLMConfiguration()``.
    timeout : float, optional
        Request timeout in seconds. If ``None``, no timeout is applied.

    Examples
    --------
    Basic usage for chat completion:

    ```python
    llm = LiteLLMLlm()
    messages = [Message.from_text("Hello!", role=Role.USER)]
    response = llm.chat(messages=messages, model="openai/gpt-4o")
    ```

    Using a different provider:

    ```python
    llm = LiteLLMLlm(api_key="sk-ant-...")
    response = llm.chat(
        messages=[Message.from_text("Hello!", role=Role.USER)],
        model="anthropic/claude-sonnet-4-6",
    )
    ```
    """

    api_key: Optional[str]
    api_base: Optional[str]
    configuration: LiteLLMConfiguration
    timeout: Optional[float]

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        configuration: Optional[LiteLLMConfiguration] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.configuration = configuration or LiteLLMConfiguration()
        self.timeout = timeout

    @retryable_model_call(RetryPolicyConfig())
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
        **kwargs,
    ) -> Response:
        """
        Send a synchronous chat completion request via LiteLLM.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages.
        model : str, optional
            LiteLLM model string (e.g. ``openai/gpt-4o``). Required unless
            provided in ``configuration.model``.
        temperature : float, optional
            Sampling temperature in [0, 2].
        top_p : float, optional
            Nucleus sampling probability mass in (0, 1].
        presence_penalty : float, optional
            Penalize new tokens based on prior appearance. [-2.0, 2.0].
        frequency_penalty : float, optional
            Penalize new tokens based on frequency. [-2.0, 2.0].
        max_tokens : int, optional
            Maximum tokens to generate.
        stop : list[str], optional
            Up to 4 sequences where generation will stop.
        **kwargs
            Additional keyword arguments forwarded to ``litellm.completion``.

        Returns
        -------
        Response
            Bridgic response containing the generated message and raw API response.
        """
        import litellm

        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model"])

        response = litellm.completion(**params)
        return self._handle_response(response)

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
        **kwargs,
    ) -> StreamResponse:
        """
        Stream a chat completion response incrementally via LiteLLM.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages.
        model : str, optional
            LiteLLM model string (e.g. ``openai/gpt-4o``).
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop
            See ``chat`` for details.
        **kwargs
            Additional keyword arguments forwarded to ``litellm.completion``.

        Yields
        ------
        MessageChunk
            Delta chunks as they arrive from the provider.
        """
        import litellm

        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model", "stream"])

        response = litellm.completion(**params)
        for chunk in response:
            delta_content = chunk.choices[0].delta.content if chunk.choices else None
            delta_content = delta_content if delta_content else ""
            yield MessageChunk(delta=delta_content, raw=chunk)

    @retryable_model_call(RetryPolicyConfig())
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
        **kwargs,
    ) -> Response:
        """
        Send an asynchronous chat completion request via LiteLLM.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages.
        model : str, optional
            LiteLLM model string (e.g. ``openai/gpt-4o``).
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop
            See ``chat`` for details.
        **kwargs
            Additional keyword arguments forwarded to ``litellm.acompletion``.

        Returns
        -------
        Response
            Bridgic response containing the generated message and raw API response.
        """
        import litellm

        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model"])

        response = await litellm.acompletion(**params)
        return self._handle_response(response)

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
        **kwargs,
    ) -> AsyncStreamResponse:
        """
        Stream a chat completion response asynchronously via LiteLLM.

        Parameters
        ----------
        messages : list[Message]
            Conversation messages.
        model : str, optional
            LiteLLM model string (e.g. ``openai/gpt-4o``).
        temperature, top_p, presence_penalty, frequency_penalty, max_tokens, stop
            See ``chat`` for details.
        **kwargs
            Additional keyword arguments forwarded to ``litellm.acompletion``.

        Yields
        ------
        MessageChunk
            Delta chunks as they arrive from the provider.
        """
        import litellm

        params = self._build_parameters(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        validate_required_params(params, ["messages", "model", "stream"])

        response = await litellm.acompletion(**params)
        async for chunk in response:
            delta_content = chunk.choices[0].delta.content if chunk.choices else None
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
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        msgs = [self._convert_message(msg) for msg in messages]
        merge_params = merge_dict(self.configuration.model_dump(), {
            "messages": msgs,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stop": stop,
            "stream": stream,
            "drop_params": True,
            **kwargs,
        })
        params = filter_dict(merge_params, exclude_none=True)

        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.timeout is not None:
            params["timeout"] = self.timeout

        return params

    @staticmethod
    def _convert_message(message: Message) -> Dict[str, str]:
        content_list = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                content_list.append(block.text)
            elif isinstance(block, ToolCallBlock):
                content_list.append(
                    f"Tool call:\n"
                    f"- id: {block.id}\n"
                    f"- name: {block.name}\n"
                    f"- arguments: {block.arguments}"
                )
            elif isinstance(block, ToolResultBlock):
                content_list.append(f"Tool result: {block.content}")
        content = "\n\n".join(content_list)

        role_map = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.AI: "assistant",
            Role.TOOL: "tool",
        }
        role = role_map.get(message.role)
        if role is None:
            raise ValueError(f"Invalid role: {message.role}")

        return {"role": role, "content": content}

    def _handle_response(self, response) -> Response:
        text = response.choices[0].message.content or ""

        if hasattr(response.choices[0].message, "refusal") and response.choices[0].message.refusal:
            warnings.warn(response.choices[0].message.refusal, RuntimeWarning)

        usage = self._extract_usage(response)
        return Response(
            message=Message.from_text(text, role=Role.AI),
            usage=usage,
            raw=response,
        )

    @staticmethod
    def _extract_usage(response) -> Optional[TokenUsage]:
        usage_data = getattr(response, "usage", None)
        if usage_data is None:
            return None

        return TokenUsage(
            model=getattr(response, "model", ""),
            prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
            completion_tokens=getattr(usage_data, "completion_tokens", 0),
            total_tokens=getattr(usage_data, "total_tokens", 0),
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "configuration": self.configuration.model_dump(),
        }

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self.api_key = state_dict.get("api_key")
        self.api_base = state_dict.get("api_base")
        self.timeout = state_dict.get("timeout")
        self.configuration = LiteLLMConfiguration(
            **state_dict.get("configuration", {})
        )
