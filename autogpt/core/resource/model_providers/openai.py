import enum
import functools
import logging
import math
import time
from types import SimpleNamespace
from typing import Callable, ParamSpec, TypeVar

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.schema import (
    Embedding,
    EmbeddingModelProvider,
    EmbeddingModelProviderModelInfo,
    LanguageModelFunction,
    LanguageModelMessage,
    LanguageModelProvider,
    LanguageModelProviderModelInfo,
    LanguageModelProviderModelResponse,
    ModelProviderBudget,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
)

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    # Values remain OpenAI-compatible aliases; the BitNet provider resolves them locally.
    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo-0613"
    GPT3_16K = "gpt-3.5-turbo-16k-0613"
    GPT4 = "gpt-4-0613"
    GPT4_32K = "gpt-4-32k-0613"


OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelProviderModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=384,
    ),
}


OPEN_AI_LANGUAGE_MODELS = {
    OpenAIModelName.GPT3: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT3,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=4096,
    ),
    OpenAIModelName.GPT3_16K: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT3_16K,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=8192,
    ),
    OpenAIModelName.GPT4: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT4,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=4096,
    ),
    OpenAIModelName.GPT4_32K: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT4_32K,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=8192,
    ),
}


OPEN_AI_MODELS = {
    **OPEN_AI_LANGUAGE_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable()


class OpenAIModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: ModelProviderCredentials
    budget: OpenAIModelProviderBudget


class OpenAIProvider(
    Configurable,
    LanguageModelProvider,
    EmbeddingModelProvider,
):
    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to a local BitNet LLM (llama.cpp).",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=ModelProviderCredentials(),
        budget=OpenAIModelProviderBudget(
            total_budget=math.inf,
            total_cost=0.0,
            remaining_budget=math.inf,
            usage=ModelProviderUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            graceful_shutdown_threshold=0.005,
            warning_threshold=0.01,
        ),
    )

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
        )

        self._create_completion = retry_handler(_create_completion)
        self._create_embedding = retry_handler(_create_embedding)

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

    async def create_language_completion(
        self,
        model_prompt: list[LanguageModelMessage],
        functions: list[LanguageModelFunction],
        model_name: OpenAIModelName,
        completion_parser: Callable[[dict], dict],
        **kwargs,
    ) -> LanguageModelProviderModelResponse:
        """Create a completion using the local BitNet model."""
        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        response = await self._create_completion(
            messages=model_prompt,
            **completion_kwargs,
        )
        response_args = {
            "model_info": OPEN_AI_LANGUAGE_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }

        message = response.choices[0].message
        if hasattr(message, "to_dict_recursive"):
            parsed_payload = message.to_dict_recursive()
        elif isinstance(message, dict):
            parsed_payload = message
        else:
            parsed_payload = {
                "role": "assistant",
                "content": getattr(message, "content", ""),
            }

        parsed_response = completion_parser(parsed_payload)
        response = LanguageModelProviderModelResponse(
            content=parsed_response, **response_args
        )
        self._budget.update_usage_and_cost(response)
        return response

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelProviderModelResponse:
        """Create an embedding using the local BitNet model."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens_used": getattr(response.usage, "completion_tokens", 0),
        }
        response = EmbeddingModelProviderModelResponse(
            **response_args,
            embedding=embedding_parser(response.data[0]["embedding"]),
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(
        self,
        model_name: OpenAIModelName,
        functions: list[LanguageModelFunction],
        **kwargs,
    ) -> dict:
        completion_kwargs = {
            "model": model_name,
            **kwargs,
        }
        if functions:
            completion_kwargs["functions"] = functions
        return completion_kwargs

    def _get_embedding_kwargs(
        self,
        model_name: OpenAIModelName,
        **kwargs,
    ) -> dict:
        return {
            "model": model_name,
            **kwargs,
        }

    def __repr__(self):
        return "OpenAIProvider(BitNet)"


async def _create_embedding(text: str, *_, **kwargs):
    """Embed text using the local BitNet provider."""
    import asyncio

    from autogpt.llm.providers import openai as bitnet

    result = await asyncio.to_thread(bitnet.create_embedding, text, **kwargs)
    if not hasattr(result, "usage"):
        result.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    if not hasattr(result, "embeddings"):
        result.embeddings = [item["embedding"] for item in result.data]
    return result


async def _create_completion(messages: list[LanguageModelMessage], *_, **kwargs):
    """Create a chat completion using the local BitNet provider."""
    import asyncio

    from autogpt.llm.providers import openai as bitnet

    payload = [
        message.model_dump() if hasattr(message, "model_dump") else dict(message)
        for message in messages
    ]
    # Function calling is not supported on local BitNet; drop unused kwargs.
    kwargs.pop("functions", None)
    return await asyncio.to_thread(bitnet.create_chat_completion, payload, **kwargs)


_T = TypeVar("_T")
_P = ParamSpec("_P")


class _OpenAIRetryHandler:
    """Retry handler for local BitNet inference."""

    _backoff_msg = "Error: BitNet call failed. Waiting {backoff} seconds..."

    def __init__(
        self,
        logger: logging.Logger,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._logger = logger
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        async def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            num_attempts = self._num_retries + 1
            for attempt in range(1, num_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == num_attempts:
                        raise
                    self._backoff(attempt)

        return _wrapped
