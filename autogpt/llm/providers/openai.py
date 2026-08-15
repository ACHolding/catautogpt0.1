"""Local BitNet LLM provider (BitNet engine + compatibility aliases).

Replaces the former OpenAI API provider. Model names like ``gpt-3.5-turbo`` /
``gpt-4`` are kept as aliases so the rest of Auto-GPT keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from autogpt.llm.base import (
    ChatModelInfo,
    EmbeddingModelInfo,
    MessageDict,
    TextModelInfo,
    TText,
)
from autogpt.llm.providers import bitnet_engine
from autogpt.models.command_registry import CommandRegistry

# Canonical BitNet chat model + compatibility aliases for old config values.
_BITNET_CHAT = ChatModelInfo(
    name="bitnet-b1.58",
    prompt_token_cost=0.0,
    completion_token_cost=0.0,
    max_tokens=bitnet_engine.DEFAULT_N_CTX,
    supports_functions=False,
)

OPEN_AI_CHAT_MODELS: dict[str, ChatModelInfo] = {
    "bitnet-b1.58": _BITNET_CHAT,
    # Aliases so existing FAST_LLM / SMART_LLM env values still resolve.
    "gpt-3.5-turbo": ChatModelInfo(**{**_BITNET_CHAT.__dict__, "name": "gpt-3.5-turbo"}),
    "gpt-3.5-turbo-0301": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-3.5-turbo-0301"}
    ),
    "gpt-3.5-turbo-0613": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-3.5-turbo-0613"}
    ),
    "gpt-3.5-turbo-16k": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-3.5-turbo-16k", "max_tokens": 8192}
    ),
    "gpt-3.5-turbo-16k-0613": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-3.5-turbo-16k-0613", "max_tokens": 8192}
    ),
    "gpt-4": ChatModelInfo(**{**_BITNET_CHAT.__dict__, "name": "gpt-4"}),
    "gpt-4-0314": ChatModelInfo(**{**_BITNET_CHAT.__dict__, "name": "gpt-4-0314"}),
    "gpt-4-0613": ChatModelInfo(**{**_BITNET_CHAT.__dict__, "name": "gpt-4-0613"}),
    "gpt-4-32k": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-4-32k", "max_tokens": 8192}
    ),
    "gpt-4-32k-0314": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-4-32k-0314", "max_tokens": 8192}
    ),
    "gpt-4-32k-0613": ChatModelInfo(
        **{**_BITNET_CHAT.__dict__, "name": "gpt-4-32k-0613", "max_tokens": 8192}
    ),
}

OPEN_AI_TEXT_MODELS = {
    "bitnet-b1.58-text": TextModelInfo(
        name="bitnet-b1.58-text",
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=bitnet_engine.DEFAULT_N_CTX,
    ),
    "text-davinci-003": TextModelInfo(
        name="text-davinci-003",
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=bitnet_engine.DEFAULT_N_CTX,
    ),
}

OPEN_AI_EMBEDDING_MODELS = {
    "bitnet-embed": EmbeddingModelInfo(
        name="bitnet-embed",
        prompt_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=bitnet_engine.DEFAULT_EMBED_DIM,
    ),
    "text-embedding-ada-002": EmbeddingModelInfo(
        name="text-embedding-ada-002",
        prompt_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=bitnet_engine.DEFAULT_EMBED_DIM,
    ),
}

OPEN_AI_MODELS: dict[str, ChatModelInfo | EmbeddingModelInfo | TextModelInfo] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_TEXT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


# Back-compat exports used by config / older imports.
_resolve_model_path = bitnet_engine.resolve_chat_model_path
get_bitnet_llm = bitnet_engine.get_chat_llm


def meter_api(func: Callable):
    """No-op compatibility wrapper (OpenAI metering removed)."""
    return func


def retry_api(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Simple local retry wrapper (replaces OpenAI rate-limit retries)."""

    def _wrapper(func: Callable):
        def _wrapped(*args, **kwargs):
            import time

            from colorama import Fore

            from autogpt.logs import logger

            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    delay = backoff_base**attempt
                    logger.warn(
                        f"{Fore.YELLOW}BitNet call failed ({err}); "
                        f"retrying in {delay:.1f}s ({attempt}/{max_retries}){Fore.RESET}"
                    )
                    time.sleep(delay)

        return _wrapped

    return _wrapper


@meter_api
@retry_api()
def create_chat_completion(
    messages: List[MessageDict],
    *_,
    **kwargs,
) -> Any:
    """Create a chat completion using the local BitNet engine."""
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    max_tokens = int(kwargs.get("max_tokens") or bitnet_engine.DEFAULT_MAX_TOKENS)
    model_name = kwargs.get("model") or "bitnet-b1.58"

    response = bitnet_engine.create_chat_completion_raw(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response.model = model_name

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        model_name,
    )
    return response


@meter_api
@retry_api()
def create_text_completion(
    prompt: str,
    *_,
    **kwargs,
) -> Any:
    """Create a text completion using the local BitNet engine."""
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    max_tokens = int(kwargs.get("max_tokens") or bitnet_engine.DEFAULT_MAX_TOKENS)
    model_name = kwargs.get("model") or "bitnet-b1.58-text"

    response = bitnet_engine.create_text_completion_raw(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response.model = model_name

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        model_name,
    )
    return response


def _coerce_embedding_texts(
    input: str | TText | List[str] | List[TText],
) -> list[str]:
    if isinstance(input, str):
        return [input]
    if isinstance(input, list) and input and isinstance(input[0], int):
        return [" ".join(str(t) for t in input)]
    if isinstance(input, list):
        texts: list[str] = []
        for item in input:
            if isinstance(item, str):
                texts.append(item)
            else:
                texts.append(" ".join(str(t) for t in item))
        return texts
    return [str(input)]


@meter_api
@retry_api()
def create_embedding(
    input: str | TText | List[str] | List[TText],
    *_,
    **kwargs,
) -> Any:
    """Create embeddings via BitNet embedding GGUF / chat fallback / hash."""
    texts = _coerce_embedding_texts(input)
    model_name = kwargs.get("model") or "bitnet-embed"
    response = bitnet_engine.create_embedding_raw(texts)
    response.model = model_name

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(sum(len(t.split()) for t in texts), 0, model_name)
    return response


@dataclass
class OpenAIFunctionCall:
    """Command call payload (name kept for compatibility)."""

    name: str
    arguments: str


@dataclass
class OpenAIFunctionSpec:
    """Function/tool schema mapped from Auto-GPT commands."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]

    @dataclass
    class ParameterSpec:
        name: str
        type: str
        description: Optional[str]
        required: bool = False

    @property
    def schema(self) -> dict[str, str | dict | list]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                    }
                    for param in self.parameters.values()
                },
                "required": [
                    param.name for param in self.parameters.values() if param.required
                ],
            },
        }

    @property
    def prompt_format(self) -> str:
        def param_signature(p_spec: OpenAIFunctionSpec.ParameterSpec) -> str:
            return (
                f"// {p_spec.description}\n" if p_spec.description else ""
            ) + f"{p_spec.name}{'' if p_spec.required else '?'}: {p_spec.type},"

        return "\n".join(
            [
                f"// {self.description}",
                f"type {self.name} = (_ :{{",
                *[param_signature(p) for p in self.parameters.values()],
                "}) => any;",
            ]
        )


def get_openai_command_specs(
    command_registry: CommandRegistry,
) -> list[OpenAIFunctionSpec]:
    return [
        OpenAIFunctionSpec(
            name=command.name,
            description=command.description,
            parameters={
                param.name: OpenAIFunctionSpec.ParameterSpec(
                    name=param.name,
                    type=param.type,
                    required=param.required,
                    description=param.description,
                )
                for param in command.parameters
            },
        )
        for command in command_registry.commands.values()
    ]


def count_openai_functions_tokens(
    functions: list[OpenAIFunctionSpec], for_model: str
) -> int:
    from autogpt.llm.utils import count_string_tokens

    return count_string_tokens(
        f"# Tools\n\n## functions\n\n{format_function_specs_as_typescript_ns(functions)}",
        for_model,
    )


def format_function_specs_as_typescript_ns(functions: list[OpenAIFunctionSpec]) -> str:
    return (
        "namespace functions {\n\n"
        + "\n\n".join(f.prompt_format for f in functions)
        + "\n\n} // namespace functions"
    )
