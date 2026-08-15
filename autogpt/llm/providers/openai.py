"""Local BitNet LLM provider (via llama-cpp-python + BitNet GGUF).

Replaces the former OpenAI API provider. Model names like ``gpt-3.5-turbo`` /
``gpt-4`` are kept as aliases so the rest of Auto-GPT keeps working unchanged.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import numpy as np
from colorama import Fore

from autogpt.llm.base import (
    ChatModelInfo,
    EmbeddingModelInfo,
    MessageDict,
    TextModelInfo,
    TText,
)
from autogpt.logs import logger
from autogpt.models.command_registry import CommandRegistry

# Canonical BitNet chat model + compatibility aliases for old config values.
_BITNET_CHAT = ChatModelInfo(
    name="bitnet-b1.58",
    prompt_token_cost=0.0,
    completion_token_cost=0.0,
    max_tokens=4096,
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
        max_tokens=4096,
    ),
    "text-davinci-003": TextModelInfo(
        name="text-davinci-003",
        prompt_token_cost=0.0,
        completion_token_cost=0.0,
        max_tokens=4096,
    ),
}

OPEN_AI_EMBEDDING_MODELS = {
    "bitnet-embed": EmbeddingModelInfo(
        name="bitnet-embed",
        prompt_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=384,
    ),
    "text-embedding-ada-002": EmbeddingModelInfo(
        name="text-embedding-ada-002",
        prompt_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=384,
    ),
}

OPEN_AI_MODELS: dict[str, ChatModelInfo | EmbeddingModelInfo | TextModelInfo] = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_TEXT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}

_llm: Any = None
_llm_path: str | None = None
_EMBED_DIM = 384


def _resolve_model_path() -> Path:
    candidates = [
        os.getenv("BITNET_MODEL_PATH"),
        os.getenv("LLM_MODEL_PATH"),
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path

    # Common local layout relative to project root.
    project = Path(__file__).resolve().parents[3]
    for pattern in (
        "models/**/*.gguf",
        "models/*.gguf",
        "*.gguf",
    ):
        matches = sorted(project.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No BitNet GGUF model found. Set BITNET_MODEL_PATH to a .gguf file "
        "(e.g. microsoft/BitNet-b1.58-2B-4T-gguf via huggingface-cli)."
    )


def get_bitnet_llm(force_reload: bool = False) -> Any:
    """Lazy-load a BitNet (or compatible) GGUF model through llama-cpp-python."""
    global _llm, _llm_path
    model_path = str(_resolve_model_path())
    if _llm is not None and _llm_path == model_path and not force_reload:
        return _llm

    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise SystemExit(
            "llama-cpp-python is required for BitNet inference. "
            "Install with: pip install llama-cpp-python"
        ) from err

    n_ctx = int(os.getenv("BITNET_N_CTX", "4096"))
    n_threads = int(os.getenv("BITNET_N_THREADS", str(os.cpu_count() or 4)))
    logger.typewriter_log(
        "BitNet: ",
        Fore.GREEN,
        f"loading {model_path} (n_ctx={n_ctx}, threads={n_threads})",
    )
    _llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=os.getenv("BITNET_VERBOSE", "False") == "True",
    )
    _llm_path = model_path
    return _llm


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


def _chat_messages_to_prompt(messages: List[MessageDict]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m.get("content") or ""} for m in messages]


@meter_api
@retry_api()
def create_chat_completion(
    messages: List[MessageDict],
    *_,
    **kwargs,
) -> Any:
    """Create a chat completion using the local BitNet model."""
    llm = get_bitnet_llm()
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    max_tokens = int(kwargs.get("max_tokens") or 512)
    model_name = kwargs.get("model") or "bitnet-b1.58"

    result = llm.create_chat_completion(
        messages=_chat_messages_to_prompt(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = result["choices"][0]["message"].get("content") or ""
    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(prompt_tokens, completion_tokens, model_name)

    return SimpleNamespace(
        model=model_name,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[
            SimpleNamespace(
                message={"role": "assistant", "content": content},
            )
        ],
    )


@meter_api
@retry_api()
def create_text_completion(
    prompt: str,
    *_,
    **kwargs,
) -> Any:
    """Create a text completion using the local BitNet model."""
    llm = get_bitnet_llm()
    temperature = float(kwargs.get("temperature", 0.0) or 0.0)
    max_tokens = int(kwargs.get("max_tokens") or 512)
    model_name = kwargs.get("model") or "bitnet-b1.58-text"

    result = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        echo=False,
    )
    text = result["choices"][0]["text"]
    usage = result.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(prompt_tokens, completion_tokens, model_name)

    return SimpleNamespace(
        model=model_name,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[SimpleNamespace(text=text)],
    )


def _hash_embedding(text: str, dims: int = _EMBED_DIM) -> list[float]:
    """Deterministic local embedding fallback when the GGUF has no embed head."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dims).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


@meter_api
@retry_api()
def create_embedding(
    input: str | TText | List[str] | List[TText],
    *_,
    **kwargs,
) -> Any:
    """Create embeddings via BitNet/llama.cpp, with a local hash fallback."""
    texts: list[str]
    if isinstance(input, str):
        texts = [input]
    elif isinstance(input, list) and input and isinstance(input[0], int):
        texts = [" ".join(str(t) for t in input)]
    elif isinstance(input, list):
        texts = []
        for item in input:
            if isinstance(item, str):
                texts.append(item)
            else:
                texts.append(" ".join(str(t) for t in item))
    else:
        texts = [str(input)]

    model_name = kwargs.get("model") or "bitnet-embed"
    data = []
    try:
        llm = get_bitnet_llm()
        for idx, text in enumerate(texts):
            emb = llm.create_embedding(text)
            vector = emb["data"][0]["embedding"]
            data.append({"index": idx, "embedding": vector})
    except Exception as err:
        logger.warn(
            f"BitNet embedding unavailable ({err}); using local hash embeddings."
        )
        for idx, text in enumerate(texts):
            data.append({"index": idx, "embedding": _hash_embedding(text)})

    from autogpt.llm.api_manager import ApiManager

    ApiManager().update_cost(sum(len(t.split()) for t in texts), 0, model_name)
    return SimpleNamespace(data=data, model=model_name)


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
