"""Functions for counting the number of tokens in a message or string."""
from __future__ import annotations

from typing import List, overload

from autogpt.compat import tiktoken_lib
from autogpt.llm.base import Message
from autogpt.logs import logger


def _is_bitnet_model(model: str) -> bool:
    return model.startswith("bitnet")


@overload
def count_message_tokens(messages: Message, model: str = "gpt-3.5-turbo") -> int:
    ...


@overload
def count_message_tokens(messages: List[Message], model: str = "gpt-3.5-turbo") -> int:
    ...


def count_message_tokens(
    messages: Message | List[Message], model: str = "gpt-3.5-turbo"
) -> int:
    """
    Returns the number of tokens used by a list of messages.
    """
    if isinstance(messages, Message):
        messages = [messages]

    if _is_bitnet_model(model):
        try:
            from autogpt.llm.providers import bitnet_engine

            payload = [
                {"role": m.role, "content": m.content or ""} for m in messages
            ]
            return bitnet_engine.count_chat_tokens(payload)
        except Exception:
            # Fall through to tiktoken estimate if the GGUF is not loaded yet.
            pass

    if model.startswith("gpt-3.5-turbo") or model.startswith("bitnet"):
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
        encoding_model = "gpt-3.5-turbo"
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
        encoding_model = "gpt-4"
    else:
        raise NotImplementedError(
            f"count_message_tokens() is not implemented for model {model}.\n"
            " See https://github.com/openai/openai-python/blob/main/chatml.md for"
            " information on how messages are converted to tokens."
        )
    try:
        encoding = tiktoken_lib.encoding_for_model(encoding_model)
    except KeyError:
        logger.warn("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken_lib.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.raw().items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_string_tokens(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.
    """
    if _is_bitnet_model(model_name):
        if string == "":
            return 0
        try:
            from autogpt.llm.providers import bitnet_engine

            return bitnet_engine.tokenize_count(string)
        except Exception:
            return max(1, len(string) // 4)

    try:
        encoding = tiktoken_lib.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken_lib.get_encoding("cl100k_base")
    return len(encoding.encode(string))
