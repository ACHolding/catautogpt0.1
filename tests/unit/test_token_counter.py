import pytest

from autogpt.compat import tiktoken_lib
from autogpt.llm.base import Message
from autogpt.llm.utils import count_message_tokens, count_string_tokens


def _using_heuristic() -> bool:
    return tiktoken_lib._load_tiktoken() is None


def test_count_message_tokens():
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    count = count_message_tokens(messages)
    if _using_heuristic():
        assert count >= 10
    else:
        assert count == 17


def test_count_message_tokens_empty_input():
    """Empty input should return 3 tokens"""
    assert count_message_tokens([]) == 3


def test_count_message_tokens_invalid_model():
    """Invalid model should raise a NotImplementedError"""
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    with pytest.raises(NotImplementedError):
        count_message_tokens(messages, model="invalid_model")


def test_count_message_tokens_gpt_4():
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    count = count_message_tokens(messages, model="gpt-4-0314")
    if _using_heuristic():
        assert count >= 10
    else:
        assert count == 15


def test_count_string_tokens():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    count = count_string_tokens(string, model_name="gpt-3.5-turbo-0301")
    if _using_heuristic():
        assert count >= 1
    else:
        assert count == 4


def test_count_string_tokens_empty_input():
    """Test that the string tokens are counted correctly."""

    assert count_string_tokens("", model_name="gpt-3.5-turbo-0301") == 0


def test_count_string_tokens_gpt_4():
    """Test that the string tokens are counted correctly."""

    string = "Hello, world!"
    count = count_string_tokens(string, model_name="gpt-4-0314")
    if _using_heuristic():
        assert count >= 1
    else:
        assert count == 4


def test_count_message_tokens_catseek_models():
    """CatSeek / DeepSeek local model ids must not raise NotImplementedError."""
    messages = [
        Message("user", "Hello"),
        Message("assistant", "Hi there!"),
    ]
    for model in ("catseek-gpu-0.1", "catseek", "deepseek-r1-14b"):
        count = count_message_tokens(messages, model=model)
        assert count >= 5


def test_count_string_tokens_catseek_models():
    string = "Hello, world!"
    for model in ("catseek-gpu-0.1", "catseek", "deepseek-r1-14b"):
        count = count_string_tokens(string, model_name=model)
        assert count >= 1


def test_chat_sequence_token_length_catseek():
    from autogpt.llm.base import ChatSequence

    seq = ChatSequence.for_model(
        "catseek-gpu-0.1",
        [Message("system", "sys"), Message("user", "do the thing")],
    )
    assert seq.token_length > 0

