from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autogpt.llm.api_manager import ApiManager
from autogpt.llm.providers import openai

api_manager = ApiManager()


@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield


def _mock_llm_response(prompt_tokens=10, completion_tokens=20, content="ok"):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class TestProviderBitNet:
    @staticmethod
    def test_create_chat_completion_updates_cost():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]
        model = "bitnet-b1.58"
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = _mock_llm_response()

        with patch(
            "autogpt.llm.providers.openai.get_bitnet_llm", return_value=mock_llm
        ):
            response = openai.create_chat_completion(messages, model=model)

        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert api_manager.get_total_prompt_tokens() == 10
        assert api_manager.get_total_completion_tokens() == 20

    @staticmethod
    def test_create_chat_completion_empty_messages():
        messages = []
        model = "bitnet-b1.58"
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = _mock_llm_response(0, 0, "")

        with patch(
            "autogpt.llm.providers.openai.get_bitnet_llm", return_value=mock_llm
        ):
            openai.create_chat_completion(messages, model=model)

        assert api_manager.get_total_prompt_tokens() == 0
        assert api_manager.get_total_completion_tokens() == 0
        assert api_manager.get_total_cost() == 0
