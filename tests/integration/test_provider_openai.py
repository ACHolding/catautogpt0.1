from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autogpt.llm.api_manager import ApiManager
from autogpt.llm.providers import openai

api_manager = ApiManager()


@pytest.fixture(autouse=True)
def reset_api_manager():
    api_manager.reset()
    yield


def _fake_response(prompt_tokens=10, completion_tokens=20, content="ok"):
    return SimpleNamespace(
        model="catseek-gpu-0.1",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
        choices=[SimpleNamespace(message={"role": "assistant", "content": content})],
    )


class TestProviderCatSeek:
    @staticmethod
    def test_create_chat_completion_updates_cost():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]
        model = "catseek-gpu-0.1"

        with patch(
            "autogpt.llm.providers.catseek_engine.create_chat_completion_raw",
            return_value=_fake_response(),
        ):
            response = openai.create_chat_completion(messages, model=model)

        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert api_manager.get_total_prompt_tokens() == 10
        assert api_manager.get_total_completion_tokens() == 20

    @staticmethod
    def test_create_chat_completion_empty_messages():
        messages = []
        model = "catseek-gpu-0.1"

        with patch(
            "autogpt.llm.providers.catseek_engine.create_chat_completion_raw",
            return_value=_fake_response(0, 0, ""),
        ):
            openai.create_chat_completion(messages, model=model)

        assert api_manager.get_total_prompt_tokens() == 0
        assert api_manager.get_total_completion_tokens() == 0
        assert api_manager.get_total_cost() == 0
