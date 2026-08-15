"""Unit tests for CatSeek-GPU fast prompt path (no GGUF required)."""

from unittest.mock import MagicMock, patch

from autogpt.llm.providers import catseek_engine


def test_apply_deepseek_chat_template_prefill_empty_think():
    prompt = catseek_engine.apply_deepseek_chat_template(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        prefill=catseek_engine.EMPTY_THINK_PREFILL,
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n</think>\n")
    assert "You are helpful." in prompt


def test_fast_mode_prefills_empty_think(monkeypatch, tmp_path):
    monkeypatch.setenv("CATSEEK_FAST", "1")
    monkeypatch.setenv("CATSEEK_MAX_TOKENS", "64")
    mock_llm = MagicMock()
    mock_llm.tokenize.return_value = [1, 2, 3, 4, 5]
    mock_llm.return_value = {
        "choices": [{"text": "Name: FastGPT\nDescription: speedy\nGoals:\n- Go"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }

    with patch.object(catseek_engine, "get_chat_llm", return_value=mock_llm), patch.object(
        catseek_engine, "_append_trace"
    ), patch.object(catseek_engine, "workspace_root", return_value=tmp_path):
        result = catseek_engine.create_chat_completion_raw(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=256,
        )

    assert "FastGPT" in result.choices[0].message["content"]
    mock_llm.assert_called_once()
    prompt = mock_llm.call_args.args[0]
    assert catseek_engine.EMPTY_THINK_PREFILL in prompt
    # Cap to CATSEEK_MAX_TOKENS.
    assert mock_llm.call_args.kwargs["max_tokens"] == 64
    assert (tmp_path / "latest_reply.txt").read_text().startswith("Name: FastGPT")


def test_slow_mode_uses_chat_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("CATSEEK_FAST", "0")
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }

    with patch.object(catseek_engine, "get_chat_llm", return_value=mock_llm), patch.object(
        catseek_engine, "_append_trace"
    ), patch.object(catseek_engine, "workspace_root", return_value=tmp_path):
        result = catseek_engine.create_chat_completion_raw(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=32,
        )

    assert result.choices[0].message["content"] == "hello"
    mock_llm.create_chat_completion.assert_called_once()
    mock_llm.assert_not_called()


def test_strip_think_helper():
    assert catseek_engine._strip_think("<think>plan</think>\nName: X") == "Name: X"
