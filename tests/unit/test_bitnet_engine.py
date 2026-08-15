"""Unit tests for the BitNet engine (no GGUF required)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from autogpt.llm.providers import bitnet_engine


def test_llama3_chat_template_includes_roles_and_generation_prompt():
    prompt = bitnet_engine.apply_llama3_chat_template(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
    )
    # BOS omitted by default (llama-completion injects it).
    assert "<|begin_of_text|>" not in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You are helpful." in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_requires_bitnet_cpp_for_i2s(monkeypatch, tmp_path):
    gguf = tmp_path / "ggml-model-i2_s.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.delenv("BITNET_FORCE_LLAMA_CPP", raising=False)
    assert bitnet_engine.requires_bitnet_cpp(gguf) is True


def test_create_chat_completion_uses_cli_for_i2s(monkeypatch, tmp_path):
    gguf = tmp_path / "ggml-model-i2_s.gguf"
    gguf.write_bytes(b"x")
    cli = tmp_path / "llama-cli"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.setenv("BITNET_MODEL_PATH", str(gguf))
    monkeypatch.setenv("BITNET_CLI", str(cli))
    monkeypatch.setenv("BITNET_BACKEND", "auto")

    with patch.object(
        bitnet_engine,
        "_complete_via_bitnet_cli",
        return_value=("via-cli", 2, 3),
    ) as cli_mock, patch.object(
        bitnet_engine, "_complete_via_llama_cpp"
    ) as llama_mock, patch.object(
        bitnet_engine, "ensure_chat_model_path", return_value=gguf
    ):
        result = bitnet_engine.create_chat_completion_raw(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=16,
        )

    assert result.choices[0].message["content"] == "via-cli"
    cli_mock.assert_called_once()
    llama_mock.assert_not_called()


def test_create_chat_completion_uses_llama_cpp_backend(monkeypatch):
    monkeypatch.setenv("BITNET_BACKEND", "llama-cpp")
    monkeypatch.setenv("BITNET_FORCE_LLAMA_CPP", "1")
    mock_llm = MagicMock()
    mock_llm.tokenize.return_value = [1, 2, 3, 4]
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hello from bitnet"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3},
    }

    with patch.object(bitnet_engine, "get_chat_llm", return_value=mock_llm), patch.object(
        bitnet_engine, "resolve_chat_model_path", return_value=MagicMock()
    ), patch.object(bitnet_engine, "ensure_chat_model_path", return_value=Path("/tmp/other.gguf")), patch.object(
        bitnet_engine, "requires_bitnet_cpp", return_value=False
    ), patch.object(bitnet_engine, "find_bitnet_cli", return_value=None):
        result = bitnet_engine.create_chat_completion_raw(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=32,
        )

    assert result.choices[0].message["content"] == "hello from bitnet"
    assert result.usage.prompt_tokens == 4
    assert result.usage.completion_tokens == 3
    mock_llm.create_chat_completion.assert_called_once()
    kwargs = mock_llm.create_chat_completion.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_k"] == 1
    assert "<|eot_id|>" in kwargs["stop"]


def test_create_embedding_hash_fallback(monkeypatch):
    monkeypatch.setenv("BITNET_EMBED_DIMS", "8")
    with patch.object(bitnet_engine, "get_embed_llm", return_value=None), patch.object(
        bitnet_engine, "get_chat_llm", side_effect=RuntimeError("no model")
    ):
        result = bitnet_engine.create_embedding_raw(["alpha", "beta"])

    assert len(result.data) == 2
    assert len(result.data[0]["embedding"]) == 8
    # Deterministic for same input.
    again = bitnet_engine.create_embedding_raw(["alpha"])
    assert result.data[0]["embedding"] == again.data[0]["embedding"]


def test_provider_create_chat_completion_updates_api_manager(monkeypatch):
    from autogpt.llm.api_manager import ApiManager
    from autogpt.llm.providers import openai as provider

    if ApiManager in ApiManager._instances:
        del ApiManager._instances[ApiManager]
    api = ApiManager()

    fake = SimpleNamespace(
        model="bitnet-b1.58",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        choices=[SimpleNamespace(message={"role": "assistant", "content": "ok"})],
    )
    monkeypatch.setattr(
        "autogpt.llm.providers.bitnet_engine.create_chat_completion_raw",
        lambda *a, **k: fake,
    )
    out = provider.create_chat_completion(
        [{"role": "user", "content": "x"}], model="bitnet-b1.58", max_tokens=16
    )
    assert out.choices[0].message["content"] == "ok"
    assert api.get_total_prompt_tokens() == 11
    assert api.get_total_completion_tokens() == 7
