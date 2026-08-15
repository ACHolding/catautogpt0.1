"""Unit tests for bitnet.cpp bootstrap helpers."""

from pathlib import Path

from autogpt.llm.providers import bitnet_cpp


def test_is_i2s_bitnet_gguf_detects_official_name():
    assert bitnet_cpp.is_i2s_bitnet_gguf(
        Path("/tmp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    )
    assert bitnet_cpp.is_i2s_bitnet_gguf(
        Path("/tmp/BitNet-b1.58-2B-4T/something.gguf")
    )
    assert not bitnet_cpp.is_i2s_bitnet_gguf(Path("/tmp/models/llama-q4.gguf"))


def test_find_llama_cli_respects_bitnet_cli(monkeypatch, tmp_path):
    cli = tmp_path / "llama-cli"
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    monkeypatch.setenv("BITNET_CLI", str(cli))
    monkeypatch.delenv("BITNET_HOME", raising=False)
    assert bitnet_cpp.find_llama_cli() == cli


def test_write_env_bitnet_home(tmp_path):
    env = tmp_path / ".env"
    env.write_text("BITNET_MODEL_PATH=/m.gguf\nFAST_LLM=bitnet-b1.58\n")
    home = tmp_path / "BitNet"
    bitnet_cpp.write_env_bitnet_home(home, env)
    text = env.read_text()
    assert f"BITNET_HOME={home}" in text
    assert "BITNET_BACKEND=bitnet.cpp" in text
    assert "BITNET_MODEL_PATH=/m.gguf" in text


def test_default_bitnet_home_avoids_hash_paths(monkeypatch, tmp_path):
    unsafe = tmp_path / "##Stuff:" / "proj"
    unsafe.mkdir(parents=True)
    monkeypatch.delenv("BITNET_HOME", raising=False)
    monkeypatch.setattr(bitnet_cpp, "project_root", lambda: unsafe)
    home = bitnet_cpp.default_bitnet_home()
    assert "#" not in str(home)
    assert ":" not in str(home)
    assert home == Path.home() / ".cache" / "catautogpt" / "BitNet"
