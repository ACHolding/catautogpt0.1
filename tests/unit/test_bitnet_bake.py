"""Tests for BitNet auto-bake helpers (no network)."""

from pathlib import Path

from autogpt.llm.providers import bitnet_bake


def test_find_local_gguf_prefers_i2_s(tmp_path: Path):
    (tmp_path / "other.gguf").write_bytes(b"x" * 2_000_000)
    target = tmp_path / "ggml-model-i2_s.gguf"
    target.write_bytes(b"y" * 2_000_000)
    assert bitnet_bake.find_local_gguf(tmp_path) == target


def test_find_local_gguf_ignores_tiny_files(tmp_path: Path):
    (tmp_path / "ggml-model-i2_s.gguf").write_bytes(b"tiny")
    assert bitnet_bake.find_local_gguf(tmp_path) is None


def test_write_env_model_path_upsert(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("FAST_LLM=bitnet-b1.58\nBITNET_MODEL_PATH=old\n")
    gguf = tmp_path / "model.gguf"
    bitnet_bake.write_env_model_path(gguf, env)
    text = env.read_text()
    assert f"BITNET_MODEL_PATH={gguf}" in text
    assert "FAST_LLM=bitnet-b1.58" in text
    assert "BITNET_MODEL_PATH=old" not in text


def test_ensure_bitnet_gguf_uses_existing(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "BitNet"
    model_dir.mkdir()
    gguf = model_dir / "ggml-model-i2_s.gguf"
    gguf.write_bytes(b"z" * 2_000_000)
    monkeypatch.setenv("BITNET_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("BITNET_AUTO_DOWNLOAD", "False")
    assert bitnet_bake.ensure_bitnet_gguf(project_root=tmp_path) == gguf
