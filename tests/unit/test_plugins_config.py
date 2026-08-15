"""Tests for plugins_config path coercion (str vs Path crash fix)."""

from pathlib import Path

from autogpt.plugins.plugins_config import PluginsConfig


def test_load_config_accepts_string_path(tmp_path: Path):
    cfg_file = tmp_path / "plugins_config.yaml"
    cfg_file.write_text("{}\n")

    # Historically crashed with: 'str' object has no attribute 'is_file'
    loaded = PluginsConfig.load_config(
        plugins_config_file=str(cfg_file),
        plugins_denylist=[],
        plugins_allowlist=[],
    )
    assert loaded.plugins == {}


def test_load_config_creates_missing_file(tmp_path: Path):
    cfg_file = tmp_path / "nested" / "plugins_config.yaml"
    loaded = PluginsConfig.load_config(
        plugins_config_file=cfg_file,
        plugins_denylist=["bad_plugin"],
        plugins_allowlist=["good_plugin"],
    )
    assert cfg_file.is_file()
    assert loaded.is_enabled("good_plugin") is True
    assert loaded.is_enabled("bad_plugin") is False
