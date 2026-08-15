import pytest

from autogpt.app.setup import _parse_aiconfig_output


def test_parse_aiconfig_output_happy_path():
    cfg = _parse_aiconfig_output(
        """Name: TestGPT
Description: a helpful test agent
Goals:
- Goal one
- Goal two
"""
    )
    assert cfg.ai_name == "TestGPT"
    assert cfg.ai_role == "a helpful test agent"
    assert cfg.ai_goals == ["Goal one", "Goal two"]


def test_parse_aiconfig_output_strips_think_blocks():
    cfg = _parse_aiconfig_output(
        """<think>
I will invent a good name and goals for this task.
</think>
Name: TestGPT
Description: a helpful test agent
Goals:
- Goal one
"""
    )
    assert cfg.ai_name == "TestGPT"
    assert cfg.ai_goals == ["Goal one"]


def test_parse_aiconfig_output_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        _parse_aiconfig_output("")


def test_parse_aiconfig_output_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing Name/Description"):
        _parse_aiconfig_output("just some prose without the required fields")
