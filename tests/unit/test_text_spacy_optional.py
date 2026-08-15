"""Tests for spaCy-optional text splitting (Python 3.14 / broken numpy)."""

from autogpt.processing import text as text_mod


def test_split_sentences_regex_fallback_when_spacy_missing(monkeypatch):
    monkeypatch.setattr(text_mod, "_SPACY", None)
    monkeypatch.setattr(text_mod, "_SPACY_LOAD_TRIED", True)
    sentences = text_mod._split_sentences(
        "Hello world. How are you? Fine!", "en_core_web_sm"
    )
    assert "Hello world." in sentences
    assert any("How are you?" in s for s in sentences)


def test_text_module_imports_without_loading_spacy():
    """Regression: top-level spaCy import used to crash Auto-GPT on Py3.14."""
    # Import path used by memory/vector → processing.text must not require spaCy.
    from autogpt.processing.text import chunk_content, split_text

    assert callable(chunk_content)
    assert callable(split_text)
