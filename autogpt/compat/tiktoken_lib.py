"""tiktoken wrapper with a safe heuristic fallback for broken wheels.

On Python 3.14 the currently published ``tiktoken`` wheels segfault inside the
native ``_tiktoken`` extension (CoreBPE). We never import that native module on
3.14+ — probing it in a subprocess still triggers macOS Crash Reporter.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_tiktoken_mod: Any | None = None
_tiktoken_checked = False

# Hard-disable native tiktoken on 3.14+; optional override for when a fixed
# wheel exists: AUTO_GPT_ALLOW_TIKTOKEN=1
_DISABLE_NATIVE_ON = (3, 14)


class _HeuristicEncoding:
    """Rough token estimate (~4 chars/token) used when tiktoken is unavailable."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        n = max(1, (len(text) + 3) // 4)
        return list(range(n))

    def decode(self, tokens: list[int]) -> str:
        # Best-effort reverse for chunking paths that round-trip encode/decode.
        # Without a real BPE table we cannot recover text from synthetic ids, so
        # callers that need decode should pass the original string themselves.
        # For chunk_content we encode the full string then slice token ids —
        # decode must return *something* length-proportional. Use placeholders
        # only if we have no attached source text.
        source = getattr(self, "_source_text", None)
        if isinstance(source, str) and source:
            # Map token index ranges back onto character spans (~4 chars/token).
            if not tokens:
                return ""
            start = tokens[0] * 4
            end = (tokens[-1] + 1) * 4
            return source[start:end]
        return " " * max(0, len(tokens) * 4)


class _SourceAwareHeuristicEncoding(_HeuristicEncoding):
    """Heuristic encoder that can decode slices back into the original text."""

    def __init__(self, source_text: str | None = None):
        self._source_text = source_text

    def encode(self, text: str) -> list[int]:
        self._source_text = text
        return super().encode(text)

    def decode(self, tokens: list[int]) -> str:
        source = self._source_text or ""
        if not tokens:
            return ""
        if not source:
            return super().decode(tokens)
        # Character-aligned windows matching encode()'s ~4 chars/token mapping.
        start = max(0, tokens[0] * 4)
        end = min(len(source), (tokens[-1] + 1) * 4)
        return source[start:end]


def _native_tiktoken_allowed() -> bool:
    if os.getenv("AUTO_GPT_FORCE_HEURISTIC_TIKTOKEN", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return False
    if sys.version_info >= _DISABLE_NATIVE_ON:
        # Opt-in only: a future fixed wheel can be enabled explicitly.
        return os.getenv("AUTO_GPT_ALLOW_TIKTOKEN", "").lower() in {
            "1",
            "true",
            "yes",
        }
    return True


def _load_tiktoken() -> Any | None:
    """Return the tiktoken module, or None when native support is unsafe/unavailable."""
    global _tiktoken_mod, _tiktoken_checked
    if _tiktoken_checked:
        return _tiktoken_mod
    _tiktoken_checked = True

    if not _native_tiktoken_allowed():
        _tiktoken_mod = None
        return None

    try:
        import tiktoken
    except Exception:  # noqa: BLE001
        _tiktoken_mod = None
        return None

    # Lightweight in-process smoke test wrapped tightly. Do NOT spawn a child:
    # a segfaulting child still pops macOS Crash Reporter even when handled.
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        assert enc.encode("hi")
        _tiktoken_mod = tiktoken
    except Exception:  # noqa: BLE001
        _tiktoken_mod = None
    return _tiktoken_mod


def encoding_for_model(model_name: str) -> Any:
    tiktoken = _load_tiktoken()
    if tiktoken is not None:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    return _SourceAwareHeuristicEncoding()


def get_encoding(name: str) -> Any:
    tiktoken = _load_tiktoken()
    if tiktoken is not None:
        return tiktoken.get_encoding(name)
    return _SourceAwareHeuristicEncoding()
