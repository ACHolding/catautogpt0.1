"""tiktoken wrapper with a safe heuristic fallback for broken wheels."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

_tiktoken_mod: Any | None = None
_tiktoken_checked = False


class _HeuristicEncoding:
    """Rough token estimate (~4 chars/token) used when tiktoken is unavailable."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        n = max(1, (len(text) + 3) // 4)
        return list(range(n))


def _load_tiktoken() -> Any | None:
    """Import tiktoken only if its native extension works in a child process."""
    global _tiktoken_mod, _tiktoken_checked
    if _tiktoken_checked:
        return _tiktoken_mod
    _tiktoken_checked = True
    try:
        import tiktoken  # noqa: F401
    except Exception:  # noqa: BLE001
        _tiktoken_mod = None
        return None

    # ABI-mismatched wheels can segfault inside encode(); probe out-of-process.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tiktoken; tiktoken.get_encoding('cl100k_base').encode('hi')",
        ],
        capture_output=True,
        timeout=20,
        check=False,
    )
    if probe.returncode == 0:
        import tiktoken

        _tiktoken_mod = tiktoken
    else:
        _tiktoken_mod = None
    return _tiktoken_mod


def encoding_for_model(model_name: str) -> Any:
    tiktoken = _load_tiktoken()
    if tiktoken is not None:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    return _HeuristicEncoding()


def get_encoding(name: str) -> Any:
    tiktoken = _load_tiktoken()
    if tiktoken is not None:
        return tiktoken.get_encoding(name)
    return _HeuristicEncoding()
