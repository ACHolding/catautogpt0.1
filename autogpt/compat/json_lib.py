"""JSON helpers with orjson when available, stdlib fallback otherwise.

Broken/outdated orjson wheels (e.g. ABI mismatches on Python 3.14) raise
ImportError on import; we catch that and fall back to the stdlib so memory
backends still start.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

try:
    import orjson as _orjson

    # Probe the extension module; a bad wheel can import the package shell
    # then fail when touching dumps/loads.
    _orjson.dumps([])
except Exception:  # noqa: BLE001 - any failure means use stdlib
    _orjson = None


if _orjson is not None:
    dumps = _orjson.dumps
    loads = _orjson.loads
    OPT_SERIALIZE_NUMPY = _orjson.OPT_SERIALIZE_NUMPY
    OPT_SERIALIZE_DATACLASS = _orjson.OPT_SERIALIZE_DATACLASS
else:
    OPT_SERIALIZE_NUMPY = 1
    OPT_SERIALIZE_DATACLASS = 2

    def _default(obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable"
        )

    def dumps(obj: Any, option: int = 0) -> bytes:  # noqa: ARG001
        return json.dumps(obj, default=_default, ensure_ascii=False).encode("utf-8")

    def loads(data: str | bytes | bytearray) -> Any:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return json.loads(data)
