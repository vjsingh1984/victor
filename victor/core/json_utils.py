# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast JSON serialization facade.

Uses ``orjson`` (Rust-backed, ~5-10x faster) when available, with
automatic fallback to stdlib ``json``.

Usage::

    from victor.core.json_utils import json_dumps, json_loads, json_dumps_bytes

    text = json_dumps({"key": "value"})            # -> str
    raw  = json_dumps_bytes({"key": "value"})       # -> bytes  (zero-copy with orjson)
    obj  = json_loads(text)                          # -> Any
    obj  = json_loads(raw)                           # -> Any  (bytes accepted)

All public functions are API-compatible with stdlib ``json`` for the
common case; callers should not need to change signatures.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_ORJSON_AVAILABLE = False
_orjson: Any = None

try:
    import orjson as _orjson  # type: ignore[no-redef]

    _ORJSON_AVAILABLE = True
except ImportError:
    pass


def is_orjson_available() -> bool:
    """Return True if orjson is importable."""
    return _ORJSON_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = False,
    indent: bool = False,
    ensure_ascii: bool = False,
    default: Optional[Any] = None,
) -> str:
    """Serialize *obj* to a JSON ``str``.

    Drop-in replacement for ``json.dumps()``.
    """
    if _ORJSON_AVAILABLE:
        opts = 0
        if sort_keys:
            opts |= _orjson.OPT_SORT_KEYS
        if indent:
            opts |= _orjson.OPT_INDENT_2
        if not ensure_ascii:
            opts |= _orjson.OPT_NON_STR_KEYS
        return _orjson.dumps(obj, option=opts or None, default=default).decode("utf-8")

    return _json.dumps(
        obj,
        sort_keys=sort_keys,
        indent=2 if indent else None,
        ensure_ascii=ensure_ascii,
        default=default,
    )


def json_dumps_bytes(
    obj: Any,
    *,
    sort_keys: bool = False,
    default: Optional[Any] = None,
) -> bytes:
    """Serialize *obj* to JSON ``bytes`` (zero-copy with orjson)."""
    if _ORJSON_AVAILABLE:
        opts = 0
        if sort_keys:
            opts |= _orjson.OPT_SORT_KEYS
        return _orjson.dumps(obj, option=opts or None, default=default)

    return _json.dumps(obj, sort_keys=sort_keys, default=default).encode("utf-8")


def json_loads(data: Union[str, bytes, bytearray, memoryview]) -> Any:
    """Deserialize JSON ``str`` or ``bytes`` to a Python object."""
    if _ORJSON_AVAILABLE:
        return _orjson.loads(data)
    return _json.loads(data)
