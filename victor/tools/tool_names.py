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

"""Compatibility wrapper for the SDK-owned tool naming registry.

New definition-layer code should import these symbols from `victor_sdk`
or `victor_sdk.constants`. This module remains for backward compatibility
with existing victor-ai callers during the migration.
"""

from __future__ import annotations

from typing import Any, Optional

# Try to import from SDK first
try:
    from victor_sdk.constants import (
        CANONICAL_TO_ALIASES,
        TOOL_ALIASES,
        ToolNameEntry,
        ToolNames,
        get_aliases,
        get_all_canonical_names,
        get_canonical_name,
        get_name_mapping,
        is_valid_tool_name,
    )
    _SDK_AVAILABLE = True
except ImportError:
    # SDK not installed yet (e.g., during victor-ai installation)
    # Create lazy loader that will import from SDK when first accessed
    _SDK_AVAILABLE = False

    # Lazy loader for SDK constants
    _SDK_CONSTANTS: Optional[dict] = None

    def _get_sdk_constants() -> dict:
        """Lazy load SDK constants when first accessed."""
        global _SDK_CONSTANTS
        if _SDK_CONSTANTS is None:
            try:
                from victor_sdk.constants import (
                    CANONICAL_TO_ALIASES,
                    TOOL_ALIASES,
                    ToolNameEntry,
                    ToolNames,
                    get_aliases,
                    get_all_canonical_names,
                    get_canonical_name,
                    get_name_mapping,
                    is_valid_tool_name,
                )
                _SDK_CONSTANTS = {
                    "CANONICAL_TO_ALIASES": CANONICAL_TO_ALIASES,
                    "TOOL_ALIASES": TOOL_ALIASES,
                    "ToolNameEntry": ToolNameEntry,
                    "ToolNames": ToolNames,
                    "get_aliases": get_aliases,
                    "get_all_canonical_names": get_all_canonical_names,
                    "get_canonical_name": get_canonical_name,
                    "get_name_mapping": get_name_mapping,
                    "is_valid_tool_name": is_valid_tool_name,
                }
            except ImportError:
                # SDK still not available, provide minimal fallback
                _SDK_CONSTANTS = {
                    "CANONICAL_TO_ALIASES": {},
                    "TOOL_ALIASES": {},
                    "ToolNameEntry": None,
                    "ToolNames": type("ToolNames", (), {}),
                    "get_aliases": lambda name: [],
                    "get_all_canonical_names": lambda: [],
                    "get_canonical_name": lambda name: name,
                    "get_name_mapping": lambda: {},
                    "is_valid_tool_name": lambda name: bool(name),
                }
        return _SDK_CONSTANTS

    class _LazyLoader:
        """Lazy loader for SDK constants that forwards to the real implementation."""

        def __getattr__(self, name: str) -> Any:
            return _get_sdk_constants()[name]

    # Create lazy loader instance
    _lazy_loader = _LazyLoader()

    # Export lazy loader as module-level names
    ToolNames = _lazy_loader.ToolNames  # type: ignore
    ToolNameEntry = _lazy_loader.ToolNameEntry  # type: ignore
    CANONICAL_TO_ALIASES = _lazy_loader.CANONICAL_TO_ALIASES  # type: ignore
    TOOL_ALIASES = _lazy_loader.TOOL_ALIASES  # type: ignore
    get_aliases = _lazy_loader.get_aliases  # type: ignore
    get_all_canonical_names = _lazy_loader.get_all_canonical_names  # type: ignore
    get_canonical_name = _lazy_loader.get_canonical_name  # type: ignore
    get_name_mapping = _lazy_loader.get_name_mapping  # type: ignore
    is_valid_tool_name = _lazy_loader.is_valid_tool_name  # type: ignore

__all__ = [
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
]
