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

from typing import Any

# Lazy loader that always tries SDK first, then falls back
class _SDKLazyLoader:
    """Lazy loader that always checks for SDK availability on each access."""

    def __getattr__(self, name: str) -> Any:
        """Always try to import from SDK first."""
        try:
            from victor_sdk.constants import ToolNames as _ToolNames
            # Return the actual attribute from the SDK
            return getattr(_ToolNames, name)
        except (ImportError, AttributeError):
            # SDK not available or attribute not found, provide fallback
            # Return a sentinel that will handle attribute access
            return _SDKFallback(name)

class _SDKFallback:
    """Fallback for when SDK is not available."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, name: str) -> Any:
        """Provide fallback values for common tool names."""
        # Return self for chained attribute access (e.g., ToolNames.READ.WRITE)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Handle if the fallback is called as a function."""
        return None

    def __eq__(self, other: Any) -> bool:
        """Handle equality comparisons."""
        return isinstance(other, str) and other == self._name

    def __hash__(self) -> int:
        """Make the fallback hashable."""
        return hash(self._name)

    def __repr__(self) -> str:
        """String representation."""
        return f"<ToolNames.{self._name} (SDK not available)>"

# Create lazy loader instance
_loader = _SDKLazyLoader()

# Export lazy loader as module-level names
ToolNames = _loader  # type: ignore

# Other exports that need lazy loading
class _ToolNameEntryLoader:
    def __getattr__(self, name: str) -> Any:
        try:
            from victor_sdk.constants import ToolNameEntry as _ToolNameEntry
            return getattr(_ToolNameEntry, name)
        except (ImportError, AttributeError):
            # Return a simple fallback
            return type("ToolNameEntry", (), {})

ToolNameEntry = _ToolNameEntryLoader()  # type: ignore

class _DictLoader:
    """Lazy loader for dict-like constants."""

    def __init__(self):
        self._dict = None

    def _load(self) -> dict:
        if self._dict is None:
            try:
                from victor_sdk.constants import (
                    CANONICAL_TO_ALIASES as _CANONICAL,
                    TOOL_ALIASES as _ALIASES,
                )
                self._dict = _CANONICAL
            except ImportError:
                self._dict = {}
        return self._dict

    def __getitem__(self, key: str) -> Any:
        return self._load().get(key)

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()

CANONICAL_TO_ALIASES = _DictLoader()  # type: ignore
TOOL_ALIASES = _DictLoader()  # type: ignore

# Functions that need lazy loading
def get_aliases(name: str) -> list:  # type: ignore
    """Get aliases for a tool name."""
    try:
        from victor_sdk.constants import get_aliases as _get_aliases
        return _get_aliases(name)
    except ImportError:
        return []

def get_all_canonical_names() -> list:  # type: ignore
    """Get all canonical tool names."""
    try:
        from victor_sdk.constants import get_all_canonical_names as _get_all
        return _get_all()
    except ImportError:
        return []

def get_canonical_name(name: str) -> str:  # type: ignore
    """Get canonical name for a tool."""
    try:
        from victor_sdk.constants import get_canonical_name as _get_canonical
        return _get_canonical(name)
    except ImportError:
        return name

def get_name_mapping() -> dict:  # type: ignore
    """Get tool name mapping."""
    try:
        from victor_sdk.constants import get_name_mapping as _get_mapping
        return _get_mapping()
    except ImportError:
        return {}

def is_valid_tool_name(name: str) -> bool:  # type: ignore
    """Check if a tool name is valid."""
    try:
        from victor_sdk.constants import is_valid_tool_name as _is_valid
        return _is_valid(name)
    except ImportError:
        return bool(name)

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
