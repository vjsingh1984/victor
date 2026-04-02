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
    # Provide fallback implementations to prevent import errors
    _SDK_AVAILABLE = False

    # Fallback type definitions
    from typing import Protocol
    from typing_extensions import runtime_checkable

    @runtime_checkable
    class ToolNames(Protocol):
        """Protocol for tool names when SDK is not available."""

    @runtime_checkable
    class ToolNameEntry(Protocol):
        """Protocol for tool name entry when SDK is not available."""

    # Fallback values
    CANONICAL_TO_ALIASES = {}  # type: ignore
    TOOL_ALIASES = {}  # type: ignore

    def get_aliases(name: str):  # type: ignore
        """Fallback implementation when SDK is not available."""
        return []

    def get_all_canonical_names():  # type: ignore
        """Fallback implementation when SDK is not available."""
        return []

    def get_canonical_name(name: str):  # type: ignore
        """Fallback implementation when SDK is not available."""
        return name

    def get_name_mapping():  # type: ignore
        """Fallback implementation when SDK is not available."""
        return {}

    def is_valid_tool_name(name: str) -> bool:  # type: ignore
        """Fallback implementation when SDK is not available."""
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
