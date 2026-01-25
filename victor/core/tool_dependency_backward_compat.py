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

"""Unified backward compatibility for tool dependency providers.

This module provides backward compatibility for deprecated tool dependency
providers while consolidating all vertical-specific logic into the framework.

Migration path:
1. Replace vertical-specific provider imports with factory function
2. Use create_vertical_tool_dependency_provider() instead
3. Remove deprecated provider classes from individual verticals

Example:
    # Old (deprecated):
    from victor.coding.tool_dependencies import CodingToolDependencyProvider
    provider = CodingToolDependencyProvider()

    # New (recommended):
    from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
    provider = create_vertical_tool_dependency_provider("coding")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    from victor.core.tool_dependency_loader import (
        YAMLToolDependencyProvider,
    )
    from victor.core.tool_types import EmptyToolDependencyProvider

from victor.core.tool_dependency_loader import (
    YAMLToolDependencyProvider,
    create_vertical_tool_dependency_provider,
)


class DeprecatedToolDependencyProvider(YAMLToolDependencyProvider):
    """Deprecated provider wrapper for backward compatibility.

    This class provides a unified deprecation path for all vertical-specific
    tool dependency providers. It wraps the framework factory function
    while emitting deprecation warnings.

    .. deprecated::
        Use ``create_vertical_tool_dependency_provider(vertical)`` instead.
    """

    def __init__(self, vertical: str, yaml_path: Optional[Path] = None, canonicalize: bool = True):
        """Initialize deprecated provider.

        Args:
            vertical: Vertical name (coding, devops, rag, research, dataanalysis)
            yaml_path: Optional custom YAML path (defaults to standard location)
            canonicalize: Whether to canonicalize tool IDs
        """
        self._vertical = vertical
        self._yaml_path = yaml_path  # type: ignore[assignment]
        self._canonicalize = canonicalize

        warnings.warn(
            f"{vertical.capitalize()}ToolDependencyProvider is deprecated. "
            f"Use create_vertical_tool_dependency_provider('{vertical}') instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Determine YAML path
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / vertical / "tool_dependencies.yaml"

        super().__init__(yaml_path=yaml_path, canonicalize=canonicalize)

    def __class_getitem__(cls, item: Any) -> type:
        """Support type hints like DeprecatedToolDependencyProvider[SomeType]."""
        return cls  # pragma: no cover


def create_deprecated_provider(vertical: str) -> Union["YAMLToolDependencyProvider", "EmptyToolDependencyProvider"]:
    """Create a deprecated provider instance for backward compatibility.

    This function is used internally to maintain backward compatibility while
    migrating to the unified factory function.

    Args:
        vertical: Vertical name (coding, devops, rag, research, dataanalysis)

    Returns:
        Provider instance created via framework factory

    Example:
        >>> provider = create_deprecated_provider("coding")
        >>> dependencies = provider.get_dependencies("read_file")
    """
    return create_vertical_tool_dependency_provider(vertical)


# Vertical-specific aliases for backward compatibility
# These will be phased out in favor of direct factory usage

VERTICAL_DEPRECATED_ALIASES: Dict[str, str] = {
    "coding": "CodingToolDependencyProvider",
    "devops": "DevOpsToolDependencyProvider",
    "rag": "RAGToolDependencyProvider",
    "research": "ResearchToolDependencyProvider",
    "dataanalysis": "DataAnalysisToolDependencyProvider",
}


def get_deprecated_class_name(vertical: str) -> str:
    """Get the deprecated class name for a vertical.

    Args:
        vertical: Vertical name

    Returns:
        Deprecated class name

    Raises:
        ValueError: If vertical is not recognized
    """
    if vertical not in VERTICAL_DEPRECATED_ALIASES:
        raise ValueError(f"Unknown vertical: {vertical}. Must be one of {list(VERTICAL_DEPRECATED_ALIASES.keys())}")
    return VERTICAL_DEPRECATED_ALIASES[vertical]


def is_deprecated_provider_vertical(vertical: str) -> bool:
    """Check if a vertical has a deprecated provider class.

    Args:
        vertical: Vertical name to check

    Returns:
        True if the vertical has a deprecated provider class
    """
    return vertical in VERTICAL_DEPRECATED_ALIASES


# Migration helper functions

def suggest_migration_code(vertical: str) -> str:
    """Generate migration code snippet for a vertical.

    Args:
        vertical: Vertical name

    Returns:
        Code snippet showing migration path
    """
    class_name = get_deprecated_class_name(vertical)
    return f"""
# Old (deprecated):
from victor.{vertical}.tool_dependencies import {class_name}
provider = {class_name}()

# New (recommended):
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
provider = create_vertical_tool_dependency_provider("{vertical}")
""".strip()


def emit_deprecation_warning(vertical: str, stacklevel: int = 3) -> None:
    """Emit a deprecation warning for a vertical-specific provider.

    Args:
        vertical: Vertical name
        stacklevel: Stack level for warning (default 3 for direct usage)
    """
    class_name = get_deprecated_class_name(vertical)
    warnings.warn(
        f"{class_name} is deprecated and will be removed in a future version. "
        f"Use create_vertical_tool_dependency_provider('{vertical}') instead. "
        f"See victor.core.tool_dependency_backwardCompat for migration guide.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
