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

"""Tool Alias Resolver - Resolves tool aliases to canonical names.

This module extracts tool alias resolution logic from ToolCoordinator,
following SRP (Single Responsibility Principle).

Responsibilities:
- Tool alias to canonical name resolution
- Shell variant selection (shell vs shell_readonly)
- Legacy name mapping
- Canonical name validation

Design Philosophy:
- Single Responsibility: Only handles alias resolution
- Configurable: Supports custom alias mappings
- Registry-aware: Checks tool registry for availability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolAliasConfig(BaseCoordinatorConfig):
    """Configuration for ToolAliasResolver.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        prefer_readonly_shell: Prefer shell_readonly over shell when both available
        strict_resolution: Raise error on unknown aliases (vs returning canonical)
    """

    prefer_readonly_shell: bool = False
    strict_resolution: bool = False


@dataclass
class ResolutionResult:
    """Result of alias resolution.

    Attributes:
        original: Original tool name/alias
        resolved: Resolved canonical name
        method: How resolution was done (direct, alias, shell_variant, canonical)
        is_legacy: Whether original was a legacy name
    """

    original: str
    resolved: str
    method: str
    is_legacy: bool = False


class ToolAliasResolver:
    """Resolver for tool aliases to canonical names.

    Extracts alias resolution logic from ToolCoordinator following SRP.
    This coordinator is responsible only for resolving tool names
    to their canonical forms.

    Example:
        resolver = ToolAliasResolver(
            tool_registry=registry,
            access_check=lambda n: n in enabled_tools
        )

        # Resolve alias
        result = resolver.resolve("bash")
        print(result.resolved)  # "shell" or "shell_readonly"

        # Check if name is alias
        if result.method != "direct":
            logger.info(f"Resolved alias {result.original} -> {result.resolved}")
    """

    # Shell aliases that map to shell variants
    SHELL_ALIASES = {
        "run",
        "bash",
        "execute",
        "cmd",
        "execute_bash",
        "shell_readonly",
        "shell",
    }

    # Legacy tool name mappings
    LEGACY_MAPPINGS: Dict[str, str] = {
        # File operations
        "read_file": "read",
        "write_file": "write",
        "edit_file": "edit",
        "list_files": "ls",
        "search_files": "grep",
        # Code operations
        "code_search": "code_search",  # Already canonical
        "semantic_search": "semantic_search",  # Already canonical
        # Web operations
        "web_search": "web_search",  # Already canonical
        "fetch_url": "web_fetch",
        # Execution
        "execute_command": "shell",
        "run_bash": "shell",
        # Git operations
        "git_command": "git",
    }

    def __init__(
        self,
        tool_registry: Optional["ToolRegistry"] = None,
        access_check: Optional[Callable[[str], bool]] = None,
        config: Optional[ToolAliasConfig] = None,
    ) -> None:
        """Initialize the alias resolver.

        Args:
            tool_registry: Tool registry for availability checking
            access_check: Function to check if tool is accessible
            config: Resolver configuration
        """
        self._registry = tool_registry
        self._access_check = access_check
        self._config = config or ToolAliasConfig()

        logger.debug(
            f"ToolAliasResolver initialized with "
            f"prefer_readonly={self._config.prefer_readonly_shell}"
        )

    # =====================================================================
    # Resolution Methods
    # =====================================================================

    def resolve(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Handles shell variants and legacy name mappings.

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            Canonical tool name
        """
        result = self.resolve_with_result(tool_name)
        return result.resolved

    def resolve_with_result(self, tool_name: str) -> ResolutionResult:
        """Resolve tool alias with detailed result.

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            ResolutionResult with resolution details
        """
        # Method 1: Direct return if not a known alias
        if tool_name not in self.SHELL_ALIASES:
            # Check for legacy mapping
            if tool_name in self.LEGACY_MAPPINGS:
                canonical = self.LEGACY_MAPPINGS[tool_name]
                return ResolutionResult(
                    original=tool_name,
                    resolved=canonical,
                    method="legacy",
                    is_legacy=True,
                )

            # Use get_canonical_name for other tools
            canonical = self._get_canonical_name(tool_name)
            if canonical != tool_name:
                return ResolutionResult(
                    original=tool_name,
                    resolved=canonical,
                    method="canonical",
                    is_legacy=True,
                )

            return ResolutionResult(
                original=tool_name,
                resolved=tool_name,
                method="direct",
                is_legacy=False,
            )

        # Method 2: Shell variant resolution
        from victor.tools.tool_names import ToolNames

        # Check if full shell is accessible
        use_full_shell = False

        if self._access_check:
            use_full_shell = self._access_check(ToolNames.SHELL)
        elif self._registry:
            use_full_shell = self._registry.is_tool_enabled(ToolNames.SHELL)

        if use_full_shell:
            logger.debug(f"Resolved '{tool_name}' to 'shell' (full shell enabled)")
            return ResolutionResult(
                original=tool_name,
                resolved=ToolNames.SHELL,
                method="shell_variant",
                is_legacy=False,
            )

        # Check if readonly shell is accessible
        use_readonly = False
        if self._access_check:
            use_readonly = self._access_check(ToolNames.SHELL_READONLY)
        elif self._registry:
            use_readonly = self._registry.is_tool_enabled(ToolNames.SHELL_READONLY)

        if use_readonly:
            logger.debug(f"Resolved '{tool_name}' to 'shell_readonly' (readonly mode)")
            return ResolutionResult(
                original=tool_name,
                resolved=ToolNames.SHELL_READONLY,
                method="shell_variant",
                is_legacy=False,
            )

        # Method 3: Fall back to canonical (will fail validation)
        canonical = self._get_canonical_name(tool_name)
        logger.debug(f"No shell variant enabled for '{tool_name}', using canonical '{canonical}'")

        return ResolutionResult(
            original=tool_name,
            resolved=canonical,
            method="canonical",
            is_legacy=False,
        )

    def is_shell_alias(self, tool_name: str) -> bool:
        """Check if a name is a shell alias.

        Args:
            tool_name: Name to check

        Returns:
            True if name is a shell alias
        """
        return tool_name in self.SHELL_ALIASES

    def is_legacy_name(self, tool_name: str) -> bool:
        """Check if a name is a legacy tool name.

        Args:
            tool_name: Name to check

        Returns:
            True if name is a legacy name
        """
        return tool_name in self.LEGACY_MAPPINGS

    # =====================================================================
    # Configuration
    # =====================================================================

    def set_access_check(self, access_check: Callable[[str], bool]) -> None:
        """Set the access check function.

        Args:
            access_check: Function that returns True if tool is accessible
        """
        self._access_check = access_check

    def set_prefer_readonly(self, prefer: bool) -> None:
        """Set preference for readonly shell.

        Args:
            prefer: True to prefer shell_readonly over shell
        """
        self._config.prefer_readonly_shell = prefer
        logger.debug(f"Prefer readonly shell set to: {prefer}")

    # =====================================================================
    # Internal Helpers
    # =====================================================================

    def _get_canonical_name(self, tool_name: str) -> str:
        """Get canonical name from tool_names module.

        Args:
            tool_name: Tool name to canonicalize

        Returns:
            Canonical tool name
        """
        try:
            from victor.tools.tool_names import get_canonical_name

            return get_canonical_name(tool_name)
        except Exception:
            # Fall back to original name if canonicalization fails
            return tool_name

    def add_legacy_mapping(self, legacy: str, canonical: str) -> None:
        """Add a custom legacy name mapping.

        Args:
            legacy: Legacy tool name
            canonical: Canonical tool name
        """
        self.LEGACY_MAPPINGS[legacy] = canonical
        logger.debug(f"Added legacy mapping: {legacy} -> {canonical}")


def create_tool_alias_resolver(
    tool_registry: Optional["ToolRegistry"] = None,
    access_check: Optional[Callable[[str], bool]] = None,
    prefer_readonly: bool = False,
) -> ToolAliasResolver:
    """Factory function to create a ToolAliasResolver.

    Args:
        tool_registry: Tool registry for availability checking
        access_check: Function to check if tool is accessible
        prefer_readonly: Prefer shell_readonly over shell

    Returns:
        Configured ToolAliasResolver instance
    """
    config = ToolAliasConfig(prefer_readonly_shell=prefer_readonly)

    return ToolAliasResolver(
        tool_registry=tool_registry,
        access_check=access_check,
        config=config,
    )


__all__ = [
    "ToolAliasResolver",
    "ToolAliasConfig",
    "ResolutionResult",
    "create_tool_alias_resolver",
]
