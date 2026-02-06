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

"""Dynamic capability definitions for the coding vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the coding vertical with custom functionality.

Refactored to use BaseVerticalCapabilityProvider, reducing from
690 lines to ~200 lines by eliminating duplicated patterns.

Example:
    # Use provider
    from victor.coding.capabilities import CodingCapabilityProvider

    provider = CodingCapabilityProvider()

    # Apply capabilities
    provider.apply_git_safety(orchestrator, block_force_push=True)
    provider.apply_code_style(orchestrator, formatter="black")

    # Get configurations
    style = provider.get_capability_config(orchestrator, "code_style")
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.framework.capabilities.base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
)
from victor.framework.protocols import CapabilityType
from victor.framework.capability_loader import CapabilityEntry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers (configure_*, get_* functions)
# =============================================================================


def configure_git_safety(
    orchestrator: Any,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    allowed_branches: Optional[list[str]] = None,
) -> None:
    """Configure git safety rules for the orchestrator.

    This capability configures the orchestrator's git safety
    checks to prevent dangerous operations.

    Args:
        orchestrator: Target orchestrator
        block_force_push: Block git push --force
        block_main_push: Block direct push to main/master
        require_tests_before_commit: Require tests pass before commit
        allowed_branches: Whitelist of branches for push
    """
    from victor.coding.safety import CodingSafetyExtension

    safety = CodingSafetyExtension()

    # Configure patterns
    if block_force_push:
        safety.add_dangerous_pattern(r"git\s+push\s+.*--force")
        safety.add_dangerous_pattern(r"git\s+push\s+-f")

    if block_main_push:
        safety.add_dangerous_pattern(r"git\s+push\s+.*\b(main|master)\b")

    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    git_safety_config = {
        "require_tests_before_commit": require_tests_before_commit,
        "allowed_branches": allowed_branches or [],
    }
    context.set_capability_config("git_safety", git_safety_config)

    logger.info("Configured git safety rules")


def configure_code_style(
    orchestrator: Any,
    *,
    formatter: str = "black",
    linter: str = "ruff",
    max_line_length: int = 100,
    enforce_type_hints: bool = True,
) -> None:
    """Configure code style preferences for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        formatter: Code formatter to use (black, autopep8, yapf)
        linter: Linter to use (ruff, flake8, pylint)
        max_line_length: Maximum line length
        enforce_type_hints: Whether to enforce type hints
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    code_style_config = {
        "formatter": formatter,
        "linter": linter,
        "max_line_length": max_line_length,
        "enforce_type_hints": enforce_type_hints,
    }
    context.set_capability_config("code_style", code_style_config)

    logger.info(f"Configured code style: formatter={formatter}, linter={linter}")


def get_code_style(orchestrator: Any) -> dict[str, Any]:
    """Get current code style configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Code style configuration dict
    """
    # SOLID DIP: Read from VerticalContext instead of direct attribute access
    context = orchestrator.vertical_context
    result = context.get_capability_config(
        "code_style",
        {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
            "enforce_type_hints": True,
        },
    )
    return result  # type: ignore[no-any-return]


def configure_test_requirements(
    orchestrator: Any,
    *,
    min_coverage: float = 0.0,
    required_test_patterns: Optional[list[str]] = None,
    test_framework: str = "pytest",
    run_tests_on_edit: bool = False,
) -> None:
    """Configure test requirements for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        min_coverage: Minimum code coverage percentage
        required_test_patterns: Patterns tests must match
        test_framework: Test framework to use
        run_tests_on_edit: Automatically run tests after edits
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    test_config = {
        "min_coverage": min_coverage,
        "required_patterns": required_test_patterns or [],
        "framework": test_framework,
        "run_on_edit": run_tests_on_edit,
    }
    context.set_capability_config("test_requirements", test_config)

    logger.info(f"Configured test requirements: framework={test_framework}")


def configure_language_server(
    orchestrator: Any,
    *,
    languages: Optional[list[str]] = None,
    enable_hover: bool = True,
    enable_references: bool = True,
    enable_symbols: bool = True,
) -> None:
    """Configure LSP settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        languages: Languages to enable LSP for
        enable_hover: Enable hover information
        enable_references: Enable find references
        enable_symbols: Enable document symbols
    """
    default_languages = ["python", "typescript", "javascript", "rust", "go"]

    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    lsp_config = {
        "languages": languages or default_languages,
        "features": {
            "hover": enable_hover,
            "references": enable_references,
            "symbols": enable_symbols,
        },
    }
    context.set_capability_config("language_server", lsp_config)

    logger.info(f"Configured LSP for languages: {languages or default_languages}")


def configure_refactoring(
    orchestrator: Any,
    *,
    enable_rename: bool = True,
    enable_extract: bool = True,
    enable_inline: bool = True,
    require_tests: bool = True,
) -> None:
    """Configure refactoring capabilities.

    Args:
        orchestrator: Target orchestrator
        enable_rename: Enable rename refactoring
        enable_extract: Enable extract method/variable
        enable_inline: Enable inline refactoring
        require_tests: Require tests before refactoring
    """
    # SOLID DIP: Store config in VerticalContext instead of direct attribute write
    context = orchestrator.vertical_context
    refactor_config = {
        "operations": {
            "rename": enable_rename,
            "extract": enable_extract,
            "inline": enable_inline,
        },
        "require_tests": require_tests,
    }
    context.set_capability_config("refactoring", refactor_config)

    logger.info("Configured refactoring capabilities")


# =============================================================================
# Capability Provider Class (Refactored to use BaseVerticalCapabilityProvider)
# =============================================================================


class CodingCapabilityProvider(BaseVerticalCapabilityProvider):
    """Provider for coding-specific capabilities.

    Refactored to inherit from BaseVerticalCapabilityProvider, eliminating
    ~500 lines of duplicated boilerplate code.

    Example:
        provider = CodingCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_git_safety(orchestrator, block_force_push=True)
        provider.apply_code_style(orchestrator, formatter="black")

        # Get configurations
        style = provider.get_capability_config(orchestrator, "code_style")
    """

    def __init__(self) -> None:
        """Initialize the coding capability provider."""
        super().__init__("coding")

    def _get_capability_definitions(self) -> dict[str, CapabilityDefinition]:
        """Define coding capability definitions.

        Returns:
            Dictionary of coding capability definitions
        """
        return {
            "git_safety": CapabilityDefinition(
                name="git_safety",
                type=CapabilityType.SAFETY,
                description="Git safety rules for preventing dangerous operations",
                version="1.0",
                configure_fn="configure_git_safety",
                default_config={
                    "block_force_push": True,
                    "block_main_push": True,
                    "require_tests_before_commit": False,
                    "allowed_branches": [],
                },
                tags=["safety", "git", "version-control"],
            ),
            "code_style": CapabilityDefinition(
                name="code_style",
                type=CapabilityType.MODE,
                description="Code style and formatting configuration",
                version="1.0",
                configure_fn="configure_code_style",
                get_fn="get_code_style",
                default_config={
                    "formatter": "black",
                    "linter": "ruff",
                    "max_line_length": 100,
                    "enforce_type_hints": True,
                },
                tags=["style", "formatting", "linting"],
            ),
            "test_requirements": CapabilityDefinition(
                name="test_requirements",
                type=CapabilityType.MODE,
                description="Test configuration and requirements",
                version="1.0",
                configure_fn="configure_test_requirements",
                default_config={
                    "min_coverage": 0.0,
                    "required_patterns": [],
                    "framework": "pytest",
                    "run_on_edit": False,
                },
                tags=["testing", "coverage", "quality"],
            ),
            "language_server": CapabilityDefinition(
                name="language_server",
                type=CapabilityType.TOOL,
                description="Language server protocol configuration",
                version="1.0",
                configure_fn="configure_language_server",
                default_config={
                    "languages": ["python", "typescript", "javascript", "rust", "go"],
                    "features": {
                        "hover": True,
                        "references": True,
                        "symbols": True,
                    },
                },
                tags=["lsp", "ide", "code-intelligence"],
            ),
            "refactoring": CapabilityDefinition(
                name="refactoring",
                type=CapabilityType.TOOL,
                description="Refactoring tool configuration",
                version="1.0",
                configure_fn="configure_refactoring",
                default_config={
                    "operations": {
                        "rename": True,
                        "extract": True,
                        "inline": True,
                    },
                    "require_tests": True,
                },
                dependencies=["language_server"],
                tags=["refactoring", "code-transformation"],
            ),
        }

    # Delegate to handler functions (required by BaseVerticalCapabilityProvider)
    def configure_git_safety(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure git safety capability."""
        configure_git_safety(orchestrator, **kwargs)

    def configure_code_style(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure code style capability."""
        configure_code_style(orchestrator, **kwargs)

    def configure_test_requirements(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure test requirements capability."""
        configure_test_requirements(orchestrator, **kwargs)

    def configure_language_server(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure language server capability."""
        configure_language_server(orchestrator, **kwargs)

    def configure_refactoring(self, orchestrator: Any, **kwargs: Any) -> None:
        """Configure refactoring capability."""
        configure_refactoring(orchestrator, **kwargs)

    def get_code_style(self, orchestrator: Any) -> dict[str, Any]:
        """Get code style configuration."""
        return get_code_style(orchestrator)


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


# Create singleton instance for generating CAPABILITIES list
_provider_instance: Optional[CodingCapabilityProvider] = None


def _get_provider() -> CodingCapabilityProvider:
    """Get or create provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = CodingCapabilityProvider()
    return _provider_instance


# Generate CAPABILITIES list from provider
CAPABILITIES: list[CapabilityEntry] = []


def _generate_capabilities_list() -> None:
    """Generate CAPABILITIES list from provider."""
    global CAPABILITIES
    if not CAPABILITIES:
        provider = _get_provider()
        CAPABILITIES.extend(provider.generate_capabilities_list())


_generate_capabilities_list()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_coding_capabilities() -> list[CapabilityEntry]:
    """Get all coding capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_coding_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for coding vertical.

    Returns:
        CapabilityLoader with coding capabilities registered
    """

    provider = _get_provider()
    return provider.create_capability_loader()


def get_capability_configs() -> dict[str, Any]:
    """Get coding capability configurations for centralized storage.

    Returns default coding configuration for VerticalContext storage.
    This replaces direct orchestrator code_style/test_config assignment.

    Returns:
        Dict with default coding capability configurations
    """
    provider = _get_provider()
    return provider.generate_capability_configs()


__all__ = [
    # Handlers
    "configure_git_safety",
    "configure_code_style",
    "configure_test_requirements",
    "configure_language_server",
    "configure_refactoring",
    "get_code_style",
    # Provider class
    "CodingCapabilityProvider",
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_coding_capabilities",
    "create_coding_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]
