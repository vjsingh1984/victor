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

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor.coding.capabilities")

    # Or use directly
    from victor.coding.capabilities import (
        get_coding_capabilities,
        CodingCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability
from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Handlers
# =============================================================================


def configure_git_safety(
    orchestrator: Any,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    allowed_branches: Optional[List[str]] = None,
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

    # Store config for commit hook
    if hasattr(orchestrator, "safety_config"):
        orchestrator.safety_config["git"] = {
            "require_tests_before_commit": require_tests_before_commit,
            "allowed_branches": allowed_branches or [],
        }

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
    if hasattr(orchestrator, "code_style"):
        orchestrator.code_style = {
            "formatter": formatter,
            "linter": linter,
            "max_line_length": max_line_length,
            "enforce_type_hints": enforce_type_hints,
        }

    logger.info(f"Configured code style: formatter={formatter}, linter={linter}")


def get_code_style(orchestrator: Any) -> Dict[str, Any]:
    """Get current code style configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Code style configuration dict
    """
    return getattr(
        orchestrator,
        "code_style",
        {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
            "enforce_type_hints": True,
        },
    )


def configure_test_requirements(
    orchestrator: Any,
    *,
    min_coverage: float = 0.0,
    required_test_patterns: Optional[List[str]] = None,
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
    if hasattr(orchestrator, "test_config"):
        orchestrator.test_config = {
            "min_coverage": min_coverage,
            "required_patterns": required_test_patterns or [],
            "framework": test_framework,
            "run_on_edit": run_tests_on_edit,
        }

    logger.info(f"Configured test requirements: framework={test_framework}")


def configure_language_server(
    orchestrator: Any,
    *,
    languages: Optional[List[str]] = None,
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

    if hasattr(orchestrator, "lsp_config"):
        orchestrator.lsp_config = {
            "languages": languages or default_languages,
            "features": {
                "hover": enable_hover,
                "references": enable_references,
                "symbols": enable_symbols,
            },
        }

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
    if hasattr(orchestrator, "refactor_config"):
        orchestrator.refactor_config = {
            "operations": {
                "rename": enable_rename,
                "extract": enable_extract,
                "inline": enable_inline,
            },
            "require_tests": require_tests,
        }

    logger.info("Configured refactoring capabilities")


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="coding_git_safety",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Git safety rules for preventing dangerous operations",
)
def coding_git_safety(
    block_force_push: bool = True,
    block_main_push: bool = True,
    **kwargs: Any,
) -> Callable:
    """Git safety capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_git_safety(
            orchestrator,
            block_force_push=block_force_push,
            block_main_push=block_main_push,
            **kwargs,
        )

    return handler


@capability(
    name="coding_style",
    capability_type=CapabilityType.MODE,  # Use MODE for configuration
    version="1.0",
    description="Code style and formatting configuration",
    getter="get_code_style",
)
def coding_style(
    formatter: str = "black",
    linter: str = "ruff",
    **kwargs: Any,
) -> Callable:
    """Code style capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_code_style(
            orchestrator,
            formatter=formatter,
            linter=linter,
            **kwargs,
        )

    return handler


@capability(
    name="coding_tests",
    capability_type=CapabilityType.MODE,  # Use MODE for configuration
    version="1.0",
    description="Test configuration and requirements",
)
def coding_tests(
    min_coverage: float = 0.0,
    test_framework: str = "pytest",
    **kwargs: Any,
) -> Callable:
    """Test requirements capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_test_requirements(
            orchestrator,
            min_coverage=min_coverage,
            test_framework=test_framework,
            **kwargs,
        )

    return handler


@capability(
    name="coding_lsp",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Language server protocol configuration",
)
def coding_lsp(languages: Optional[List[str]] = None, **kwargs: Any) -> Callable:
    """LSP capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_language_server(orchestrator, languages=languages, **kwargs)

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class CodingCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for coding-specific capabilities.

    This class provides a structured way to access and apply
    coding capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = CodingCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_git_safety(orchestrator)
        provider.apply_code_style(orchestrator, formatter="black")

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("git_safety")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "git_safety": configure_git_safety,
            "code_style": configure_code_style,
            "test_requirements": configure_test_requirements,
            "language_server": configure_language_server,
            "refactoring": configure_refactoring,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "git_safety": CapabilityMetadata(
                name="git_safety",
                description="Git safety rules for preventing dangerous operations",
                version="1.0",
                tags=["safety", "git", "version-control"],
            ),
            "code_style": CapabilityMetadata(
                name="code_style",
                description="Code style and formatting configuration",
                version="1.0",
                tags=["style", "formatting", "linting"],
            ),
            "test_requirements": CapabilityMetadata(
                name="test_requirements",
                description="Test configuration and requirements",
                version="1.0",
                tags=["testing", "coverage", "quality"],
            ),
            "language_server": CapabilityMetadata(
                name="language_server",
                description="Language server protocol configuration",
                version="1.0",
                tags=["lsp", "ide", "code-intelligence"],
            ),
            "refactoring": CapabilityMetadata(
                name="refactoring",
                description="Refactoring tool configuration",
                version="1.0",
                dependencies=["language_server"],
                tags=["refactoring", "code-transformation"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_git_safety(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply git safety capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Git safety options
        """
        configure_git_safety(orchestrator, **kwargs)
        self._applied.add("git_safety")

    def apply_code_style(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply code style capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Code style options
        """
        configure_code_style(orchestrator, **kwargs)
        self._applied.add("code_style")

    def apply_test_requirements(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply test requirements capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Test options
        """
        configure_test_requirements(orchestrator, **kwargs)
        self._applied.add("test_requirements")

    def apply_language_server(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply language server capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: LSP options
        """
        configure_language_server(orchestrator, **kwargs)
        self._applied.add("language_server")

    def apply_refactoring(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply refactoring capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Refactoring options
        """
        configure_refactoring(orchestrator, **kwargs)
        self._applied.add("refactoring")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all coding capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_git_safety(orchestrator)
        self.apply_code_style(orchestrator)
        self.apply_test_requirements(orchestrator)
        self.apply_language_server(orchestrator)
        self.apply_refactoring(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_git_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_git_safety",
            description="Git safety rules for preventing dangerous operations",
        ),
        handler=configure_git_safety,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_style",
            capability_type=CapabilityType.MODE,  # Use MODE for configuration
            version="1.0",
            setter="configure_code_style",
            getter="get_code_style",
            description="Code style and formatting configuration",
        ),
        handler=configure_code_style,
        getter_handler=get_code_style,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_tests",
            capability_type=CapabilityType.MODE,  # Use MODE for configuration
            version="1.0",
            setter="configure_test_requirements",
            description="Test configuration and requirements",
        ),
        handler=configure_test_requirements,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_lsp",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_language_server",
            description="Language server protocol configuration",
        ),
        handler=configure_language_server,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="coding_refactoring",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_refactoring",
            description="Refactoring tool configuration",
        ),
        handler=configure_refactoring,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_coding_capabilities() -> List[CapabilityEntry]:
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
    from victor.framework import CapabilityLoader

    loader = CapabilityLoader()

    # Register all coding capabilities
    for entry in CAPABILITIES:
        loader._register_capability_internal(
            capability=entry.capability,
            handler=entry.handler,
            getter_handler=entry.getter_handler,
            source_module="victor.coding.capabilities",
        )

    return loader


__all__ = [
    # Handlers
    "configure_git_safety",
    "configure_code_style",
    "configure_test_requirements",
    "configure_language_server",
    "configure_refactoring",
    "get_code_style",
    # Provider class and base types
    "CodingCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_coding_capabilities",
    "create_coding_capability_loader",
]
