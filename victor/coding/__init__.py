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

"""Coding Vertical Package.

Victor's primary vertical for software development, providing:
- Code exploration and understanding
- Bug fixing and refactoring
- Feature implementation
- Testing and verification
- Git operations and version control

This package contains all coding-specific logic extracted from the framework,
enabling the framework to remain domain-agnostic while providing rich
coding assistant functionality.

Package Structure:
    assistant.py        - CodingAssistant vertical class
    middleware.py       - Code correction middleware
    safety.py           - Coding-specific safety patterns
    prompts.py          - Task type hints and prompt contributions
    mode_config.py      - Mode configurations and tool budgets
    tool_dependencies.py - Code tool dependency graph
    service_provider.py - DI service registration
    workflows/          - Coding-specific workflows

Usage:
    from victor.coding import CodingAssistant

    # Get vertical configuration
    config = CodingAssistant.get_config()

    # Get extensions for framework integration
    extensions = CodingAssistant.get_extensions()
"""

from victor.coding.assistant import CodingAssistant
from victor.coding.middleware import (
    CodingMiddleware,
    CodeCorrectionMiddleware,
)
from victor.coding.safety import CodingSafetyExtension
from victor.coding.prompts import CodingPromptContributor
from victor.coding.mode_config import CodingModeConfigProvider
from victor.coding.service_provider import CodingServiceProvider
from victor.coding.capabilities import (
    CodingCapabilityProvider,
    get_coding_capabilities,
    create_coding_capability_loader,
)

# Import canonical tool dependency provider instead of deprecated class
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for coding vertical
CodingToolDependencyProvider = create_vertical_tool_dependency_provider("coding")


# Auto-register escape hatches (OCP-compliant)
def _register_escape_hatches() -> None:
    """Register coding vertical's escape hatches with the global registry.

    This function runs on module import to automatically register escape hatches.
    Using the registry's discover_from_all_verticals() method is the OCP-compliant
    approach, as it doesn't require the framework to know about specific verticals.
    """
    try:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry

        # Import escape hatches module to trigger side-effect registration
        from victor.coding import escape_hatches  # noqa: F401

        # Register with global registry
        registry = EscapeHatchRegistry.get_instance()
        registry.register_from_vertical(
            "coding",
            conditions=escape_hatches.CONDITIONS,
            transforms=escape_hatches.TRANSFORMS,
        )
    except Exception:
        # If registration fails, it's not critical
        pass


# Register on import
_register_escape_hatches()

__all__ = [
    # Main vertical
    "CodingAssistant",
    # Extensions
    "CodingMiddleware",
    "CodeCorrectionMiddleware",
    "CodingSafetyExtension",
    "CodingPromptContributor",
    "CodingModeConfigProvider",
    "CodingToolDependencyProvider",  # Now uses canonical provider
    "CodingServiceProvider",
    # Phase 4 - Dynamic Capabilities
    "CodingCapabilityProvider",
    "get_coding_capabilities",
    "create_coding_capability_loader",
]
