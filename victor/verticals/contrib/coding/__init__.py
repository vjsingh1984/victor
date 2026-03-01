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
    assistant.py              - CodingAssistant vertical class
    middleware.py             - Code correction middleware
    safety.py                  - Coding-specific safety patterns (legacy)
    safety_enhanced.py        - Enhanced safety with SafetyCoordinator
    conversation_enhanced.py  - Enhanced conversation with ConversationCoordinator
    prompts.py                - Task type hints and prompt contributions
    mode_config.py            - Mode configurations and tool budgets
    tool_dependencies.py      - Code tool dependency graph
    service_provider.py       - DI service registration
    workflows/                - Coding-specific workflows

Usage:
    from victor.coding import CodingAssistant

    # Get vertical configuration
    config = CodingAssistant.get_config()

    # Get extensions for framework integration
    extensions = CodingAssistant.get_extensions()

    # Use enhanced features
    from victor.coding import EnhancedCodingSafetyExtension, EnhancedCodingConversationManager

    safety_ext = EnhancedCodingSafetyExtension()
    conv_mgr = EnhancedCodingConversationManager()
"""

from victor.verticals.contrib.coding.assistant import CodingAssistant
from victor.verticals.contrib.coding.middleware import (
    CodingMiddleware,
    CodeCorrectionMiddleware,
)
from victor.verticals.contrib.coding.safety import CodingSafetyExtension
from victor.verticals.contrib.coding.safety_enhanced import (
    CodingSafetyRules,
    EnhancedCodingSafetyExtension,
)
from victor.verticals.contrib.coding.conversation_enhanced import (
    CodingContext,
    EnhancedCodingConversationManager,
)
from victor.verticals.contrib.coding.prompts import CodingPromptContributor
from victor.verticals.contrib.coding.mode_config import CodingModeConfigProvider
from victor.verticals.contrib.coding.service_provider import CodingServiceProvider
from victor.verticals.contrib.coding.capabilities import (
    CodingCapabilityProvider,
    get_coding_capabilities,
    create_coding_capability_loader,
)

# Import canonical tool dependency provider instead of deprecated class
from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

# Create canonical provider for coding vertical
CodingToolDependencyProvider = create_vertical_tool_dependency_provider("coding")

__all__ = [
    # Main vertical
    "CodingAssistant",
    # Extensions
    "CodingMiddleware",
    "CodeCorrectionMiddleware",
    "CodingSafetyExtension",
    # Enhanced Extensions (with new coordinators)
    "EnhancedCodingSafetyExtension",
    "EnhancedCodingConversationManager",
    "CodingSafetyRules",
    "CodingContext",
    # Other extensions
    "CodingPromptContributor",
    "CodingModeConfigProvider",
    "CodingToolDependencyProvider",  # Now uses canonical provider
    "CodingServiceProvider",
    # Phase 4 - Dynamic Capabilities
    "CodingCapabilityProvider",
    "get_coding_capabilities",
    "create_coding_capability_loader",
]
