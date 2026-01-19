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

"""Generic capability providers for vertical extensions.

This module contains framework-level capability providers that can be reused
across all verticals to promote code reuse and maintain SOLID principles.

Phase 2: Generic Capabilities - Framework Layer (SOLID Refactoring)

Capabilities:
- BaseCapabilityProvider: Abstract base for vertical capability providers
- CapabilityMetadata: Metadata for capability registration
- BaseVerticalCapabilityProvider: Comprehensive base for vertical providers (eliminates ~2000 LOC)
- CapabilityRegistry: Global registry for vertical capability management
- CapabilityDefinition: Declarative capability definitions
- ConfigurationCapabilityProvider: Generic configuration management (configure_X + get_X pattern)
- FileOperationsCapability: Common file operation tools (read, write, edit, grep)
- PromptContributionCapability: Common prompt hints for task types
- PrivacyCapabilityProvider: Framework-level privacy and PII management (cross-vertical)
"""

# Base classes for capability providers
from .base import BaseCapabilityProvider, CapabilityMetadata

# Vertical capability provider base classes (new)
from .base_vertical_capability_provider import (
    BaseVerticalCapabilityProvider,
    CapabilityDefinition,
    _map_capability_type,
)

# Global registry for capability providers
from .registry import CapabilityRegistry

# Concrete capability implementations
from .configuration import ConfigurationCapabilityProvider, configure_capability, get_capability
from .file_operations import FileOperationsCapability
from .prompt_contributions import PromptContributionCapability
from .privacy import PrivacyCapabilityProvider

__all__ = [
    # Base classes
    "BaseCapabilityProvider",
    "CapabilityMetadata",
    # Vertical capability providers (NEW - eliminates ~2000 LOC duplication)
    "BaseVerticalCapabilityProvider",
    "CapabilityDefinition",
    "_map_capability_type",
    "CapabilityRegistry",
    # Concrete implementations
    "ConfigurationCapabilityProvider",
    "configure_capability",
    "get_capability",
    "FileOperationsCapability",
    "PromptContributionCapability",
    "PrivacyCapabilityProvider",
]
