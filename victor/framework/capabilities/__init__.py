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

Phase 3: Generic Capabilities - Framework Layer

Capabilities:
- BaseCapabilityProvider: Abstract base for vertical capability providers
- CapabilityMetadata: Metadata for capability registration
- FileOperationsCapability: Common file operation tools (read, write, edit, grep)
- PromptContributionCapability: Common prompt hints for task types
- PrivacyCapabilityProvider: Framework-level privacy and PII management (cross-vertical)
"""

# Base classes for capability providers
from .base import BaseCapabilityProvider, CapabilityMetadata

# Concrete capability implementations
from .file_operations import FileOperationsCapability
from .prompt_contributions import PromptContributionCapability
from .privacy import PrivacyCapabilityProvider

__all__ = [
    # Base classes
    "BaseCapabilityProvider",
    "CapabilityMetadata",
    # Concrete implementations
    "FileOperationsCapability",
    "PromptContributionCapability",
    "PrivacyCapabilityProvider",
]
