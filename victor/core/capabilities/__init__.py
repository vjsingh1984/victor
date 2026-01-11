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

"""Centralized capability system for verticals.

This module provides a consolidated, YAML-based configuration system for
vertical capabilities, replacing scattered capability definitions.

Example:
    from victor.core.capabilities import CapabilityLoader, CapabilityType

    # Load capabilities for a vertical
    loader = CapabilityLoader.from_vertical("coding")
    capabilities = loader.list_capabilities()

    # Get specific capability
    review_cap = loader.get_capability("code_review")
    print(review_cap.description)
"""

from victor.core.capabilities.base_loader import (
    Capability,
    CapabilityLoader,
    CapabilitySet,
    CapabilityType,
)

__all__ = [
    "Capability",
    "CapabilityLoader",
    "CapabilitySet",
    "CapabilityType",
]
