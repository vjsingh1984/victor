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

"""Tool Capability System.

Groups tools by functional capabilities for easier composition.

Phase 2, Work Stream 2.2: Tool Capability Groups
"""

from victor.tools.capabilities.system import (
    ToolCapability,
    CapabilityDefinition,
    CapabilityRegistry,
    CapabilitySelector,
)

__all__ = [
    "ToolCapability",
    "CapabilityDefinition",
    "CapabilityRegistry",
    "CapabilitySelector",
]
