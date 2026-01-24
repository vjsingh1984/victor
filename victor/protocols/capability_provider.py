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

"""Capability provider protocols for DI injection.

Phase 1.4: Auto-inject FileOperationsCapability via DI.

This module re-exports FileOperationsCapabilityProtocol from the canonical
location (victor.agent.protocols) for consistency with the protocols package.
"""

# Re-export canonical FileOperationsCapabilityProtocol from victor.agent.protocols
from victor.agent.protocols import FileOperationsCapabilityProtocol

__all__ = ["FileOperationsCapabilityProtocol"]
