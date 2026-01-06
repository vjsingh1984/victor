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

"""Capability provider framework for verticals.

This module provides the base classes for vertical capability providers,
enabling consistent capability registration and discovery across all verticals.

Exports:
    BaseCapabilityProvider: Abstract base class for capability providers.
    CapabilityMetadata: Dataclass for capability metadata.

Example:
    from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

    class MyCapabilityProvider(BaseCapabilityProvider[MyCapability]):
        def get_capabilities(self) -> Dict[str, MyCapability]:
            return {"my_cap": self._my_capability}

        def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
            return {
                "my_cap": CapabilityMetadata(
                    name="my_cap",
                    description="My custom capability"
                )
            }
"""

from victor.framework.capabilities.base import (
    BaseCapabilityProvider,
    CapabilityMetadata,
)

__all__ = [
    "BaseCapabilityProvider",
    "CapabilityMetadata",
]
