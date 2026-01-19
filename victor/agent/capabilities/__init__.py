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

"""Dynamic capability registration system.

This package provides Open/Closed Principle (OCP) compliance for Victor's
capability system. External packages can register capabilities via entry points
without modifying core code.

Main Components:
- CapabilityBase: Abstract base class for capabilities
- CapabilitySpec: Dataclass for capability metadata
- DynamicCapabilityRegistry: Registry with entry point support
- Built-in capabilities: Default capability implementations

Usage (External Package):
    # 1. Create a capability class
    from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec

    class MyCapability(CapabilityBase):
        @classmethod
        def get_spec(cls) -> CapabilitySpec:
            return CapabilitySpec(
                name="my_capability",
                method_name="set_my_capability",
                version="1.0",
                description="My custom capability"
            )

    # 2. Register via entry points in pyproject.toml
    # [project.entry-points."victor.capabilities"]
    # my_capability = "my_package.capabilities:MyCapability"

Usage (Runtime):
    from victor.agent.capabilities.registry import DynamicCapabilityRegistry

    registry = DynamicCapabilityRegistry()
    method = registry.get_method_for_capability("enabled_tools")
    # Returns: "set_enabled_tools"
"""

from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec
from victor.agent.capabilities.registry import DynamicCapabilityRegistry

__all__ = [
    "CapabilityBase",
    "CapabilitySpec",
    "DynamicCapabilityRegistry",
]
