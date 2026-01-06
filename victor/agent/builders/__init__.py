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

"""Builder pattern implementations for AgentOrchestrator initialization.

This module provides builder classes that extract component initialization
logic from AgentOrchestrator.__init__() to improve testability and maintainability.

Part of HIGH-005: Initialization Complexity reduction.

Available Builders:
    ComponentBuilder: Abstract base class for all builders
    FactoryAwareBuilder: Base class for builders that use OrchestratorFactory
    ServiceBuilder: Builds core services (DI container, controllers, etc.)
    ToolBuilder: Builds tools and tool-related components
    ProviderBuilder: Builds provider and provider-related components

Usage:
    from victor.agent.builders import ServiceBuilder, ToolBuilder, ProviderBuilder

    # Build services
    service_builder = ServiceBuilder(settings)
    services = service_builder.build()

    # Build provider components
    provider_builder = ProviderBuilder(settings)
    provider_components = provider_builder.build(provider=my_provider)

    # Build tool components
    tool_builder = ToolBuilder(settings)
    tool_components = tool_builder.build(
        service_provider=services['service_provider']
    )
"""

from victor.agent.builders.base import ComponentBuilder, FactoryAwareBuilder
from victor.agent.builders.service_builder import ServiceBuilder
from victor.agent.builders.tool_builder import ToolBuilder
from victor.agent.builders.provider_builder import ProviderBuilder

__all__ = [
    "ComponentBuilder",
    "FactoryAwareBuilder",
    "ServiceBuilder",
    "ToolBuilder",
    "ProviderBuilder",
]
