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

"""Intelligent integration builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory


class IntelligentIntegrationBuilder(FactoryAwareBuilder):
    """Build intelligent integration and sub-agent orchestration components."""

    def __init__(self, settings, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs) -> Dict[str, Any]:
        """Build intelligent integration components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # Intelligent pipeline integration (lazy initialization via factory)
        orchestrator._intelligent_integration = None
        orchestrator._intelligent_integration_config = factory.create_integration_config()
        orchestrator._intelligent_pipeline_enabled = getattr(
            self.settings, "intelligent_pipeline_enabled", True
        )
        components["intelligent_integration_config"] = orchestrator._intelligent_integration_config
        components["intelligent_pipeline_enabled"] = orchestrator._intelligent_pipeline_enabled

        # Sub-agent orchestration (via factory) - lazy initialization for parallel task delegation
        (
            orchestrator._subagent_orchestrator,
            orchestrator._subagent_orchestration_enabled,
        ) = factory.setup_subagent_orchestration()
        components["subagent_orchestrator"] = orchestrator._subagent_orchestrator
        components["subagent_orchestration_enabled"] = orchestrator._subagent_orchestration_enabled

        self._register_components(components)
        return components
