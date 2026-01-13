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

"""Agent coordinators for orchestrator decomposition.

This package provides coordinators that extract specific responsibilities
from the monolithic orchestrator, following the Single Responsibility Principle.

Coordinators:
    - ConfigCoordinator: Configuration loading and validation
    - PromptCoordinator: Prompt building from contributors
    - ContextCoordinator: Context management and compaction
    - AnalyticsCoordinator: Analytics collection and export
    - ChatCoordinator: Chat and streaming operations
"""

from victor.agent.coordinators.config_coordinator import (
    ConfigCoordinator,
    ValidationResult,
    OrchestratorConfig,
    SettingsConfigProvider,
    EnvironmentConfigProvider,
)

from victor.agent.coordinators.prompt_coordinator import (
    PromptCoordinator,
    PromptBuildError,
    BasePromptContributor,
    SystemPromptContributor,
    TaskHintContributor,
)

from victor.agent.coordinators.context_coordinator import (
    ContextCoordinator,
    ContextCompactionError,
    BaseCompactionStrategy,
    TruncationCompactionStrategy,
)

from victor.agent.coordinators.analytics_coordinator import (
    AnalyticsCoordinator,
    SessionAnalytics,
    BaseAnalyticsExporter,
    ConsoleAnalyticsExporter,
)

from victor.agent.coordinators.chat_coordinator import ChatCoordinator

__all__ = [
    # ConfigCoordinator
    "ConfigCoordinator",
    "ValidationResult",
    "OrchestratorConfig",
    "SettingsConfigProvider",
    "EnvironmentConfigProvider",
    # PromptCoordinator
    "PromptCoordinator",
    "PromptBuildError",
    "BasePromptContributor",
    "SystemPromptContributor",
    "TaskHintContributor",
    # ContextCoordinator
    "ContextCoordinator",
    "ContextCompactionError",
    "BaseCompactionStrategy",
    "TruncationCompactionStrategy",
    # AnalyticsCoordinator
    "AnalyticsCoordinator",
    "SessionAnalytics",
    "BaseAnalyticsExporter",
    "ConsoleAnalyticsExporter",
    # ChatCoordinator
    "ChatCoordinator",
]
