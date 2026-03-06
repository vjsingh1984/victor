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

"""Service implementations for SOLID refactoring.

This package contains the service implementations that extract
functionality from the monolithic AgentOrchestrator into
focused, single-responsibility services.

Services:
    ChatService: Chat flow coordination and streaming
    ToolService: Tool selection and execution
    ContextService: Context management and metrics
    ProviderService: Provider management and switching
    RecoveryService: Error recovery and resilience
    SessionService: Session lifecycle management
    LLMDecisionService: LLM-assisted decision making fallback
"""

from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.agent.services.context_service import ContextService, ContextServiceConfig
from victor.agent.services.decision_service import LLMDecisionService, LLMDecisionServiceConfig
from victor.agent.services.provider_service import ProviderService
from victor.agent.services.recovery_service import RecoveryService, RecoveryContextImpl
from victor.agent.services.session_service import SessionService, SessionInfoImpl
from victor.agent.services.tool_service import (
    ToolBudgetExceededError,
    ToolService,
    ToolServiceConfig,
)

__all__ = [
    "ChatService",
    "ChatServiceConfig",
    "ContextService",
    "ContextServiceConfig",
    "LLMDecisionService",
    "LLMDecisionServiceConfig",
    "ProviderService",
    "RecoveryService",
    "RecoveryContextImpl",
    "SessionService",
    "SessionInfoImpl",
    "ToolBudgetExceededError",
    "ToolService",
    "ToolServiceConfig",
]
