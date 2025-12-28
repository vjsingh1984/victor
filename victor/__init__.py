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

"""
Victor - A universal terminal-based coding agent supporting multiple LLM providers.

Supports frontier models (Claude, GPT, Gemini) and open-source models
(Ollama, LMStudio, vLLM) with unified tool calling integration.

Package Structure (v0.3.0+):
    - victor (victor-core): Core framework, providers, orchestration
    - victor_coding: Coding vertical with Tree-sitter, LSP, code tools

Simple API (recommended):
    from victor import Agent

    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

Full API (advanced):
    from victor import AgentOrchestrator, Settings

    settings = Settings()
    orchestrator = await AgentOrchestrator.from_settings(settings, "default")

For coding-specific features:
    from victor_coding import CodingVertical
    from victor_coding.codebase import CodebaseIndex
"""

__version__ = "0.3.0"
__author__ = "Vijaykumar Singh"
__email__ = "singhvjd@gmail.com"
__license__ = "Apache-2.0"

# Core classes (existing API - for backward compatibility)
from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings

# Framework API (simplified - new golden path)
from victor.framework import (
    Agent,
    AgentConfig,
    AgentError,
    BudgetExhaustedError,
    CancellationError,
    ChatSession,
    ConfigurationError,
    Event,
    EventType,
    ProviderError,
    Stage,
    State,
    StateHooks,
    Task,
    TaskResult,
    TaskType,
    ToolCategory,
    ToolError,
    Tools,
    ToolSet,
)

__all__ = [
    # Framework API (5 core concepts + supporting classes)
    "Agent",
    "Task",
    "Tools",
    "State",
    "Event",
    # Supporting classes
    "ChatSession",
    "TaskResult",
    "TaskType",
    "ToolSet",
    "ToolCategory",
    "Stage",
    "StateHooks",
    "EventType",
    "AgentConfig",
    # Errors
    "AgentError",
    "ProviderError",
    "ToolError",
    "ConfigurationError",
    "BudgetExhaustedError",
    "CancellationError",
    # Core classes (existing API)
    "AgentOrchestrator",
    "Settings",
]
