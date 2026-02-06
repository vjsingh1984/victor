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

Package Structure (v0.4.0+):
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
    from victor.coding.codebase import CodebaseIndex
"""

from typing import Any

import warnings

# Suppress pydantic warnings from third-party libraries (lancedb, etc.)
# These warnings are not under our control and don't affect functionality
warnings.filterwarnings(
    "ignore",
    message='Field "model_" has conflict with protected namespace "model_"',
    category=UserWarning,
)

__version__ = "0.5.0"
__author__ = "Vijaykumar Singh"
__email__ = "singhvjd@gmail.com"
__license__ = "Apache-2.0"

# Core classes (existing API - for backward compatibility)
# AgentOrchestrator is lazy-loaded to improve startup time
# Most users should use `from victor import Agent` instead
from victor.config.settings import Settings


def __getattr__(name: str) -> Any:
    """Lazy import AgentOrchestrator to improve startup time.

    The recommended API is `from victor import Agent`, which is imported immediately.
    AgentOrchestrator is only needed for advanced use cases, so it's loaded on-demand.
    """
    import importlib

    if name == "AgentOrchestrator":
        from victor.agent import AgentOrchestrator as AO

        globals()["AgentOrchestrator"] = AO
        return AO

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Framework API (simplified - new golden path)
# Import only what's actually exported from victor.framework
from victor.framework import (  # type: ignore[attr-defined]  # mypy can't verify lazy-loaded exports via __getattr__
    Agent,
    AgentConfig,
    AgentError,
    AgentExecutionEvent,
    BudgetExhaustedError,
    CancellationError,
    ChatSession,
    ConfigurationError,
    ConversationStage,
    EventType,
    FrameworkTaskType,
    ProviderError,
    State,
    StateHooks,
    Task,
    TaskResult,
    ToolCategory,
    ToolError,
    ToolSet,
)

__all__ = [
    # Framework API (5 core concepts + supporting classes)
    "Agent",
    "Task",
    "State",
    "AgentExecutionEvent",
    # Supporting classes
    "ChatSession",
    "TaskResult",
    "FrameworkTaskType",
    "ToolSet",
    "ToolCategory",
    "ConversationStage",
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
