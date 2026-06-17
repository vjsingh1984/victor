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

Advanced Config:
    from victor import Agent, UnifiedAgentConfig

    agent = await Agent.create(config=UnifiedAgentConfig.high_budget())

Full API (advanced):
    from victor import AgentOrchestrator, Settings

    settings = Settings()
    orchestrator = await AgentOrchestrator.from_settings(settings, "default")

For coding-specific features, install the victor-coding package:
    pip install victor-coding
Verticals are discovered automatically via the victor.plugins entry point.
"""

import importlib
import os
import sys
from typing import Any, Callable

_MIN_SUPPORTED_PYTHON = (3, 10)


def _ensure_supported_python() -> None:
    """Fail fast with a clear error on unsupported interpreters."""
    current = sys.version_info[:3]
    if current >= _MIN_SUPPORTED_PYTHON:
        return

    required = ".".join(str(part) for part in _MIN_SUPPORTED_PYTHON)
    running = ".".join(str(part) for part in current)
    raise RuntimeError(
        f"Victor requires Python {required}+; current interpreter is Python {running}. "
        "Use Python 3.10 or newer and activate the correct virtual environment."
    )


_ensure_supported_python()

from victor._contracts_bootstrap import prefer_repo_local_victor_contracts

prefer_repo_local_victor_contracts(__file__)

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-ai")
except Exception:
    __version__ = "0.0.0"  # fallback for editable installs without metadata
__author__ = "Vijaykumar Singh"
__email__ = "singhvjd@gmail.com"
__license__ = "Apache-2.0"

_LIGHT_IMPORT = str(os.getenv("VICTOR_LIGHT_IMPORT", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class _CallableModuleProxy:
    """Expose a callable decorator API without shadowing a real submodule.

    `unittest.mock.patch("victor.agent...")` relies on `victor.agent` resolving
    to the package module. At the same time, the public API expects `victor.agent`
    to be callable as a decorator. This proxy forwards attribute access to the
    real submodule while preserving the callable interface.
    """

    def __init__(self, module_name: str, func: Callable[..., Any]) -> None:
        self._module_name = module_name
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        module = importlib.import_module(self._module_name)
        return getattr(module, name)

    def __dir__(self) -> list[str]:
        module = importlib.import_module(self._module_name)
        return sorted(set(dir(module)) | set(self.__dict__) | set(dir(type(self))))

    def __repr__(self) -> str:
        return f"<callable module proxy for {self._module_name}>"


if not _LIGHT_IMPORT:
    # Core classes (existing API - for backward compatibility)
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

    # Framework API (simplified - new golden path)
    from victor.framework import (
        Agent,
        AgentConfig,
        UnifiedAgentConfig,
        AgentError,
        AgentExecutionEvent,
        BudgetExhaustedError,
        CancellationError,
        ChatSession,
        ConfigurationError,
        EventType,
        FrameworkTaskType,
        ProviderError,
        Stage,
        State,
        StateHooks,
        Task,
        TaskResult,
        ToolCategory,
        ToolError,
        Tools,
        ToolSet,
    )

    # Decorator API — @victor.agent / @victor.task
    from victor.framework.decorators import (
        agent as _agent_decorator,
        task,
        AgentCallable,
        TaskDefinition,
    )

    agent = _CallableModuleProxy("victor.agent", _agent_decorator)

    __all__ = [
        # Decorator API
        "agent",
        "task",
        "AgentCallable",
        "TaskDefinition",
        # Framework API (5 core concepts + supporting classes)
        "Agent",
        "Task",
        "Tools",
        "State",
        "AgentExecutionEvent",
        # Supporting classes
        "ChatSession",
        "TaskResult",
        "FrameworkTaskType",
        "ToolSet",
        "ToolCategory",
        "Stage",
        "StateHooks",
        "EventType",
        "AgentConfig",
        "UnifiedAgentConfig",
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
else:
    # Light mode intentionally avoids importing orchestrator/provider stacks.
    from victor.config.settings import Settings

    __all__ = ["Settings"]
