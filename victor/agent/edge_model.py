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

"""Edge model provider for micro-decisions.

A lightweight local LLM provider for fast, token-free micro-decisions:
task classification, tool selection, completion detection, etc.

Runs via Ollama with small models (TinyLlama, Qwen2.5-Coder:1.5b) on CPU.
Falls back to heuristic when Ollama is unavailable.

Usage:
    from victor.agent.edge_model import create_edge_decision_service

    service = create_edge_decision_service()
    if service:
        # Use for micro-decisions — zero cloud LLM token cost
        result = service.decide_sync(
            DecisionType.TASK_TYPE_CLASSIFICATION,
            context={"message_excerpt": "Fix the auth bug"},
        )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class EdgeModelConfig:
    """Configuration for the edge model provider.

    Attributes:
        enabled: Whether edge model is active
        provider: Backend provider ("ollama" or "disabled")
        model: Model name (e.g., "qwen2.5-coder:1.5b", "tinyllama")
        timeout_ms: Hard timeout for edge model calls
        max_tokens: Max response tokens (micro-decisions need 30-60)
        cache_ttl: Cache TTL in seconds for identical decisions
        confidence_threshold: Below this, fall back to heuristic
        base_url: Ollama server URL
        tool_selection_enabled: Use edge model for tool ranking
        prompt_focus_enabled: Use edge model for system prompt trimming
        max_tools: Max tools to recommend in tool selection
    """

    enabled: bool = True
    provider: str = "ollama"
    model: str = "qwen3.5:2b"
    timeout_ms: int = 4000
    max_tokens: int = 50
    cache_ttl: int = 120
    confidence_threshold: float = 0.6
    base_url: str = "http://localhost:11434"
    tool_selection_enabled: bool = True
    prompt_focus_enabled: bool = True
    max_tools: int = 6


# Tool selection prompt — edge model picks most relevant tools
TOOL_SELECTION_PROMPT = """You select tools for an AI coding assistant. Given the user's request, pick the most relevant tools. Respond ONLY with a JSON object.

User request:
{message_excerpt}

Current stage: {stage}
Recent tools used: {recent_tools}

Available tools: {available_tools}

Pick 5-6 most relevant tools for this task.
Respond with JSON: {{"tools": ["tool1", "tool2", ...], "confidence": 0.0-1.0}}"""

# System prompt focus — edge model picks which sections to include
PROMPT_FOCUS_PROMPT = """You optimize system prompts for an AI coding assistant. Given the task, select which prompt sections are needed. Respond ONLY with a JSON object.

Task type: {task_type}
User request excerpt: {message_excerpt}

Available sections:
- "grounding": Base rules (always respond from tool output)
- "completion": Task completion signal markers (**DONE**, **SUMMARY**)
- "tool_guidance": How to call tools correctly
- "file_pagination": Large file reading hints
- "concise_mode": Output brevity directives
- "parallel_read": Batch file reading optimization

Pick only sections needed for this task.
Respond with JSON: {{"sections": ["section1", ...], "confidence": 0.0-1.0}}"""


def create_edge_decision_service(
    config: Optional[EdgeModelConfig] = None,
) -> Optional[Any]:
    """Create an LLMDecisionService backed by a local edge model.

    Uses Ollama with a small model for micro-decisions. Returns None if
    Ollama is unavailable or edge model is disabled.

    Args:
        config: Edge model configuration (uses defaults if None)

    Returns:
        Configured LLMDecisionService, or None if unavailable
    """
    if config is None:
        config = EdgeModelConfig()

    if not config.enabled or config.provider == "disabled":
        logger.debug("Edge model disabled")
        return None

    try:
        from victor.agent.services.decision_service import (
            LLMDecisionService,
            LLMDecisionServiceConfig,
        )
        from victor.providers.registry import ProviderRegistry

        # Create provider using the standard registry — supports any provider type
        # (ollama, deepseek, openai, etc.) via config.provider
        provider_kwargs: Dict[str, Any] = {
            "timeout": config.timeout_ms // 1000,
        }

        if config.provider == "ollama":
            provider_kwargs["base_url"] = config.base_url
            # Verify model is available for Ollama
            if not _check_model_available(config.base_url, config.model):
                logger.warning(
                    "Edge model '%s' not available in Ollama. "
                    "Pull it with: ollama pull %s",
                    config.model,
                    config.model,
                )
                return None
        else:
            # For cloud providers, get API key
            try:
                from victor.config.api_keys import get_api_key

                api_key = get_api_key(config.provider)
                if api_key:
                    provider_kwargs["api_key"] = api_key
            except Exception:
                pass

        provider = ProviderRegistry.create(config.provider, **provider_kwargs)

        decision_config = LLMDecisionServiceConfig(
            confidence_threshold=config.confidence_threshold,
            micro_budget=20,  # Higher budget — edge calls are free
            timeout_ms=config.timeout_ms,
            cache_ttl=config.cache_ttl,
            temperature=0.0,
            max_tokens_override=config.max_tokens,
        )

        service = LLMDecisionService(
            provider=provider,
            model=config.model,
            config=decision_config,
        )
        logger.info(
            "Edge model decision service created: model=%s, timeout=%dms",
            config.model,
            config.timeout_ms,
        )
        return service

    except Exception as e:
        logger.debug("Failed to create edge decision service: %s", e)
        return None


def select_tools_with_edge_model(
    service: Any,
    user_message: str,
    available_tools: List[str],
    stage: str = "initial",
    recent_tools: Optional[List[str]] = None,
    max_tools: int = 6,
) -> Optional[List[str]]:
    """Use edge model to select relevant tools from the full set.

    Uses the decision service's decide_sync() which handles async-in-sync
    context safely (returns heuristic fallback when inside an event loop).

    Args:
        service: LLMDecisionService (edge-backed)
        user_message: The user's request
        available_tools: All available tool names
        stage: Current conversation stage
        recent_tools: Recently used tool names
        max_tools: Maximum tools to return

    Returns:
        List of selected tool names, or None if edge model unavailable
    """
    try:
        from victor.agent.decisions.schemas import DecisionType

        decision = service.decide_sync(
            DecisionType.TOOL_SELECTION,
            context={
                "message_excerpt": user_message[:200],
                "stage": stage,
                "available_tools": ", ".join(available_tools),
            },
            heuristic_confidence=0.0,
        )

        if decision.source in ("heuristic", "budget_exhausted", "timeout_fallback"):
            return None

        result = decision.result
        if not hasattr(result, "tools"):
            return None

        valid = [t for t in result.tools if t in set(available_tools)]
        if valid:
            logger.info(
                "Edge tool selection: %d tools selected (from %d available)",
                len(valid),
                len(available_tools),
            )
            return valid[:max_tools]

    except Exception as e:
        logger.debug("Edge tool selection failed: %s", e)

    return None


def select_prompt_sections_with_edge_model(
    service: Any,
    user_message: str,
    task_type: str,
    available_sections: List[str],
) -> Optional[List[str]]:
    """Use edge model to select which system prompt sections to include.

    Args:
        service: LLMDecisionService (edge-backed)
        user_message: The user's request
        task_type: Classified task type
        available_sections: All available prompt section names

    Returns:
        List of section names to include, or None if unavailable
    """
    try:
        from victor.agent.decisions.chain import should_use_llm

        if not should_use_llm("prompt_focus"):
            return None

        from victor.agent.decisions.schemas import DecisionType

        decision = service.decide_sync(
            DecisionType.PROMPT_FOCUS,
            context={
                "task_type": task_type,
                "message_excerpt": user_message[:200],
                "available_sections": ", ".join(available_sections),
            },
            heuristic_confidence=0.0,
        )

        if decision.source in ("heuristic", "budget_exhausted", "timeout_fallback"):
            return None

        result = decision.result
        if not hasattr(result, "sections"):
            return None

        valid = [s for s in result.sections if s in set(available_sections)]
        if valid:
            logger.info(
                "Edge prompt focus: %d/%d sections selected",
                len(valid),
                len(available_sections),
            )
            return valid

    except Exception as e:
        logger.debug("Edge prompt focus failed: %s", e)

    return None


def _check_model_available(base_url: str, model: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = {m.get("name", "").split(":")[0] for m in models}
            # Also check full name with tag
            model_full_names = {m.get("name", "") for m in models}
            return model in model_names or model in model_full_names
    except Exception:
        pass
    return False
