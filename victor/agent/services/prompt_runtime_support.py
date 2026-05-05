# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility-only prompt runtime support wrapper.

.. deprecated::
    Superseded by ``victor.agent.prompt_pipeline.UnifiedPromptPipeline``.
    This module exists only to preserve deprecated imports while keeping
    live prompt orchestration on the canonical pipeline path.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptRuntimeSupport:
    """Deprecated compatibility wrapper over ``UnifiedPromptPipeline``."""

    def __init__(
        self,
        prompt_builder: Any = None,
        get_context_window: Optional[Callable[[], int]] = None,
        provider_name: str = "",
        model_name: str = "",
        get_tools: Optional[Callable[[], Optional[Any]]] = None,
        get_mode_controller: Optional[Callable[[], Optional[object]]] = None,
        task_analyzer: Optional[Any] = None,
        session_id: str = "",
    ) -> None:
        warnings.warn(
            "PromptRuntimeSupport is deprecated and will be removed in v1.0.0. "
            "Use UnifiedPromptPipeline from victor.agent.prompt_pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._prompt_builder = prompt_builder
        self._get_context_window = get_context_window or (lambda: 128000)
        self._provider_name = provider_name
        self._model_name = model_name
        self._get_tools = get_tools
        self._get_mode_controller = get_mode_controller
        self._task_analyzer = task_analyzer
        self._session_id = session_id
        self._pipeline = None

        if prompt_builder is None:
            return

        try:
            from victor.agent.content_registry import ContentRegistry
            from victor.agent.optimization_injector import OptimizationInjector
            from victor.agent.prompt_pipeline import UnifiedPromptPipeline

            self._pipeline = UnifiedPromptPipeline(
                provider=None,
                builder=prompt_builder,
                registry=ContentRegistry(),
                optimizer=OptimizationInjector(),
                task_analyzer=task_analyzer,
                get_context_window=self._get_context_window,
                session_id=session_id,
            )
            logger.debug("PromptRuntimeSupport delegated to UnifiedPromptPipeline")
        except Exception as exc:
            logger.debug("PromptRuntimeSupport pipeline initialization failed: %s", exc)

    def build_system_prompt(self) -> str:
        """Build the compatibility system prompt through the canonical pipeline."""
        if self._pipeline is not None:
            return self._pipeline.build_system_prompt()
        if self._prompt_builder is not None:
            return self._prompt_builder.build()
        return ""

    def resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases through the canonical pipeline when available."""
        if self._pipeline is not None:
            return self._pipeline.resolve_shell_variant(tool_name)
        return tool_name

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify the current task via the canonical pipeline when available."""
        analyzer = self._task_analyzer
        if analyzer is not None and hasattr(analyzer, "classify_task_keywords"):
            try:
                return analyzer.classify_task_keywords(user_message)
            except Exception as exc:
                logger.debug("PromptRuntimeSupport keyword classification failed: %s", exc)
        if self._pipeline is not None:
            return self._pipeline.classify_task_keywords(user_message)
        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self,
        user_message: str,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Classify the current task with history context."""
        analyzer = self._task_analyzer
        if analyzer is not None and hasattr(analyzer, "classify_task_with_context"):
            try:
                return analyzer.classify_task_with_context(user_message, history)
            except Exception as exc:
                logger.debug(
                    "PromptRuntimeSupport contextual classification failed: %s",
                    exc,
                )
        if self._pipeline is not None:
            return self._pipeline.classify_task_with_context(user_message, history)
        return {"task_type": "default", "confidence": 0.0}
