# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility-only wrapper over the deprecated system prompt coordinator."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

from victor.agent.services.system_prompt_runtime import SystemPromptCoordinator

logger = logging.getLogger(__name__)


class PromptRuntimeSupport(SystemPromptCoordinator):
    """Deprecated compatibility wrapper over ``SystemPromptCoordinator``.

    New code should use ``UnifiedPromptPipeline`` directly. This class exists
    only to preserve legacy imports while routing behavior through the existing
    compatibility coordinator surface instead of carrying its own prompt logic.
    """

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
        super().__init__(
            prompt_builder=prompt_builder,
            get_context_window=get_context_window,
            provider_name=provider_name,
            model_name=model_name,
            get_tools=get_tools,
            get_mode_controller=get_mode_controller,
            task_analyzer=task_analyzer,
            session_id=session_id,
            _emit_deprecation_warning=False,
        )

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Preserve legacy analyzer method support before delegating."""
        analyzer = self._task_analyzer
        if analyzer is not None and hasattr(analyzer, "classify_task_keywords"):
            try:
                return analyzer.classify_task_keywords(user_message)
            except Exception as exc:
                logger.debug("PromptRuntimeSupport keyword classification failed: %s", exc)
        return super().classify_task_keywords(user_message)

    def classify_task_with_context(
        self,
        user_message: str,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Preserve legacy contextual analyzer support before delegating."""
        analyzer = self._task_analyzer
        if analyzer is not None and hasattr(analyzer, "classify_task_with_context"):
            try:
                return analyzer.classify_task_with_context(user_message, history)
            except Exception as exc:
                logger.debug(
                    "PromptRuntimeSupport contextual classification failed: %s",
                    exc,
                )
        return super().classify_task_with_context(user_message, history)
