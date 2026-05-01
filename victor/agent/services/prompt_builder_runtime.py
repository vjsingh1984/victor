"""Service-owned prompt-builder runtime helper."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptBuilderRuntime:
    """Bridge prompt-builder runtime state off the concrete orchestrator."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def sync_prompt_builder_runtime_state(self) -> None:
        """Align prompt-builder state with current mode and enabled tools."""
        runtime = self._runtime
        builder = getattr(runtime, "prompt_builder", None)
        if builder is None:
            return

        cache_invalidated = False
        try:
            enabled_tools = sorted(runtime.get_enabled_tools())
            if builder.available_tools != enabled_tools:
                builder.available_tools = enabled_tools
                cache_invalidated = True
        except Exception as exc:
            logger.debug("Failed to sync enabled tools into prompt builder: %s", exc)
            if builder.available_tools:
                builder.available_tools = []
                cache_invalidated = True

        try:
            mode_prompt = runtime.get_mode_system_prompt()
            if builder.mode_prompt_addition != mode_prompt:
                builder.mode_prompt_addition = mode_prompt
                cache_invalidated = True
        except Exception as exc:
            logger.debug("Failed to sync mode prompt into prompt builder: %s", exc)
            if builder.mode_prompt_addition:
                builder.mode_prompt_addition = ""
                cache_invalidated = True

        if cache_invalidated:
            builder.invalidate_cache()

    def build_system_prompt_fallback(self) -> str:
        """Build the non-pipeline system prompt path."""
        runtime = self._runtime
        builder = getattr(runtime, "prompt_builder", None)
        if builder is None:
            return ""

        prompt_orchestrator = getattr(runtime, "_prompt_orchestrator", None)
        if prompt_orchestrator is None:
            from victor.agent.prompt_orchestrator import get_prompt_orchestrator

            prompt_orchestrator = get_prompt_orchestrator()

        prompt_built_hook = None
        runtime_support = getattr(runtime, "_prompt_runtime_support", None)
        if runtime_support is not None and hasattr(runtime_support, "_emit_prompt_used_event"):
            prompt_built_hook = runtime_support._emit_prompt_used_event
        else:
            legacy_coordinator = getattr(runtime, "_system_prompt_coordinator", None)
            if legacy_coordinator is not None and hasattr(
                legacy_coordinator, "_emit_prompt_used_event"
            ):
                prompt_built_hook = legacy_coordinator._emit_prompt_used_event

        return prompt_orchestrator.build_system_prompt(
            builder_type="legacy",
            provider=getattr(runtime, "provider_name", ""),
            model=getattr(runtime, "model", ""),
            task_type=getattr(builder, "task_type", "default"),
            builder=builder,
            get_context_window=runtime._get_model_context_window,
            on_prompt_built=prompt_built_hook,
        )

    def compose_system_prompt(self, base_system_prompt: str) -> str:
        """Add project-context guidance to the built system prompt when present."""
        project_context = getattr(self._runtime, "project_context", None)
        if project_context and getattr(project_context, "content", None):
            return base_system_prompt + "\n\n" + project_context.get_system_prompt_addition()
        return base_system_prompt

    def update_system_prompt_for_query(self, query_classification=None) -> None:
        """Rebuild the runtime system prompt with query-aware classification."""
        runtime = self._runtime
        pipeline = getattr(runtime, "_prompt_pipeline", None)
        is_frozen = pipeline.is_frozen if pipeline else getattr(runtime, "_system_prompt_frozen", False)
        if getattr(runtime, "_kv_optimization_enabled", False) and is_frozen:
            logger.debug("[cache] System prompt frozen - skipping rebuild for query classification")
            return

        builder = getattr(runtime, "prompt_builder", None)
        if query_classification is not None and builder is not None:
            builder.query_classification = query_classification
            builder.invalidate_cache()

        base_system_prompt = runtime.build_system_prompt()
        runtime._system_prompt = self.compose_system_prompt(base_system_prompt)

        if getattr(runtime, "_kv_optimization_enabled", False):
            runtime._system_prompt_frozen = True

        self.sync_conversation_system_prompt()

    def refresh_system_prompt(
        self,
        query_classification=None,
        *,
        preserve_existing_classification: bool = True,
    ) -> None:
        """Reset prompt runtime caches and rebuild the active system prompt."""
        runtime = self._runtime
        pipeline = getattr(runtime, "_prompt_pipeline", None)
        if pipeline is not None:
            pipeline.unfreeze()
        runtime._system_prompt_frozen = False
        runtime._session_tools = None

        builder = getattr(runtime, "prompt_builder", None)
        if builder is not None:
            builder.invalidate_cache()

        if query_classification is None and preserve_existing_classification and builder is not None:
            query_classification = getattr(builder, "query_classification", None)

        self.update_system_prompt_for_query(query_classification=query_classification)

    def get_system_prompt(self) -> str:
        """Return the current prompt-builder output for protocol consumers."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is None:
            return ""
        return builder.build()

    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom prompt through the prompt-builder surface when supported."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is not None and hasattr(builder, "set_custom_prompt"):
            builder.set_custom_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append extra prompt content through the same custom-prompt surface."""
        current = self.get_system_prompt()
        self.set_system_prompt(current + "\n\n" + content)

    def sync_conversation_system_prompt(self) -> None:
        """Push the current runtime system prompt into the live conversation state."""
        runtime = self._runtime
        conversation = getattr(runtime, "conversation", None)
        if conversation is None:
            return

        prompt = getattr(runtime, "_system_prompt", "")
        conversation.system_prompt = prompt
        if getattr(conversation, "_system_added", False) and getattr(conversation, "_messages", None):
            if conversation._messages[0].role == "system":
                from victor.providers.base import Message

                conversation._messages[0] = Message(role="system", content=prompt)
