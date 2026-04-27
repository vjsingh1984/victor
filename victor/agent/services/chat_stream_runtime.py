# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Private streaming runtime adapter for the canonical chat service path."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin

if TYPE_CHECKING:
    from victor.providers.base import StreamChunk
    from victor.agent.streaming.pipeline import StreamingChatPipeline
    from victor.agent.streaming.intent_classification import IntentClassificationHandler
    from victor.agent.streaming.continuation import ContinuationHandler
    from victor.agent.streaming.tool_execution import ToolExecutionHandler

logger = logging.getLogger(__name__)


class ServiceStreamingRuntime(ChatStreamHelperMixin):
    """Service-owned runtime adapter for the canonical streaming pipeline.

    This adapter keeps the live streaming path off the deprecated ChatCoordinator
    shim while reusing the shared service-owned streaming helper implementations.
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator
        self._intent_classification_handler: Optional["IntentClassificationHandler"] = None
        self._continuation_handler: Optional["ContinuationHandler"] = None
        self._tool_execution_handler: Optional["ToolExecutionHandler"] = None
        self._streaming_pipeline: Optional["StreamingChatPipeline"] = None

    def get_pipeline(self) -> "StreamingChatPipeline":
        """Get or create the canonical streaming pipeline for the service path."""
        if self._streaming_pipeline is None:
            from victor.agent.streaming import create_streaming_chat_pipeline
            from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService

            orch = self._orchestrator
            perception = (
                orch.get_capability_value("perception_integration")
                if orch.has_capability("perception_integration")
                else None
            )
            fulfillment = (
                orch.get_capability_value("fulfillment_detector")
                if orch.has_capability("fulfillment_detector")
                else None
            )
            runtime_intelligence = getattr(getattr(orch, "__dict__", {}), "get", lambda *_: None)(
                "_runtime_intelligence"
            )
            if runtime_intelligence is None:
                runtime_intelligence = RuntimeIntelligenceService.from_orchestrator(
                    orch,
                    perception_integration=perception,
                    optimization_injector=getattr(orch, "_optimization_injector", None),
                )
                orch._runtime_intelligence = runtime_intelligence
            self._streaming_pipeline = create_streaming_chat_pipeline(
                self,
                runtime_intelligence=runtime_intelligence,
                perception=perception,
                fulfillment=fulfillment,
            )
        return self._streaming_pipeline

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator["StreamChunk"]:
        """Stream a response through the canonical service-owned pipeline."""
        _ = kwargs.pop("_preserve_iteration", None)
        fallback_iteration = kwargs.pop("_fallback_iteration", 0)

        if fallback_iteration > 0:
            logger.info(
                "[ServiceStreamingRuntime] Using fallback iteration: %s", fallback_iteration
            )
            kwargs["_fallback_iteration"] = fallback_iteration

        orch = self._orchestrator
        pipeline = self.get_pipeline()

        try:
            async for chunk in pipeline.run(user_message, **kwargs):
                yield chunk
        finally:
            ctx = None
            if orch.has_capability("current_stream_context") and orch.get_capability_value(
                "current_stream_context"
            ):
                ctx = orch.get_capability_value("current_stream_context")
            else:
                instance_dict = getattr(orch, "__dict__", {})
                if "_current_stream_context" in instance_dict:
                    ctx = instance_dict.get("_current_stream_context")

            if ctx is not None:
                if hasattr(ctx, "cumulative_usage"):
                    for key in orch._cumulative_token_usage:
                        if key in ctx.cumulative_usage:
                            orch._cumulative_token_usage[key] += ctx.cumulative_usage[key]
                    if orch._cumulative_token_usage["total_tokens"] == 0:
                        orch._cumulative_token_usage["total_tokens"] = (
                            orch._cumulative_token_usage["prompt_tokens"]
                            + orch._cumulative_token_usage["completion_tokens"]
                        )

                    prompt_tokens = ctx.cumulative_usage.get("prompt_tokens", 0)
                    if prompt_tokens > 0:
                        try:
                            ctrl = orch._conversation_controller
                            total_chars = sum(len(m.content) for m in ctrl.messages)
                            ctrl.record_actual_usage(prompt_tokens, total_chars)
                        except Exception:
                            pass

                runtime_snapshot = getattr(ctx, "runtime_override_snapshot", None)
                self._restore_stream_runtime_overrides(runtime_snapshot)
                ctx.runtime_override_snapshot = None

            if "_current_stream_context" in getattr(orch, "__dict__", {}):
                orch._current_stream_context = None
