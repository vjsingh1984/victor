# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Private streaming runtime adapter for the canonical chat service path."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Mapping, Optional

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin

if TYPE_CHECKING:
    from victor.providers.base import StreamChunk
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
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
        self._streaming_executor: Optional["StreamingChatExecutor"] = None

    @staticmethod
    def _iter_runtime_dicts(orch: Any) -> tuple[dict[str, Any], ...]:
        """Yield direct and adapter-host instance dictionaries for runtime state lookup."""
        instance_dict = getattr(orch, "__dict__", {})
        dicts = [instance_dict] if isinstance(instance_dict, dict) else []
        host = instance_dict.get("_orchestrator") if isinstance(instance_dict, dict) else None
        host_dict = getattr(host, "__dict__", None)
        if isinstance(host_dict, dict) and host_dict is not instance_dict:
            dicts.append(host_dict)
        return tuple(dicts)

    @staticmethod
    def _has_capability(orch: Any, name: str) -> bool:
        """Resolve capability presence across direct orchestrators and adapters."""
        checker = getattr(orch, "has_capability", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except Exception:
                logger.debug("Capability probe failed for %s", name, exc_info=True)
        for instance_dict in ServiceStreamingRuntime._iter_runtime_dicts(orch):
            if name in instance_dict or f"_{name}" in instance_dict:
                return True
        return False

    @staticmethod
    def _get_capability_value(orch: Any, name: str) -> Any:
        """Resolve capability values across direct orchestrators and adapters."""
        getter = getattr(orch, "get_capability_value", None)
        if callable(getter):
            try:
                value = getter(name)
                if value is not None:
                    return value
            except Exception:
                logger.debug("Capability read failed for %s", name, exc_info=True)
        for instance_dict in ServiceStreamingRuntime._iter_runtime_dicts(orch):
            if name in instance_dict:
                return instance_dict.get(name)
            underscored = f"_{name}"
            if underscored in instance_dict:
                return instance_dict.get(underscored)
        return None

    def get_executor(self) -> "StreamingChatExecutor":
        """Get or create the canonical service-owned streaming executor."""
        if self._streaming_executor is None:
            from victor.agent.services.chat_stream_executor import create_streaming_chat_executor
            from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService

            orch = self._orchestrator
            state_host = self._get_runtime_state_host(orch)
            state_dict = self._get_runtime_state_dict(orch)
            perception = self._get_capability_value(orch, "perception_integration")
            fulfillment = self._get_capability_value(orch, "fulfillment_detector")
            runtime_intelligence = state_dict.get("_runtime_intelligence")
            if runtime_intelligence is None:
                runtime_intelligence = RuntimeIntelligenceService.from_orchestrator(
                    orch,
                    perception_integration=perception,
                    optimization_injector=state_dict.get("_optimization_injector"),
                )
                state_host._runtime_intelligence = runtime_intelligence
            self._streaming_executor = create_streaming_chat_executor(
                self,
                runtime_intelligence=runtime_intelligence,
                perception=perception,
                fulfillment=fulfillment,
            )
        return self._streaming_executor

    @staticmethod
    def _coerce_unit_float(value: Any, default: float = 0.0) -> float:
        """Normalize optional quality values into the canonical [0, 1] range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(getattr(value, "value", value)).strip()
        return text or None

    def _classify_stream_outcome(self, ctx: Any, *, failed: bool = False) -> Dict[str, Any]:
        """Derive a shared stream outcome snapshot for telemetry finalizers."""
        quality_score = self._coerce_unit_float(getattr(ctx, "last_quality_score", 0.0))
        has_substantial_content = False
        if hasattr(ctx, "has_substantial_content"):
            try:
                has_substantial_content = bool(ctx.has_substantial_content())
            except Exception:
                has_substantial_content = False

        if failed:
            status = "error"
        elif quality_score >= 0.7:
            status = "completed"
        elif has_substantial_content or quality_score >= 0.45:
            status = "resolved"
        else:
            status = "failed"

        return {
            "status": status,
            "quality_score": quality_score,
            "has_substantial_content": has_substantial_content,
        }

    def _normalize_stream_recovery_event(
        self,
        event: Any,
        *,
        recovered: bool,
        failed: bool,
    ) -> Optional[Dict[str, Any]]:
        """Translate raw stream recovery telemetry into degradation-event schema."""
        if hasattr(event, "to_dict"):
            event = event.to_dict()
        if not isinstance(event, Mapping):
            return None

        action = self._coerce_optional_text(event.get("action"))
        reason = self._coerce_optional_text(event.get("reason"))
        strategy_name = self._coerce_optional_text(event.get("strategy_name"))
        failure_type = self._coerce_optional_text(event.get("failure_type")) or "STREAMING_RECOVERY"
        reasons = []
        for value in (action, strategy_name, reason):
            if value and value not in reasons:
                reasons.append(value)

        adaptation_cost = event.get("adaptation_cost")
        try:
            adaptation_cost_value = float(adaptation_cost) if adaptation_cost is not None else 1.0
        except (TypeError, ValueError):
            adaptation_cost_value = 1.0

        confidence = event.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        normalized = {
            "source": self._coerce_optional_text(event.get("source")) or "streaming_recovery",
            "kind": self._coerce_optional_text(event.get("kind")) or "recovery_action",
            "failure_type": failure_type,
            "provider": self._coerce_optional_text(event.get("provider")),
            "model": self._coerce_optional_text(event.get("model")),
            "task_type": self._coerce_optional_text(event.get("task_type")),
            "pre_degraded": True,
            "post_degraded": bool(failed),
            "recovered": bool(recovered),
            "adaptation_cost": adaptation_cost_value,
            "iteration": event.get("iteration"),
            "degradation_reasons": reasons,
            "action": action,
            "reason": reason,
            "strategy_name": strategy_name,
            "confidence": confidence_value,
            "fallback_provider": self._coerce_optional_text(event.get("fallback_provider")),
            "fallback_model": self._coerce_optional_text(event.get("fallback_model")),
        }
        return normalized

    def _build_stream_degradation_feedback_payload(
        self,
        ctx: Any,
        *,
        failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Finalize stream degradation/recovery telemetry into evaluation-ready events."""
        degradation_events = list(getattr(ctx, "degradation_events", []) or [])
        recovery_events = list(getattr(ctx, "recovery_events", []) or [])
        if not degradation_events and not recovery_events:
            return None

        outcome = self._classify_stream_outcome(ctx, failed=failed)
        recovered = outcome["status"] in {"completed", "resolved"} and not failed
        for recovery_event in recovery_events:
            normalized = self._normalize_stream_recovery_event(
                recovery_event,
                recovered=recovered,
                failed=failed,
            )
            if normalized is not None:
                degradation_events.append(normalized)

        return {
            "status": outcome["status"],
            "completion_score": outcome["quality_score"],
            "degradation_events": degradation_events,
            "recovery_events": recovery_events,
        }

    def _build_stream_topology_feedback_payload(
        self,
        ctx: Any,
        *,
        failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Build a live topology-feedback payload from the completed stream context."""
        topology_events = list(getattr(ctx, "topology_events", []) or [])
        if not topology_events:
            return None

        outcome_summary = self._classify_stream_outcome(ctx, failed=failed)
        status = outcome_summary["status"]
        quality_score = outcome_summary["quality_score"]
        has_substantial_content = outcome_summary["has_substantial_content"]

        outcome = {
            "status": status,
            "completion_score": quality_score,
            "tool_calls": getattr(ctx, "tool_calls_used", 0),
            "turns": getattr(ctx, "total_iterations", 0),
            "runtime": "streaming",
            "force_completion": bool(getattr(ctx, "force_completion", False)),
            "has_substantial_content": has_substantial_content,
        }
        topology_events[-1]["outcome"] = dict(outcome)

        return {
            "status": status,
            "completion_score": quality_score,
            "tool_calls": outcome["tool_calls"],
            "turns": outcome["turns"],
            "topology_events": topology_events,
        }

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
        state_host = self._get_runtime_state_host(orch)
        executor = self.get_executor()
        stream_failed = False

        try:
            async for chunk in executor.run(user_message, **kwargs):
                yield chunk
        except Exception:
            stream_failed = True
            raise
        finally:
            ctx = None
            current_stream_context = self._get_capability_value(orch, "current_stream_context")
            if current_stream_context is not None:
                ctx = current_stream_context
            else:
                ctx = self._get_runtime_state_dict(orch).get("_current_stream_context")

            if ctx is not None:
                state_dict = self._get_runtime_state_dict(orch)
                if hasattr(ctx, "cumulative_usage"):
                    cumulative_usage = state_dict.get("_cumulative_token_usage")
                    for key in cumulative_usage or {}:
                        if key in ctx.cumulative_usage:
                            cumulative_usage[key] += ctx.cumulative_usage[key]
                    if isinstance(cumulative_usage, dict) and cumulative_usage["total_tokens"] == 0:
                        cumulative_usage["total_tokens"] = (
                            cumulative_usage["prompt_tokens"]
                            + cumulative_usage["completion_tokens"]
                        )

                    prompt_tokens = ctx.cumulative_usage.get("prompt_tokens", 0)
                    if prompt_tokens > 0:
                        try:
                            ctrl = state_dict.get("_conversation_controller")
                            total_chars = sum(len(m.content) for m in ctrl.messages)
                            ctrl.record_actual_usage(prompt_tokens, total_chars)
                        except Exception:
                            pass

                topology_feedback_payload = self._build_stream_topology_feedback_payload(
                    ctx,
                    failed=stream_failed,
                )
                if topology_feedback_payload is not None:
                    ctx.topology_events = list(topology_feedback_payload["topology_events"])
                    runtime_intelligence = state_dict.get("_runtime_intelligence")
                    if runtime_intelligence is not None and hasattr(
                        runtime_intelligence, "record_topology_outcome"
                    ):
                        try:
                            runtime_intelligence.record_topology_outcome(topology_feedback_payload)
                        except Exception as exc:
                            logger.debug(
                                "Failed to record streaming topology runtime outcome: %s",
                                exc,
                            )

                degradation_feedback_payload = self._build_stream_degradation_feedback_payload(
                    ctx,
                    failed=stream_failed,
                )
                if degradation_feedback_payload is not None:
                    ctx.degradation_events = list(
                        degradation_feedback_payload["degradation_events"]
                    )
                    ctx.recovery_events = list(degradation_feedback_payload["recovery_events"])

                runtime_snapshot = getattr(ctx, "runtime_override_snapshot", None)
                self._restore_stream_runtime_overrides(runtime_snapshot)
                ctx.runtime_override_snapshot = None

            if "_current_stream_context" in self._get_runtime_state_dict(orch):
                state_host._current_stream_context = None
