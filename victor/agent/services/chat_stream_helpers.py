# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared streaming helper implementations for the canonical chat service path."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.paradigm_router import get_paradigm_router
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements
from victor.agent.topology_contract import TopologyAction
from victor.agent.topology_grounder import TopologyGrounder
from victor.agent.topology_selector import TopologySelector
from victor.agent.topology_telemetry import (
    build_topology_telemetry_event,
    emit_topology_telemetry_event,
)
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.core.loop_thresholds import DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.framework.task.direct_response import classify_direct_response_prompt
from victor.framework.topology_runtime import prepare_topology_runtime_contract
from victor.framework.task import TaskComplexity
from victor.providers.base import Message, StreamChunk

if TYPE_CHECKING:
    from victor.agent.streaming.context import StreamingChatContext

logger = logging.getLogger(__name__)
_MISSING = object()


class ChatStreamHelperMixin:
    """Shared streaming helper methods reused by service and compatibility shims."""

    @staticmethod
    def _get_runtime_state_host(runtime_host: Any) -> Any:
        """Return the concrete object that owns raw runtime instance state."""
        resolver = (
            getattr(runtime_host, "_resolve_runtime_state_host", None)
            if hasattr(type(runtime_host), "_resolve_runtime_state_host")
            else None
        )
        if callable(resolver):
            return resolver(runtime_host)
        instance_dict = getattr(runtime_host, "__dict__", {})
        if isinstance(instance_dict, dict) and "_orchestrator" in instance_dict:
            return instance_dict["_orchestrator"]
        return runtime_host

    @classmethod
    def _get_runtime_state_dict(cls, runtime_host: Any) -> Dict[str, Any]:
        """Return the instance dictionary for raw runtime state lookups."""
        resolver = (
            getattr(runtime_host, "_resolve_runtime_state_dict", None)
            if hasattr(type(runtime_host), "_resolve_runtime_state_dict")
            else None
        )
        if callable(resolver):
            resolved = resolver(runtime_host)
            if isinstance(resolved, dict):
                return resolved
        state_host = cls._get_runtime_state_host(runtime_host)
        instance_dict = getattr(state_host, "__dict__", {})
        return instance_dict if isinstance(instance_dict, dict) else {}

    def _has_runtime_capability(self, name: str) -> bool:
        """Return whether the active runtime exposes a named capability."""
        resolver = getattr(self, "_resolve_runtime_capability_presence", None)
        if callable(resolver):
            return bool(resolver(name))

        orch = self._orchestrator
        checker = getattr(orch, "has_capability", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except Exception:
                logger.debug("Capability probe failed for %s", name, exc_info=True)

        state_dict = self._get_runtime_state_dict(orch)
        return name in state_dict or f"_{name}" in state_dict

    def _get_runtime_capability_value(self, name: str, default: Any = None) -> Any:
        """Return a named runtime capability or state-backed value."""
        resolver = getattr(self, "_resolve_runtime_capability_value", None)
        if callable(resolver):
            return resolver(name, default)

        orch = self._orchestrator
        getter = getattr(orch, "get_capability_value", None)
        if callable(getter):
            try:
                value = getter(name)
                if value is not None:
                    return value
            except Exception:
                logger.debug("Capability read failed for %s", name, exc_info=True)

        state_dict = self._get_runtime_state_dict(orch)
        if name in state_dict:
            return state_dict.get(name)
        return state_dict.get(f"_{name}", default)

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Compatibility delegate for context and iteration limit handling."""
        orch = self._orchestrator

        runtime_helper = None
        helper_resolver = getattr(self, "_get_orchestrator_runtime_helper", None)
        if callable(helper_resolver):
            runtime_helper = helper_resolver("_handle_context_and_iteration_limits_runtime")
        else:
            runtime_helper = getattr(orch, "_handle_context_and_iteration_limits_runtime", None)

        if callable(runtime_helper):
            return await runtime_helper(
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            )

        return False, None

    async def _prepare_stream(self, user_message: str, **kwargs: Any) -> tuple[
        Any,
        float,
        float,
        Dict[str, int],
        int,
        int,
        int,
        bool,
        TrackerTaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        orch = self._orchestrator

        orch._cancel_event = asyncio.Event()
        orch._is_streaming = True

        stream_metrics = orch._metrics_collector.init_stream_metrics()
        start_time = stream_metrics.start_time
        total_tokens: float = 0

        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        orch.conversation.ensure_system_prompt()
        orch._system_added = True
        orch._session_state.reset_for_new_turn()
        orch.unified_tracker.reset()
        orch.reminder_manager.reset()

        usage_analytics = self._get_runtime_capability_value("usage_analytics")
        if self._has_runtime_capability("usage_analytics") and usage_analytics:
            usage_analytics.start_session()

        tool_sequence_tracker = self._get_runtime_capability_value("tool_sequence_tracker")
        if self._has_runtime_capability("tool_sequence_tracker") and tool_sequence_tracker:
            tool_sequence_tracker.clear_history()

        if orch._context_manager and hasattr(orch._context_manager, "start_background_compaction"):
            await orch._context_manager.start_background_compaction(interval_seconds=15.0)

        max_total_iterations = orch.unified_tracker.config.get("max_total_iterations", 50)

        fallback_iteration = kwargs.get("_fallback_iteration", 0)
        if fallback_iteration > 0:
            total_iterations = fallback_iteration
            logger.info(
                f"[Fallback] Preserving iteration count: continuing from iteration {total_iterations} "
                f"(preserved {total_iterations} iterations of progress)"
            )
        else:
            total_iterations = 0

        force_completion = False

        orch.add_message("user", user_message)

        if self._has_runtime_capability("usage_analytics") and usage_analytics:
            usage_analytics.record_turn()

        unified_task_type = orch.unified_tracker.detect_task_type(user_message)
        logger.info(f"Task type detected: {unified_task_type.value}")

        prompt_requirements = extract_prompt_requirements(user_message)
        if prompt_requirements.has_explicit_requirements():
            orch.unified_tracker._progress.has_prompt_requirements = True

            if (
                prompt_requirements.tool_budget
                and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget
            ):
                orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)
                logger.info(
                    f"Dynamic budget from prompt: {prompt_requirements.tool_budget} "
                    f"(files={prompt_requirements.file_count}, fixes={prompt_requirements.fix_count})"
                )

            if (
                prompt_requirements.iteration_budget
                and prompt_requirements.iteration_budget
                > orch.unified_tracker._task_config.max_exploration_iterations
            ):
                orch.unified_tracker.set_max_iterations(prompt_requirements.iteration_budget)
                logger.info(
                    f"Dynamic iterations from prompt: {prompt_requirements.iteration_budget}"
                )

        direct_response = classify_direct_response_prompt(user_message)
        intelligent_task = None
        if not direct_response.is_direct_response:
            intelligent_task = asyncio.create_task(
                orch._prepare_intelligent_request(
                    task=user_message,
                    task_type=unified_task_type.value,
                )
            )

        max_exploration_iterations = orch.unified_tracker.max_exploration_iterations

        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )

        intelligent_context = await intelligent_task if intelligent_task is not None else None
        if intelligent_context:
            if intelligent_context.get("system_prompt_addition"):
                orch.add_message(
                    "user",
                    f"[SYSTEM-REMINDER: {intelligent_context['system_prompt_addition']}]",
                    metadata=build_internal_history_metadata("system_reminder"),
                )
                logger.debug("Injected intelligent pipeline optimized prompt")
        elif direct_response.is_direct_response:
            logger.debug("Skipping intelligent prompt injection for direct-response prompt")

        return (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        )

    async def _create_stream_context(
        self, user_message: str, **kwargs: Any
    ) -> "StreamingChatContext":
        """Create a StreamingChatContext with all prepared data."""
        from victor.agent.streaming import create_stream_context

        orch = self._orchestrator
        (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = await self._prepare_stream(user_message, **kwargs)

        task_keywords = orch._classify_task_keywords(user_message)

        ctx = create_stream_context(
            user_message=user_message,
            max_iterations=max_total_iterations,
            max_exploration=max_exploration_iterations,
            tool_budget=complexity_tool_budget,
            max_blocked_before_force=getattr(
                orch.settings,
                "recovery_blocked_consecutive_threshold",
                DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
            ),
        )

        ctx.stream_metrics = stream_metrics
        ctx.start_time = start_time
        ctx.total_tokens = total_tokens
        ctx.cumulative_usage = cumulative_usage
        ctx.total_iterations = total_iterations
        ctx.force_completion = force_completion
        ctx.unified_task_type = unified_task_type
        ctx.task_classification = task_classification
        ctx.complexity_tool_budget = complexity_tool_budget

        task_type_val = task_keywords.get("task_type", "default")
        # Use keyword-based classification as fallback when embedding classifier returns GENERAL
        # but keywords suggest analysis (e.g., "structural analysis", "framework analysis")
        keyword_is_analysis = task_keywords.get(
            "is_analysis_task",
            task_type_val in ("analysis", "analyze"),
        )
        unified_is_analysis = unified_task_type.value in ("analyze", "analysis")

        # Fallback: if unified says GENERAL but keywords say analysis, trust keywords
        ctx.is_analysis_task = (
            keyword_is_analysis
            or unified_is_analysis
            or (unified_task_type.value == "general" and keyword_is_analysis)
        )
        ctx.is_action_task = task_keywords.get(
            "is_action_task",
            task_type_val in ("action", "create_simple", "create_complex"),
        )
        ctx.needs_execution = task_keywords.get(
            "needs_execution",
            task_type_val in ("execution", "action"),
        )
        ctx.coarse_task_type = task_keywords.get("coarse_task_type", task_type_val)

        if task_classification and hasattr(task_classification, "complexity"):
            ctx.is_complex_task = task_classification.complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.ANALYSIS,
            )

        from victor.agent.services.turn_execution_runtime import TurnExecutor

        ctx.is_qa_task = TurnExecutor._is_question_only(user_message)
        ctx.goals = orch._tool_planner.infer_goals_from_message(user_message)
        ctx.tool_budget = orch.tool_budget
        ctx.tool_calls_used = orch.tool_calls_used
        ctx.task_completion_detector = orch._task_completion_detector
        await self._initialize_stream_topology_context(ctx, user_message)

        return ctx

    async def _initialize_stream_topology_context(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> None:
        """Build and ground a topology plan for the streaming runtime."""
        if not self._is_stream_topology_enabled() or stream_ctx.topology_plan is not None:
            return

        orch = self._orchestrator
        task_classification = stream_ctx.task_classification
        task_type = str(
            getattr(task_classification, "task_type", None)
            or stream_ctx.coarse_task_type
            or getattr(stream_ctx.unified_task_type, "value", "default")
            or "default"
        )
        tool_budget = (
            stream_ctx.complexity_tool_budget
            if stream_ctx.complexity_tool_budget is not None
            else stream_ctx.tool_budget
        )
        if tool_budget is None:
            tool_budget = getattr(orch, "tool_budget", 10) or 10

        routing_context: Dict[str, Any] = {
            "iteration_budget": stream_ctx.max_total_iterations,
            "tool_budget": int(tool_budget),
            "available_team_formations": self._default_team_formations(),
        }

        provider_hints = await self._get_stream_topology_provider_hints(
            task_type=task_type,
            context=routing_context,
        )
        if provider_hints:
            routing_context.update(provider_hints)
            provider_candidates: List[str] = []
            primary_provider = provider_hints.get("provider_hint")
            if isinstance(primary_provider, str) and primary_provider:
                provider_candidates.append(primary_provider)
            for fallback in provider_hints.get("fallback_chain", []):
                if isinstance(fallback, str) and fallback:
                    provider_candidates.append(fallback)
            if provider_candidates:
                routing_context["provider_candidates"] = list(dict.fromkeys(provider_candidates))
        runtime_intelligence = self._get_runtime_state_dict(orch).get("_runtime_intelligence")
        structured_routing_policy = None
        if runtime_intelligence is not None and hasattr(
            runtime_intelligence, "get_structured_routing_policy"
        ):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                structured_routing_policy = runtime_intelligence.get_structured_routing_policy(
                    query=user_message,
                    scope_context=learned_scope_context,
                )
            except Exception as exc:
                logger.debug("Streaming structured routing policy unavailable: %s", exc)
            else:
                if structured_routing_policy is not None:
                    if hasattr(structured_routing_policy, "to_dict"):
                        serialized_policy = structured_routing_policy.to_dict()
                        if isinstance(serialized_policy, dict):
                            stream_ctx.structured_routing_policy = serialized_policy
                    learned_topology_context = structured_routing_policy.selector_context()
                    if isinstance(learned_topology_context, dict) and learned_topology_context:
                        routing_context.update(learned_topology_context)
        elif runtime_intelligence is not None and hasattr(
            runtime_intelligence, "get_topology_routing_context"
        ):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                learned_topology_context = runtime_intelligence.get_topology_routing_context(
                    query=user_message, scope_context=learned_scope_context
                )
            except Exception as exc:
                logger.debug("Streaming topology feedback hints unavailable: %s", exc)
            else:
                if learned_topology_context:
                    routing_context.update(learned_topology_context)

        paradigm_router = getattr(self, "_paradigm_router", None)
        if paradigm_router is None:
            paradigm_router = get_paradigm_router()
            self._paradigm_router = paradigm_router

        topology_selector = getattr(self, "_topology_selector", None)
        if topology_selector is None:
            topology_selector = TopologySelector()
            self._topology_selector = topology_selector

        topology_grounder = getattr(self, "_topology_grounder", None)
        if topology_grounder is None:
            topology_grounder = TopologyGrounder()
            self._topology_grounder = topology_grounder

        topology_input = paradigm_router.build_topology_input(
            task_type=task_type,
            query=user_message,
            history_length=len(getattr(orch, "messages", []) or []),
            query_complexity=self._stream_query_complexity(stream_ctx),
            tool_budget=int(tool_budget),
            context=routing_context,
        )
        topology_decision = topology_selector.select(topology_input)
        topology_plan = topology_grounder.ground(topology_decision)
        prepared_runtime = prepare_topology_runtime_contract(
            topology_plan,
            orchestrator=orch,
            task_type=task_type,
            complexity=str(
                getattr(
                    getattr(task_classification, "complexity", None),
                    "value",
                    getattr(task_classification, "complexity", None),
                )
                or "medium"
            ),
        )
        topology_overrides = dict(prepared_runtime.runtime_context_overrides)

        stream_ctx.topology_input = topology_input.to_dict()
        stream_ctx.topology_decision = topology_decision.to_dict()
        stream_ctx.topology_plan = topology_plan.to_dict()
        stream_ctx.topology_preparation = prepared_runtime.to_result(
            prepared=prepared_runtime.team_plan is not None
        )
        if prepared_runtime.team_plan is not None:
            stream_ctx.topology_plan["team_name"] = prepared_runtime.team_plan.team_name
            stream_ctx.topology_plan["team_display_name"] = prepared_runtime.team_plan.display_name
            stream_ctx.topology_plan["member_count"] = prepared_runtime.team_plan.member_count
        stream_ctx.runtime_context_overrides = dict(topology_overrides)
        stream_ctx.provider_kwargs = self._stream_provider_call_overrides(topology_overrides)

        if topology_plan.tool_budget is not None:
            adjusted_tool_budget = max(0, int(topology_plan.tool_budget))
            stream_ctx.tool_budget = adjusted_tool_budget
            stream_ctx.complexity_tool_budget = adjusted_tool_budget
        if topology_plan.iteration_budget is not None:
            adjusted_iteration_budget = max(1, int(topology_plan.iteration_budget))
            stream_ctx.max_total_iterations = adjusted_iteration_budget
            stream_ctx.max_exploration_iterations = min(
                max(1, stream_ctx.max_exploration_iterations),
                adjusted_iteration_budget,
            )
        if topology_decision.action == TopologyAction.DIRECT_RESPONSE:
            stream_ctx.is_qa_task = True

        stream_ctx.runtime_override_snapshot = self._apply_stream_runtime_overrides(
            topology_overrides
        )

        topology_event = build_topology_telemetry_event(
            topology_input,
            topology_decision,
            grounded_plan=topology_plan,
            outcome={"status": "planned", "runtime": "streaming"},
        )
        stream_ctx.topology_events.append(topology_event.to_dict())
        await emit_topology_telemetry_event(topology_event)

        logger.info(
            "[StreamingTopology] action=%s topology=%s provider=%s formation=%s",
            topology_decision.action.value,
            topology_decision.topology.value,
            topology_plan.provider,
            topology_plan.formation,
        )

    def _is_stream_topology_enabled(self) -> bool:
        """Return whether streaming topology routing is enabled."""
        explicit = getattr(self, "_topology_enabled", None)
        if explicit is not None:
            return bool(explicit)
        settings = getattr(self._orchestrator, "settings", None)
        if settings is not None and hasattr(settings, "enable_topology_routing"):
            return bool(settings.enable_topology_routing)
        return True

    async def _get_stream_topology_provider_hints(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch provider-routing hints when the active provider exposes them."""
        orch = self._orchestrator
        provider = getattr(orch, "provider", None)
        model_hint = getattr(orch, "model", None)
        preferred_providers = context.get("preferred_providers")

        hint_sources = [
            getattr(provider, "get_topology_provider_hints", None),
            getattr(getattr(provider, "engine", None), "get_topology_provider_hints", None),
        ]
        for get_hints in hint_sources:
            if not callable(get_hints):
                continue
            try:
                hints = await get_hints(
                    task_type=task_type,
                    model_hint=model_hint,
                    preferred_providers=preferred_providers,
                )
            except Exception as exc:
                logger.debug("Streaming topology provider hints unavailable: %s", exc)
                continue
            if isinstance(hints, dict):
                return hints
        return {}

    @staticmethod
    def _stream_query_complexity(stream_ctx: "StreamingChatContext") -> Optional[float]:
        """Convert streaming task complexity into a stable numeric score."""
        task_classification = stream_ctx.task_classification
        complexity = getattr(task_classification, "complexity", None)
        if complexity is not None:
            complexity_value = getattr(complexity, "value", str(complexity)).lower()
            if (
                "analysis" in complexity_value
                or "complex" in complexity_value
                or "high" in complexity_value
            ):
                return 0.8
            if "medium" in complexity_value or "moderate" in complexity_value:
                return 0.5
            return 0.2

        if stream_ctx.is_analysis_task or stream_ctx.is_complex_task:
            return 0.8
        if stream_ctx.is_action_task:
            return 0.5
        return 0.2

    @staticmethod
    def _default_team_formations() -> List[str]:
        """Return canonical team formation hints for topology grounding."""
        from victor.teams.types import TeamFormation

        return [formation.value for formation in TeamFormation]

    @staticmethod
    def _stream_provider_call_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Extract provider-facing runtime override hints."""
        provider_keys = (
            "provider_hint",
            "execution_mode",
            "escalation_target",
            "topology_action",
            "topology_kind",
            "topology_metadata",
        )
        return {
            key: value
            for key, value in overrides.items()
            if key in provider_keys and value is not None
        }

    def _apply_stream_runtime_overrides(
        self,
        overrides: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply temporary runtime overrides for the streaming turn."""
        if not overrides:
            return None

        orch = self._orchestrator
        state_dict = self._get_runtime_state_dict(orch)
        snapshot: Dict[str, Any] = {
            "orchestrator_runtime_context": state_dict.get(
                "_runtime_tool_context_overrides",
                _MISSING,
            ),
        }

        merged_context: Dict[str, Any] = {}
        previous_runtime_context = snapshot["orchestrator_runtime_context"]
        if isinstance(previous_runtime_context, dict):
            merged_context.update(previous_runtime_context)
        merged_context.update(overrides)
        orch._runtime_tool_context_overrides = merged_context

        tool_budget = self._coerce_stream_int_override(overrides.get("tool_budget"))
        if tool_budget is not None:
            if hasattr(orch, "tool_budget"):
                snapshot["orchestrator_tool_budget"] = getattr(orch, "tool_budget", _MISSING)
                try:
                    orch.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

            task_coordinator = getattr(orch, "task_coordinator", None)
            if task_coordinator is not None and hasattr(task_coordinator, "tool_budget"):
                snapshot["task_coordinator_tool_budget"] = getattr(
                    task_coordinator,
                    "tool_budget",
                    _MISSING,
                )
                try:
                    task_coordinator.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

            tool_service = getattr(orch, "_tool_service", None)
            if tool_service is not None and hasattr(tool_service, "get_tool_budget"):
                snapshot["tool_service_budget"] = getattr(
                    tool_service,
                    "budget",
                    (
                        tool_service.get_budget_info().get("max")
                        if hasattr(tool_service, "get_budget_info")
                        else tool_service.get_tool_budget()
                    ),
                )
                if hasattr(tool_service, "set_tool_budget"):
                    try:
                        tool_service.set_tool_budget(max(0, tool_budget))
                    except Exception:
                        pass

            tool_pipeline = getattr(orch, "_tool_pipeline", None)
            pipeline_config = getattr(tool_pipeline, "config", None)
            if pipeline_config is not None and hasattr(pipeline_config, "tool_budget"):
                snapshot["pipeline_tool_budget"] = getattr(
                    pipeline_config,
                    "tool_budget",
                    _MISSING,
                )
                try:
                    pipeline_config.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

        return snapshot

    def _restore_stream_runtime_overrides(
        self,
        snapshot: Optional[Dict[str, Any]],
    ) -> None:
        """Restore runtime state after one streaming turn completes."""
        if not snapshot:
            return

        orch = self._orchestrator
        state_dict = self._get_runtime_state_dict(orch)
        previous_runtime_context = snapshot.get("orchestrator_runtime_context", _MISSING)
        if previous_runtime_context is _MISSING:
            if "_runtime_tool_context_overrides" in state_dict:
                delattr(orch, "_runtime_tool_context_overrides")
        else:
            orch._runtime_tool_context_overrides = previous_runtime_context

        previous_orchestrator_budget = snapshot.get("orchestrator_tool_budget", _MISSING)
        if previous_orchestrator_budget is not _MISSING and hasattr(orch, "tool_budget"):
            try:
                orch.tool_budget = previous_orchestrator_budget
            except Exception:
                pass

        task_coordinator = getattr(orch, "task_coordinator", None)
        previous_task_coordinator_budget = snapshot.get("task_coordinator_tool_budget", _MISSING)
        if (
            previous_task_coordinator_budget is not _MISSING
            and task_coordinator is not None
            and hasattr(task_coordinator, "tool_budget")
        ):
            try:
                task_coordinator.tool_budget = previous_task_coordinator_budget
            except Exception:
                pass

        tool_service = getattr(orch, "_tool_service", None)
        previous_service_budget = snapshot.get("tool_service_budget", _MISSING)
        if (
            previous_service_budget is not _MISSING
            and tool_service is not None
            and hasattr(tool_service, "set_tool_budget")
        ):
            try:
                tool_service.set_tool_budget(previous_service_budget)
            except Exception:
                pass

        tool_pipeline = getattr(orch, "_tool_pipeline", None)
        pipeline_config = getattr(tool_pipeline, "config", None)
        previous_pipeline_budget = snapshot.get("pipeline_tool_budget", _MISSING)
        if (
            previous_pipeline_budget is not _MISSING
            and pipeline_config is not None
            and hasattr(pipeline_config, "tool_budget")
        ):
            try:
                pipeline_config.tool_budget = previous_pipeline_budget
            except Exception:
                pass

    @staticmethod
    def _coerce_stream_int_override(value: Any) -> Optional[int]:
        """Convert runtime override values into integers when possible."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        orch = self._orchestrator

        if orch.task_coordinator._reminder_manager is None:
            orch.task_coordinator.set_reminder_manager(orch.reminder_manager)

        return orch.task_coordinator.prepare_task(
            user_message, unified_task_type, orch.conversation_controller
        )

    async def _run_iteration_pre_checks(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-iteration checks: cancellation, compaction, time limit."""
        orch = self._orchestrator

        if orch._check_cancellation():
            logger.info("Stream cancelled by user request")
            orch._is_streaming = False
            orch._record_intelligent_outcome(
                success=False,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=False,
                completed=False,
            )
            yield StreamChunk(
                content="\n\n[Cancelled by user]\n",
                is_final=True,
            )
            return

        if orch._context_compactor:
            compaction_action = orch._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=orch.tool_calls_used,
                task_complexity=TaskComplexity.COMPLEX.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )
                # P0 FIX: Set compaction state on stream context for post-compaction continuation
                stream_ctx.compaction_occurred = True
                stream_ctx.last_compaction_turn = stream_ctx.total_iterations
                stream_ctx.compaction_message_removed_count = compaction_action.messages_removed
                # Get summary from controller if available
                if hasattr(orch, "conversation_controller") and orch.conversation_controller:
                    summaries = orch.conversation_controller.get_compaction_summaries()
                    if summaries:
                        stream_ctx.compaction_summary = summaries[-1]
                logger.info(
                    f"Post-compaction continuation enabled at turn {stream_ctx.total_iterations}"
                )

        time_limit = getattr(orch.settings, "stream_idle_timeout_seconds", 300)
        if stream_ctx.is_over_time_limit(time_limit):
            logger.warning(f"Stream time limit exceeded: {stream_ctx.elapsed_time():.1f}s")
            yield StreamChunk(
                content=f"\n\n[Session exceeded {time_limit}s idle timeout - providing summary]\n",
                is_final=False,
            )
            stream_ctx.force_completion = True

        stream_ctx.increment_iteration()

        if stream_ctx.pending_grounding_feedback:
            logger.info("Injecting pending grounding feedback as system message")
            orch.add_message(
                "user",
                f"[GROUNDING-FEEDBACK: {stream_ctx.pending_grounding_feedback}]",
                metadata=build_internal_history_metadata("grounding_feedback"),
            )
            stream_ctx.pending_grounding_feedback = ""

    async def _stream_provider_response(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Stream response from provider with rate limit retry."""
        return await self._stream_with_rate_limit_retry(tools, provider_kwargs, stream_ctx)

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry."""
        orch = self._orchestrator
        base_wait = orch._provider_service.get_rate_limit_wait_time(exc)
        backoff_multiplier = 2**attempt
        wait_time = base_wait * backoff_multiplier
        return min(wait_time, 300.0)

    async def _stream_with_rate_limit_retry(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
        max_retries: int = 3,
    ) -> tuple[str, Any, float, bool]:
        """Stream provider response with automatic rate limit retry."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._stream_provider_response_inner(
                    tools, provider_kwargs, stream_ctx
                )
            except ProviderRateLimitError as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait_time = self._get_rate_limit_wait_time(exc, attempt)
                    endpoint_info = (
                        f"{self.provider_name}:{self.model}"
                        if hasattr(self, "provider_name")
                        else "API"
                    )
                    logger.warning(
                        f"[yellow]⚠ Rate limit hit for {endpoint_info}[/] "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.0f}s before retry..."
                    )
                    logger.debug(f"Rate limit detail: {str(exc)[:300]}")

                    if attempt == 0:
                        logger.info(
                            "[dim]Tip: To avoid rate limits, consider:[/]\n"
                            "[dim]  • Using an API key instead of free tier[/]\n"
                            "[dim]  • Adding a small delay between requests[/]\n"
                            "[dim]  • Reducing request frequency[/]"
                        )

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str:
                    last_exception = exc
                    if attempt < max_retries:
                        wait_time = self._get_rate_limit_wait_time(exc, attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        logger.debug(f"Rate limit detail: {type(exc).__name__}: {str(exc)[:200]}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Rate limit retry exhausted without exception")

    async def _stream_provider_response_inner(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Inner implementation of stream_provider_response without retry logic."""
        orch = self._orchestrator

        full_content = ""
        tool_calls = None
        garbage_detected = False
        consecutive_garbage_chunks = 0
        max_garbage_chunks = 3
        total_tokens: float = 0

        assembled = orch.get_assembled_messages(
            current_query=stream_ctx.user_message if stream_ctx else None
        )
        async for chunk in orch.provider.stream(
            messages=assembled,
            model=orch.model,
            temperature=orch.temperature,
            max_tokens=orch.max_tokens,
            tools=tools,
            **provider_kwargs,
        ):
            chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                chunk,
                consecutive_garbage_chunks,
                max_garbage_chunks,
                garbage_detected,
            )
            if chunk is None:
                continue

            full_content += chunk.content
            stream_ctx.stream_metrics.total_chunks += 1
            if chunk.content:
                orch._metrics_collector.record_first_token()
                total_tokens += len(chunk.content) / 4
                stream_ctx.stream_metrics.total_content_length += len(chunk.content)

            if chunk.tool_calls:
                logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                tool_calls = chunk.tool_calls
                stream_ctx.stream_metrics.tool_calls_count += len(chunk.tool_calls)

            if chunk.usage:
                for key in stream_ctx.cumulative_usage:
                    stream_ctx.cumulative_usage[key] += chunk.usage.get(key, 0)
                logger.debug(
                    f"Chunk usage: in={chunk.usage.get('prompt_tokens', 0)} "
                    f"out={chunk.usage.get('completion_tokens', 0)} "
                    f"cache_read={chunk.usage.get('cache_read_input_tokens', 0)}"
                )

            if tool_calls:
                # Tool calls signal the end of this turn — the SSE stream is done.
                # Break immediately; do NOT start a new stream call with the same
                # messages (that would duplicate full_content).
                logger.debug("Tool calls received, breaking stream loop")
                break

        if garbage_detected and not tool_calls:
            logger.info("Setting force_completion due to garbage detection")

        if not tool_calls and not stream_ctx.force_completion:
            content_length = len(full_content.strip()) if full_content else 0

            if content_length == 0:
                logger.warning("Stream completed without content or tool calls")
                full_content = "[No content received from provider - stream may have failed]"
            elif content_length < 50:
                logger.warning(f"Stream completed with very short content ({content_length} chars)")
                logger.debug(f"Short content: {full_content}")

        logger.debug(
            "Stream completion summary - content: %s chars, tool_calls: %s",
            len(full_content) if full_content else 0,
            len(tool_calls) if tool_calls else 0,
        )

        stream_ctx.total_tokens = total_tokens
        return full_content, tool_calls, total_tokens, garbage_detected

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int,
        garbage_detected: bool,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk."""
        orch = self._orchestrator
        if chunk.content and orch.sanitizer.is_garbage_content(chunk.content):
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    garbage_detected = True
                    logger.warning(
                        f"Garbage content detected after {len(chunk.content)} chars - stopping stream early"
                    )
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0
        return chunk, consecutive_garbage_chunks, garbage_detected

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: "StreamingChatContext",
        tools: Any,
    ) -> tuple[bool, Any, Optional[StreamChunk]]:
        """Handle empty response recovery with multi-strategy retry."""
        orch = self._orchestrator

        recovery_temps = [0.7, 0.9]
        for temp in recovery_temps:
            try:
                orch.add_message(
                    "system",
                    "Please provide a response to the user's question. "
                    "If you need to use tools, go ahead. Otherwise, provide a text answer.",
                )

                provider_kwargs: Dict[str, Any] = dict(
                    getattr(stream_ctx, "provider_kwargs", {}) or {}
                )
                if (
                    orch.thinking
                    or provider_kwargs.get("execution_mode") == "escalated_single_agent"
                ):
                    provider_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 10000,
                    }

                full_content = ""
                recovered_tool_calls = None

                retry_assembled = orch.get_assembled_messages(
                    current_query=stream_ctx.user_message if stream_ctx else None
                )
                async for chunk in orch.provider.stream(
                    messages=retry_assembled,
                    model=orch.model,
                    temperature=temp,
                    max_tokens=orch.max_tokens,
                    tools=tools,
                    **provider_kwargs,
                ):
                    if chunk.content:
                        full_content += chunk.content
                    if chunk.tool_calls:
                        recovered_tool_calls = chunk.tool_calls
                        break

                if recovered_tool_calls:
                    logger.info(f"Recovery at temperature {temp} produced tool calls")
                    return True, recovered_tool_calls, None

                if full_content.strip():
                    logger.info(
                        f"Recovery at temperature {temp} produced content "
                        f"({len(full_content)} chars)"
                    )
                    sanitized = orch.sanitizer.sanitize(full_content)
                    if sanitized:
                        orch.add_message("assistant", sanitized)
                    final_chunk = orch._chunk_generator.generate_content_chunk(
                        sanitized or full_content, is_final=True
                    )
                    return True, None, final_chunk

            except Exception as exc:
                logger.warning(f"Recovery attempt at temperature {temp} failed: {exc}")
                continue

        return False, None, None
