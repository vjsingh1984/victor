from __future__ import annotations

"""Degradation and recovery feedback summarization helpers."""

from collections import Counter
from typing import Any, Iterable, Mapping, Optional


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_sequence(value: Any, key: str) -> list[Any]:
    raw_value = _extract_value(value, key, [])
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return []


def _normalize_event(event: Any) -> Optional[dict[str, Any]]:
    if event is None:
        return None
    if hasattr(event, "to_dict"):
        event = event.to_dict()
    if not isinstance(event, Mapping):
        return None

    source = _coerce_optional_text(event.get("source"))
    kind = _coerce_optional_text(event.get("kind"))
    if source is None and kind is None:
        return None

    reasons = [
        str(reason).strip()
        for reason in list(event.get("degradation_reasons") or event.get("reasons") or [])
        if str(reason).strip()
    ]
    return {
        "source": source,
        "kind": kind,
        "failure_type": _coerce_optional_text(event.get("failure_type")),
        "provider": _coerce_optional_text(event.get("provider")),
        "model": _coerce_optional_text(event.get("model")),
        "task_type": _coerce_optional_text(event.get("task_type")),
        "pre_degraded": bool(event.get("pre_degraded", False)),
        "post_degraded": bool(event.get("post_degraded", event.get("degraded", False))),
        "recovered": bool(event.get("recovered", False)),
        "adaptation_cost": _coerce_float(event.get("adaptation_cost")),
        "time_to_recover_seconds": _coerce_float(event.get("time_to_recover_seconds")),
        "iteration": _coerce_int(event.get("iteration")),
        "reasons": reasons,
    }


def extract_degradation_events(value: Any) -> list[dict[str, Any]]:
    """Extract normalized degradation events from task, trace, or payload objects."""
    events: list[dict[str, Any]] = []
    for item in _extract_sequence(value, "degradation_events"):
        normalized = _normalize_event(item)
        if normalized is not None:
            events.append(normalized)

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            for item in _extract_sequence(container, "degradation_events"):
                normalized = _normalize_event(item)
                if normalized is not None:
                    events.append(normalized)

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        events.extend(extract_degradation_events(trace))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.get("source"),
            event.get("kind"),
            event.get("failure_type"),
            event.get("provider"),
            event.get("model"),
            event.get("task_type"),
            event.get("pre_degraded"),
            event.get("post_degraded"),
            event.get("recovered"),
            event.get("adaptation_cost"),
            event.get("time_to_recover_seconds"),
            event.get("iteration"),
            tuple(event.get("reasons") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def summarize_degradation_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level degradation summary suitable for benchmark artifacts."""
    events = extract_degradation_events(value)
    if not events:
        return None

    source_counts = Counter(event["source"] for event in events if event.get("source"))
    kind_counts = Counter(event["kind"] for event in events if event.get("kind"))
    failure_types = Counter(event["failure_type"] for event in events if event.get("failure_type"))
    provider_counts = Counter(event["provider"] for event in events if event.get("provider"))
    reason_counts = Counter(
        reason for event in events for reason in list(event.get("reasons") or []) if reason
    )
    degraded_event_count = sum(
        1 for event in events if event.get("pre_degraded") or event.get("post_degraded")
    )
    recovered_event_count = sum(1 for event in events if event.get("recovered"))
    adaptation_costs = [
        float(event["adaptation_cost"])
        for event in events
        if event.get("adaptation_cost") is not None
    ]
    recovery_times = [
        float(event["time_to_recover_seconds"])
        for event in events
        if event.get("time_to_recover_seconds") is not None
    ]

    return {
        "event_count": len(events),
        "degraded_event_count": degraded_event_count,
        "recovered_event_count": recovered_event_count,
        "content_degradation_detected": any(
            event.get("kind") == "content_repetition" for event in events
        ),
        "provider_degradation_detected": any(
            event.get("source") == "provider_performance" for event in events
        ),
        "avg_adaptation_cost": round(
            sum(adaptation_costs) / max(1, len(adaptation_costs)),
            4,
        )
        if adaptation_costs
        else 0.0,
        "avg_time_to_recover_seconds": round(
            sum(recovery_times) / max(1, len(recovery_times)),
            4,
        )
        if recovery_times
        else 0.0,
        "sources": dict(source_counts),
        "kinds": dict(kind_counts),
        "failure_types": dict(failure_types),
        "providers": dict(provider_counts),
        "reasons": dict(reason_counts),
    }


def aggregate_degradation_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate degradation summaries across benchmark task results."""
    summaries = [summary for value in values if (summary := summarize_degradation_feedback(value))]
    if not summaries:
        return {
            "tasks_with_degradation_feedback": 0,
            "degradation_feedback_coverage": 0.0,
            "degradation_event_count": 0,
            "degraded_task_count": 0,
            "recovered_task_count": 0,
            "degradation_recovery_rate": 0.0,
            "avg_degradation_adaptation_cost": 0.0,
            "avg_degradation_time_to_recover_seconds": 0.0,
            "content_degradation_task_count": 0,
            "provider_degradation_task_count": 0,
            "degradation_sources": {},
            "degradation_kinds": {},
            "degradation_failure_types": {},
            "degradation_providers": {},
            "degradation_reasons": {},
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    source_counts = Counter()
    kind_counts = Counter()
    failure_types = Counter()
    providers = Counter()
    reasons = Counter()
    degraded_task_count = 0
    recovered_task_count = 0
    content_degradation_task_count = 0
    provider_degradation_task_count = 0

    for summary in summaries:
        source_counts.update(dict(summary.get("sources") or {}))
        kind_counts.update(dict(summary.get("kinds") or {}))
        failure_types.update(dict(summary.get("failure_types") or {}))
        providers.update(dict(summary.get("providers") or {}))
        reasons.update(dict(summary.get("reasons") or {}))
        if int(summary.get("degraded_event_count", 0) or 0) > 0:
            degraded_task_count += 1
        if int(summary.get("recovered_event_count", 0) or 0) > 0:
            recovered_task_count += 1
        if bool(summary.get("content_degradation_detected")):
            content_degradation_task_count += 1
        if bool(summary.get("provider_degradation_detected")):
            provider_degradation_task_count += 1

    return {
        "tasks_with_degradation_feedback": len(summaries),
        "degradation_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "degradation_event_count": sum(int(summary.get("event_count", 0) or 0) for summary in summaries),
        "degraded_task_count": degraded_task_count,
        "recovered_task_count": recovered_task_count,
        "degradation_recovery_rate": round(recovered_task_count / max(1, degraded_task_count), 4),
        "avg_degradation_adaptation_cost": round(
            sum(float(summary.get("avg_adaptation_cost", 0.0) or 0.0) for summary in summaries)
            / max(1, len(summaries)),
            4,
        ),
        "avg_degradation_time_to_recover_seconds": round(
            sum(
                float(summary.get("avg_time_to_recover_seconds", 0.0) or 0.0)
                for summary in summaries
            )
            / max(1, len(summaries)),
            4,
        ),
        "content_degradation_task_count": content_degradation_task_count,
        "provider_degradation_task_count": provider_degradation_task_count,
        "degradation_sources": dict(source_counts),
        "degradation_kinds": dict(kind_counts),
        "degradation_failure_types": dict(failure_types),
        "degradation_providers": dict(providers),
        "degradation_reasons": dict(reasons),
    }


__all__ = [
    "aggregate_degradation_feedback",
    "extract_degradation_events",
    "summarize_degradation_feedback",
]
