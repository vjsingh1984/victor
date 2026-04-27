from __future__ import annotations

from victor.evaluation.degradation_feedback import (
    aggregate_degradation_feedback,
    extract_degradation_events,
    summarize_degradation_feedback,
)


def test_extract_degradation_events_reads_task_metadata():
    task_payload = {
        "metadata": {
            "degradation_events": [
                {
                    "source": "provider_performance",
                    "kind": "provider_recovered",
                    "failure_type": "PROVIDER_ERROR",
                    "provider": "ollama",
                    "pre_degraded": True,
                    "post_degraded": False,
                    "recovered": True,
                    "adaptation_cost": 2,
                    "time_to_recover_seconds": 4.5,
                    "degradation_reasons": ["failure_streak"],
                }
            ]
        }
    }

    events = extract_degradation_events(task_payload)

    assert len(events) == 1
    assert events[0]["provider"] == "ollama"
    assert events[0]["recovered"] is True


def test_summarize_degradation_feedback_returns_task_level_summary():
    summary = summarize_degradation_feedback(
        {
            "metadata": {
                "degradation_events": [
                    {
                        "source": "provider_performance",
                        "kind": "persistent_provider_degradation",
                        "failure_type": "PROVIDER_ERROR",
                        "provider": "ollama",
                        "post_degraded": True,
                        "recovered": False,
                        "adaptation_cost": 3,
                        "degradation_reasons": ["failure_streak"],
                    },
                    {
                        "source": "agentic_loop",
                        "kind": "content_repetition",
                        "failure_type": "STUCK_LOOP",
                        "post_degraded": True,
                        "recovered": False,
                        "adaptation_cost": 3,
                        "degradation_reasons": ["content_repetition"],
                    },
                ]
            }
        }
    )

    assert summary is not None
    assert summary["event_count"] == 2
    assert summary["degraded_event_count"] == 2
    assert summary["provider_degradation_detected"] is True
    assert summary["content_degradation_detected"] is True
    assert summary["sources"] == {"provider_performance": 1, "agentic_loop": 1}


def test_aggregate_degradation_feedback_rolls_up_counts():
    metrics = aggregate_degradation_feedback(
        [
            {
                "metadata": {
                    "degradation_events": [
                        {
                            "source": "provider_performance",
                            "kind": "provider_recovered",
                            "failure_type": "PROVIDER_ERROR",
                            "provider": "ollama",
                            "pre_degraded": True,
                            "post_degraded": False,
                            "recovered": True,
                            "adaptation_cost": 2,
                            "time_to_recover_seconds": 4.0,
                            "degradation_reasons": ["failure_streak"],
                        }
                    ]
                }
            },
            {
                "metadata": {
                    "degradation_events": [
                        {
                            "source": "agentic_loop",
                            "kind": "content_repetition",
                            "failure_type": "STUCK_LOOP",
                            "post_degraded": True,
                            "recovered": False,
                            "adaptation_cost": 3,
                            "degradation_reasons": ["content_repetition"],
                        }
                    ]
                }
            },
        ],
        total_tasks=2,
    )

    assert metrics["tasks_with_degradation_feedback"] == 2
    assert metrics["degradation_feedback_coverage"] == 1.0
    assert metrics["degradation_event_count"] == 2
    assert metrics["degraded_task_count"] == 2
    assert metrics["recovered_task_count"] == 1
    assert metrics["degradation_recovery_rate"] == 0.5
    assert metrics["content_degradation_task_count"] == 1
    assert metrics["provider_degradation_task_count"] == 1
    assert metrics["degradation_sources"] == {"provider_performance": 1, "agentic_loop": 1}
