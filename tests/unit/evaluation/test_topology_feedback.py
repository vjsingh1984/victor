from victor.evaluation.topology_feedback import (
    aggregate_topology_feedback,
    extract_topology_events,
    summarize_topology_feedback,
)


def test_extract_topology_events_finds_nested_metadata_events():
    payload = {
        "metadata": {
            "topology_events": [
                {
                    "action": "team_plan",
                    "topology": "team",
                    "execution_mode": "team_execution",
                    "formation": "parallel",
                    "provider": "openai",
                    "confidence": 0.82,
                }
            ]
        }
    }

    events = extract_topology_events(payload)

    assert len(events) == 1
    assert events[0]["action"] == "team_plan"
    assert events[0]["topology"] == "team"


def test_summarize_topology_feedback_derives_reward_and_counts():
    payload = {
        "status": "passed",
        "completion_score": 0.9,
        "tool_calls": 6,
        "turns": 3,
        "metadata": {
            "topology_events": [
                {
                    "action": "parallel_exploration",
                    "topology": "parallel_exploration",
                    "execution_mode": "parallel_exploration",
                    "provider": "openai",
                    "confidence": 0.71,
                },
                {
                    "action": "team_plan",
                    "topology": "team",
                    "execution_mode": "team_execution",
                    "formation": "parallel",
                    "provider": "openai",
                    "confidence": 0.84,
                    "fallback_action": "escalate_model",
                },
            ]
        },
    }

    summary = summarize_topology_feedback(payload)

    assert summary is not None
    assert summary["event_count"] == 2
    assert summary["selected_action"] == "parallel_exploration"
    assert summary["final_action"] == "team_plan"
    assert summary["final_topology"] == "team"
    assert summary["fallback_count"] == 1
    assert summary["avg_confidence"] > 0.7
    assert 0.0 <= summary["topology_reward"] <= 1.0


def test_aggregate_topology_feedback_reports_coverage_and_action_mix():
    task_results = [
        {
            "status": "passed",
            "completion_score": 1.0,
            "metadata": {
                "topology_events": [
                    {
                        "action": "single_agent",
                        "topology": "single_agent",
                        "execution_mode": "single_agent",
                        "confidence": 0.78,
                    }
                ]
            },
        },
        {
            "status": "failed",
            "completion_score": 0.2,
            "metadata": {
                "topology_events": [
                    {
                        "action": "team_plan",
                        "topology": "team",
                        "execution_mode": "team_execution",
                        "formation": "hierarchical",
                        "confidence": 0.88,
                    }
                ]
            },
        },
        {"status": "passed", "completion_score": 1.0},
    ]

    metrics = aggregate_topology_feedback(task_results, total_tasks=3)

    assert metrics["tasks_with_topology_feedback"] == 2
    assert metrics["topology_feedback_coverage"] == 0.6667
    assert metrics["topology_actions"] == {"single_agent": 1, "team_plan": 1}
    assert metrics["topology_kinds"] == {"single_agent": 1, "team": 1}
    assert 0.0 <= metrics["avg_topology_reward"] <= 1.0
