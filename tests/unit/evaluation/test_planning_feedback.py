from victor.evaluation.planning_feedback import (
    aggregate_planning_feedback,
    extract_planning_events,
    summarize_planning_feedback,
)
from victor.evaluation.protocol import TaskResult, TaskStatus


def test_extract_planning_events_finds_nested_metadata_events():
    payload = {
        "metadata": {
            "planning_events": [
                {
                    "selection_policy": "experiment_forced_slow_path",
                    "used_llm_planning": True,
                    "task_type": "action",
                    "force_reason": "experiment_constraints: tests_pass",
                    "constraint_tags": ["tests_pass"],
                    "experiment_support": 0.5,
                }
            ]
        }
    }

    events = extract_planning_events(payload)

    assert len(events) == 1
    assert events[0]["selection_policy"] == "experiment_forced_slow_path"
    assert events[0]["constraint_tags"] == ["tests_pass"]


def test_summarize_planning_feedback_derives_task_level_summary():
    result = TaskResult(
        task_id="task-1",
        status=TaskStatus.PASSED,
        completion_score=0.91,
        metadata={
            "planning_events": [
                {
                    "selection_policy": "experiment_forced_slow_path",
                    "used_llm_planning": True,
                    "force_reason": "experiment_constraints: tests_pass",
                    "constraint_tags": ["tests_pass"],
                    "experiment_support": 0.4,
                }
            ]
        },
    )

    summary = summarize_planning_feedback(result)

    assert summary is not None
    assert summary["final_policy"] == "experiment_forced_slow_path"
    assert summary["forced_by_runtime_feedback"] is False
    assert summary["completion_score"] == 0.91
    assert summary["constraint_tags"] == {"tests_pass": 1}


def test_aggregate_planning_feedback_exposes_policy_mix_and_delta():
    values = [
        TaskResult(
            task_id="task-fast",
            status=TaskStatus.FAILED,
            completion_score=0.25,
            metadata={
                "planning_events": [
                    {
                        "selection_policy": "heuristic_fast_path",
                        "used_llm_planning": False,
                        "task_type": "action",
                    }
                ]
            },
        ),
        TaskResult(
            task_id="task-forced",
            status=TaskStatus.PASSED,
            completion_score=0.9,
            metadata={
                "planning_events": [
                    {
                        "selection_policy": "experiment_forced_slow_path",
                        "used_llm_planning": True,
                        "task_type": "action",
                        "force_reason": "experiment_constraints: tests_pass",
                        "constraint_tags": ["tests_pass"],
                        "experiment_support": 0.33,
                    }
                ]
            },
        ),
    ]

    metrics = aggregate_planning_feedback(values, total_tasks=2)

    assert metrics["tasks_with_planning_feedback"] == 2
    assert metrics["planning_feedback_coverage"] == 1.0
    assert metrics["planning_policy_counts"] == {
        "heuristic_fast_path": 1,
        "experiment_forced_slow_path": 1,
    }
    assert metrics["planning_force_reasons"] == {"experiment_constraints: tests_pass": 1}
    assert metrics["planning_constraint_tags"] == {"tests_pass": 1}
    assert metrics["planning_used_llm_rate"] == 0.5
    assert metrics["planning_fast_path_rate"] == 0.5
    assert metrics["avg_completion_by_planning_policy"]["heuristic_fast_path"] == 0.25
    assert metrics["avg_completion_by_planning_policy"]["experiment_forced_slow_path"] == 0.9
    assert metrics["planning_forced_slow_path_completion_delta"] == 0.65
