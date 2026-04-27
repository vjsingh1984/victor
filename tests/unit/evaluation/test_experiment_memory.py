from __future__ import annotations

import json

from victor.evaluation.experiment_analyzer import analyze_evaluation_result
from victor.evaluation.experiment_memory import (
    ExperimentInsight,
    ExperimentMemoryRecord,
    ExperimentMemoryStore,
    ExperimentScope,
)
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)


def _build_result() -> EvaluationResult:
    return EvaluationResult(
        config=EvaluationConfig(
            benchmark=BenchmarkType.GUIDE,
            model="gpt-5",
            provider="openai",
            prompt_candidate_hash="cand-123",
            prompt_section_name="GROUNDING_RULES",
            dataset_metadata={"source_name": "GUIDE", "version": "2026.04"},
        ),
        task_results=[
            TaskResult(
                task_id="guide-1",
                status=TaskStatus.FAILED,
                completion_score=0.3,
                failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
                failure_details={
                    "missing_actions": ["click"],
                    "optimization_summary": {
                        "feasible": False,
                        "reward": 0.24,
                        "reward_components": {"completion": 0.24},
                        "feasibility_failures": ["tests_pass", "task_complete"],
                    },
                },
                metadata={
                    "topology_events": [
                        {
                            "action": "team_plan",
                            "topology": "team",
                            "execution_mode": "team_execution",
                            "formation": "parallel",
                            "provider": "openai",
                            "selection_policy": "learned_close_override",
                            "fallback_action": "direct_answer",
                            "confidence": 0.81,
                        }
                    ]
                },
            ),
            TaskResult(
                task_id="guide-2",
                status=TaskStatus.PASSED,
                completion_score=0.94,
                failure_details={
                    "optimization_summary": {
                        "feasible": True,
                        "reward": 0.88,
                        "reward_components": {"completion": 0.88},
                        "feasibility_failures": [],
                    }
                },
                metadata={
                    "topology_events": [
                        {
                            "action": "direct_answer",
                            "topology": "single_agent",
                            "execution_mode": "single_agent",
                            "provider": "openai",
                            "selection_policy": "heuristic",
                            "confidence": 0.74,
                        }
                    ]
                },
            ),
        ],
    )


def test_analyze_evaluation_result_distills_structured_experiment_memory(tmp_path):
    result = _build_result()

    record = analyze_evaluation_result(
        result,
        source_result_path=tmp_path / "eval_guide_20260427_010101.json",
    )

    assert record.scope.benchmark == "guide"
    assert record.scope.provider == "openai"
    assert record.scope.model == "gpt-5"
    assert record.scope.prompt_candidate_hash == "cand-123"
    assert record.scope.section_name == "GROUNDING_RULES"
    assert record.summary_metrics["topology_learned_override_optimization_reward_delta"] < 0
    assert len(record.task_summaries) == 2
    assert record.task_summaries[0].optimization["feasible"] is False
    assert record.task_summaries[0].topology["final_selection_policy"] == "learned_close_override"

    insight_kinds = {insight.kind for insight in record.insights}
    assert "failed_hypothesis" in insight_kinds
    assert "environment_constraint" in insight_kinds
    assert "next_candidate" in insight_kinds
    assert "learned_close_override" in record.keywords
    assert "tests_pass" in record.keywords


def test_experiment_memory_store_persists_and_supports_structured_search(tmp_path):
    store = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    older = ExperimentMemoryRecord(
        record_id="older",
        created_at=10.0,
        scope=ExperimentScope(
            benchmark="guide",
            provider="openai",
            model="gpt-5",
            prompt_candidate_hash="cand-old",
            section_name="GROUNDING_RULES",
        ),
        summary_metrics={"pass_rate": 0.5},
        task_summaries=[],
        insights=[
            ExperimentInsight(
                kind="next_candidate",
                summary="Increase code intelligence coverage before adding more workers.",
                confidence=0.7,
            )
        ],
        keywords=["code_intelligence", "coverage"],
    )
    newer = ExperimentMemoryRecord(
        record_id="newer",
        created_at=20.0,
        scope=ExperimentScope(
            benchmark="guide",
            provider="openai",
            model="gpt-5",
            prompt_candidate_hash="cand-new",
            section_name="GROUNDING_RULES",
        ),
        summary_metrics={"pass_rate": 0.25},
        task_summaries=[],
        insights=[
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Learned close override underperformed heuristic routing on GUIDE.",
                confidence=0.92,
                evidence={"selection_policy": "learned_close_override"},
            )
        ],
        keywords=["learned_close_override", "heuristic", "guide"],
    )

    store.record(older)
    store.record(newer)

    reloaded = ExperimentMemoryStore(persist_path=tmp_path / "experiment_memory.json")
    matches = reloaded.search("heuristic learned override", provider="openai", limit=2)

    assert len(reloaded) == 2
    assert [match.record_id for match in matches] == ["newer"]
    assert reloaded.get_recent(limit=1)[0].record_id == "newer"
    payload = json.loads((tmp_path / "experiment_memory.json").read_text())
    assert payload["records"][0]["record_id"] == "older"
    assert payload["records"][1]["record_id"] == "newer"
