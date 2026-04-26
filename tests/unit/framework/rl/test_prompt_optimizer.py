# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for GEPA-inspired prompt optimizer."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.config.prompt_optimization_settings import PromptOptimizationSettings
from victor.core.container import ServiceContainer, reset_container, set_container
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.credit_tracking_service import CreditTrackingService
from victor.framework.rl.experiment_coordinator import (
    ExperimentCoordinator,
    ExperimentStatus,
    VariantType,
)
from victor.framework.rl.pareto import ParetoEntry, ParetoFrontier
from victor.framework.rl.learners.prompt_optimizer import (
    ExecutionTrace,
    GEPAStrategy,
    PromptCandidate,
    PromptOptimizerLearner,
)


@pytest.fixture
def db():
    """In-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def learner(db):
    """Create a prompt optimizer learner."""
    return PromptOptimizerLearner(name="test_optimizer", db_connection=db)


class TestPromptCandidate:
    """Tests for PromptCandidate dataclass."""

    def test_initial_mean(self):
        c = PromptCandidate("test", "text", "hash", 1, "parent")
        assert c.mean == 0.5  # alpha=1, beta=1 → 0.5

    def test_update_success(self):
        c = PromptCandidate("test", "text", "hash", 1, "parent")
        c.update(True)
        assert c.alpha == 2.0
        assert c.beta_val == 1.0
        assert c.sample_count == 1
        assert c.mean > 0.5

    def test_update_failure(self):
        c = PromptCandidate("test", "text", "hash", 1, "parent")
        c.update(False)
        assert c.alpha == 1.0
        assert c.beta_val == 2.0
        assert c.mean < 0.5

    def test_sample_returns_float(self):
        c = PromptCandidate("test", "text", "hash", 1, "parent")
        sample = c.sample()
        assert 0.0 <= sample <= 1.0


class TestGEPAStrategyHeuristic:
    """Tests for GEPAStrategy heuristic paths (no Ollama required).

    LLM-dependent tests moved to tests/integration/framework/rl/test_gepa_integration.py
    """

    def test_mutate_adds_guidance_for_file_failures(self):
        """Heuristic mutation adds file verification guidance."""
        strategy = GEPAStrategy()
        strategy._provider = None  # Force heuristic path
        strategy._provider_name = None
        reflection = "file_not_found: 5 errors"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert "ls()" in result or "Verify file" in result

    def test_mutate_adds_guidance_for_edit_failures(self):
        """Heuristic mutation adds edit guidance."""
        strategy = GEPAStrategy()
        strategy._provider = None
        strategy._provider_name = None
        reflection = "edit mismatch: 3 errors"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert "old_str" in result or "editing" in result

    def test_mutate_no_change_when_no_failures(self):
        """Heuristic mutation makes no change when no failures detected."""
        strategy = GEPAStrategy()
        strategy._provider = None
        strategy._provider_name = None
        reflection = "Success rate: 10/10 (100%)"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert result == "Original."

    def test_reflect_includes_agent_credit_guidance(self):
        strategy = GEPAStrategy()
        strategy._provider = None
        strategy._provider_name = None
        traces = [
            ExecutionTrace(
                session_id="s1",
                task_type="action",
                provider="ollama",
                model="qwen",
                tool_calls=2,
                tool_failures={"edit_mismatch": 1},
                success=False,
                completion_score=0.4,
                tokens_used=200,
                agent_guidance=(
                    "Agent execution credit (from recent team runs):\n"
                    "- executor_1: low effectiveness."
                ),
            )
        ]

        reflection = strategy.reflect(traces, "GROUNDING_RULES", "Base prompt.")

        assert "Agent execution credit" in reflection
        assert "executor_1" in reflection


class TestPromptOptimizerLearner:
    """Tests for PromptOptimizerLearner."""

    def test_init_creates_tables(self, learner):
        cursor = learner.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "agent_prompt_candidate" in tables

    def test_evolvable_sections(self, learner):
        assert "ASI_TOOL_EFFECTIVENESS_GUIDANCE" in learner.EVOLVABLE_SECTIONS
        assert "GROUNDING_RULES" in learner.EVOLVABLE_SECTIONS
        assert "COMPLETION_GUIDANCE" in learner.EVOLVABLE_SECTIONS

    def test_get_recommendation_no_candidates(self, learner):
        rec = learner.get_recommendation(
            "ollama",
            "qwen",
            "action",
            section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
        )
        assert rec is None

    def test_save_and_load_candidate(self, db):
        learner1 = PromptOptimizerLearner(name="test1", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST_SECTION",
            provider="ollama",
            text="Evolved prompt text",
            text_hash="abc123",
            generation=1,
            parent_hash="parent",
        )
        key = learner1._candidate_key("TEST_SECTION", "ollama")
        learner1._candidates[key] = [candidate]
        learner1._save_candidate(candidate)

        # Load in new learner instance
        learner2 = PromptOptimizerLearner(name="test2", db_connection=db)
        assert key in learner2._candidates
        assert len(learner2._candidates[key]) == 1
        loaded = learner2._candidates[key][0]
        assert loaded.text == "Evolved prompt text"
        assert loaded.generation == 1
        assert loaded.provider == "ollama"

    def test_get_recommendation_with_candidate(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Better prompt",
            text_hash="hash1",
            generation=1,
            parent_hash="parent",
            alpha=5.0,
            beta_val=1.0,
            sample_count=10,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]

        rec = learner.get_recommendation("ollama", "qwen", "action", section_name="TEST")
        assert rec is not None
        assert rec.value == "Better prompt"
        assert rec.confidence > 0

    def test_get_recommendation_reason_uses_strategy_chain_when_present(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Layered prompt",
            text_hash="hash_chain",
            generation=2,
            parent_hash="parent",
            alpha=6.0,
            beta_val=1.0,
            sample_count=8,
            strategy_name="gepa",
        )
        candidate.strategy_chain = "gepa+cot_distillation"
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]

        rec = learner.get_recommendation("ollama", "qwen", "action", section_name="TEST")

        assert rec is not None
        assert "gepa+cot_distillation" in rec.reason

    def test_get_recommendation_falls_back_to_default(self, db):
        """Provider-specific miss should fall back to 'default' candidates."""
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="default",
            text="Default prompt",
            text_hash="hash2",
            generation=1,
            parent_hash="parent",
            sample_count=5,
        )
        key = learner._candidate_key("TEST", "default")
        learner._candidates[key] = [candidate]

        # Query with a provider that has no specific candidates
        rec = learner.get_recommendation("xai", "grok", "action", section_name="TEST")
        assert rec is not None
        assert rec.value == "Default prompt"

    def test_record_outcome_updates_candidate(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Prompt",
            text_hash="h",
            generation=1,
            parent_hash="p",
            sample_count=1,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]

        outcome = RLOutcome(
            provider="ollama",
            model="qwen",
            task_type="action",
            success=True,
            quality_score=0.8,
            metadata={"prompt_section": "TEST"},
        )
        learner.record_outcome(outcome)
        assert candidate.alpha > 1.0  # Updated

    def test_record_outcome_updates_pareto_instance_scores(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db, use_pareto=True)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Prompt",
            text_hash="paretohash",
            generation=1,
            parent_hash="p",
            is_active=True,
            sample_count=1,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]
        frontier = ParetoFrontier()
        frontier.add_candidate(candidate.text_hash, candidate.text, candidate.generation)
        learner._pareto_frontiers[key] = frontier

        outcome = RLOutcome(
            provider="ollama",
            model="qwen",
            task_type="action",
            success=True,
            quality_score=0.8,
            metadata={"prompt_section": "TEST", "task_id": "task-123"},
        )
        learner.record_outcome(outcome)

        frontier_entry = learner.get_pareto_frontier("TEST", "ollama").get_frontier()[0]
        assert frontier_entry.instance_scores["task-123::ollama"] == pytest.approx(0.82)
        assert candidate.instance_scores["task-123::ollama"] == pytest.approx(0.82)
        assert candidate.coverage_count == 1

        row = learner.db.execute(
            "SELECT best_candidate_hash, best_score, sample_count "
            "FROM agent_prompt_pareto_instance "
            "WHERE section_name = ? AND provider = ? AND instance_id = ?",
            ("TEST", "ollama", "task-123::ollama"),
        ).fetchone()
        assert row == ("paretohash", pytest.approx(0.82), 2)

    def test_evolve_records_full_strategy_chain_metadata(self, db):
        settings = SimpleNamespace(
            prompt_optimization=PromptOptimizationSettings(
                enabled=True,
                default_strategies=[],
                section_strategies={
                    "ASI_TOOL_EFFECTIVENESS_GUIDANCE": ["gepa", "cot_distillation"],
                },
            )
        )
        traces = [
            ExecutionTrace(
                session_id=f"s{i}",
                task_type="action",
                provider="anthropic" if i < 3 else "ollama",
                model="model-a",
                tool_calls=4,
                tool_failures={"edit_mismatch": 1} if i == 0 else {},
                success=True,
                completion_score=0.9 if i < 3 else 0.5,
                tokens_used=500,
            )
            for i in range(5)
        ]

        with patch("victor.config.settings.get_settings", return_value=settings):
            learner = PromptOptimizerLearner(name="test", db_connection=db)

        with (
            patch.object(learner, "_collect_learning_traces", return_value=traces),
            patch.object(learner, "_enrich_traces_with_credit"),
            patch.object(learner, "_apply_section_strategies", return_value="mutated prompt"),
        ):
            candidate = learner.evolve(
                "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
                "base prompt",
                provider="ollama",
            )

        assert candidate is not None
        assert getattr(candidate, "strategy_chain", None) == "gepa+cot_distillation"

    def test_init_pareto_frontiers_uses_section_provider_key(self, db):
        base = PromptOptimizerLearner(name="base", db_connection=db)
        candidate = PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="ollama",
            text="candidate",
            text_hash="pareto1",
            generation=1,
            parent_hash="parent",
            sample_count=3,
        )
        key = base._candidate_key("GROUNDING_RULES", "ollama")
        base._candidates[key] = [candidate]
        base._save_candidate(candidate)

        learner = PromptOptimizerLearner(name="pareto", db_connection=db, use_pareto=True)

        assert key in learner._pareto_frontiers

    def test_export_metrics(self, learner):
        metrics = learner.export_metrics()
        assert "total_candidates" in metrics
        assert metrics["total_candidates"] == 0

    def test_categorize_failure(self):
        assert (
            PromptOptimizerLearner._categorize_failure("File not found: foo.py") == "file_not_found"
        )
        assert (
            PromptOptimizerLearner._categorize_failure("old_str not found in bar.py")
            == "edit_mismatch"
        )
        assert (
            PromptOptimizerLearner._categorize_failure("Ambiguous match - found 2 times")
            == "edit_ambiguous"
        )
        assert PromptOptimizerLearner._categorize_failure("something else") == "other"

    def test_evolve_completion_guidance_section(self, db):
        """Verify COMPLETION_GUIDANCE can be evolved like other sections."""
        learner = PromptOptimizerLearner(name="test", db_connection=db)

        # Manually insert a candidate for COMPLETION_GUIDANCE
        candidate = PromptCandidate(
            section_name="COMPLETION_GUIDANCE",
            provider="default",
            text="Evolved completion guidance text",
            text_hash="comp_hash",
            generation=1,
            parent_hash="parent",
            alpha=5.0,
            beta_val=1.0,
            sample_count=10,
        )
        key = learner._candidate_key("COMPLETION_GUIDANCE", "default")
        learner._candidates[key] = [candidate]

        rec = learner.get_recommendation(
            "anthropic",
            "claude",
            "action",
            section_name="COMPLETION_GUIDANCE",
        )
        assert rec is not None
        assert rec.value == "Evolved completion guidance text"

    def test_candidate_key_format(self, learner):
        """Verify candidate key uses section::provider format."""
        key = learner._candidate_key("COMPLETION_GUIDANCE", "ollama")
        assert key == "COMPLETION_GUIDANCE::ollama"

    def test_save_and_load_candidate_persists_benchmark_state(self, db):
        learner1 = PromptOptimizerLearner(name="test1", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST_SECTION",
            provider="ollama",
            text="Evolved prompt text",
            text_hash="bench123",
            generation=2,
            parent_hash="parent",
            benchmark_score=0.92,
            benchmark_runs=3,
            benchmark_passed=True,
            is_active=True,
            strategy_name="prefpo",
            requires_benchmark=True,
        )
        key = learner1._candidate_key("TEST_SECTION", "ollama")
        learner1._candidates[key] = [candidate]
        learner1._save_candidate(candidate)

        learner2 = PromptOptimizerLearner(name="test2", db_connection=db)
        loaded = learner2._candidates[key][0]
        assert loaded.benchmark_score == pytest.approx(0.92)
        assert loaded.benchmark_runs == 3
        assert loaded.benchmark_passed is True
        assert loaded.is_active is True
        assert loaded.strategy_name == "prefpo"
        assert loaded.requires_benchmark is True

    def test_save_and_load_candidate_persists_frontier_metadata(self, db):
        learner1 = PromptOptimizerLearner(name="test1", db_connection=db, use_pareto=True)
        candidate = PromptCandidate(
            section_name="TEST_SECTION",
            provider="ollama",
            text="Evolved prompt text",
            text_hash="front123",
            generation=2,
            parent_hash="parent",
        )
        candidate.instance_scores = {"task-1::qwen": 1.0, "task-2::qwen": 0.0}
        candidate.coverage_count = 1
        candidate.is_on_frontier = False
        candidate.char_length = 18
        key = learner1._candidate_key("TEST_SECTION", "ollama")
        learner1._candidates[key] = [candidate]
        learner1._save_candidate(candidate)

        learner2 = PromptOptimizerLearner(name="test2", db_connection=db, use_pareto=True)
        loaded = learner2._candidates[key][0]
        assert loaded.instance_scores == {"task-1::qwen": 1.0, "task-2::qwen": 0.0}
        assert loaded.coverage_count == 2
        assert loaded.is_on_frontier is True
        assert loaded.char_length == len("Evolved prompt text")

    def test_record_benchmark_result_updates_running_average(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Prompt",
            text_hash="benchhash",
            generation=1,
            parent_hash="parent",
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]
        learner._save_candidate(candidate)

        learner.record_benchmark_result(
            section_name="TEST",
            provider="ollama",
            text_hash="benchhash",
            score=0.8,
            passed=False,
        )
        learner.record_benchmark_result(
            section_name="TEST",
            provider="ollama",
            text_hash="benchhash",
            score=1.0,
            passed=True,
        )

        updated = learner._candidates[key][0]
        assert updated.benchmark_runs == 2
        assert updated.benchmark_score == pytest.approx(0.9)
        assert updated.benchmark_passed is True

    def test_promote_candidate_marks_only_target_active(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        first = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Prompt A",
            text_hash="a1",
            generation=1,
            parent_hash="parent",
            benchmark_score=0.85,
            benchmark_runs=2,
            benchmark_passed=True,
            is_active=True,
        )
        second = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Prompt B",
            text_hash="b2",
            generation=2,
            parent_hash="a1",
            benchmark_score=0.91,
            benchmark_runs=3,
            benchmark_passed=True,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [first, second]
        learner._save_candidate(first)
        learner._save_candidate(second)

        learner.promote_candidate(section_name="TEST", provider="ollama", text_hash="b2")

        assert first.is_active is False
        assert second.is_active is True

    def test_rollback_active_candidate_reactivates_best_approved_candidate(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        stable = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Stable prompt",
            text_hash="stable",
            generation=1,
            parent_hash="parent",
            benchmark_score=0.88,
            benchmark_runs=4,
            benchmark_passed=True,
        )
        risky = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Risky prompt",
            text_hash="risky",
            generation=2,
            parent_hash="stable",
            benchmark_score=0.93,
            benchmark_runs=5,
            benchmark_passed=True,
            is_active=True,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [stable, risky]
        learner._save_candidate(stable)
        learner._save_candidate(risky)

        learner.rollback_active_candidate(
            section_name="TEST",
            provider="ollama",
            failed_text_hash="risky",
        )

        assert risky.is_active is False
        assert stable.is_active is True

    def test_get_recommendation_prefers_active_benchmark_passed_candidate(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        legacy = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Legacy best by samples",
            text_hash="legacy",
            generation=1,
            parent_hash="parent",
            alpha=10.0,
            beta_val=1.0,
            sample_count=20,
        )
        approved = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Approved active prompt",
            text_hash="approved",
            generation=2,
            parent_hash="legacy",
            benchmark_score=0.94,
            benchmark_runs=3,
            benchmark_passed=True,
            is_active=True,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [legacy, approved]

        rec = learner.get_recommendation("ollama", "qwen", "action", section_name="TEST")

        assert rec is not None
        assert rec.value == "Approved active prompt"

    def test_get_recommendation_skips_pending_benchmark_gated_candidate(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        pending_prefpo = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Pending PrefPO candidate",
            text_hash="pending",
            generation=2,
            parent_hash="baseline",
            alpha=20.0,
            beta_val=1.0,
            sample_count=25,
            strategy_name="prefpo",
            requires_benchmark=True,
        )
        fallback = PromptCandidate(
            section_name="TEST",
            provider="default",
            text="Approved default baseline",
            text_hash="baseline",
            generation=1,
            parent_hash="parent",
            benchmark_score=0.9,
            benchmark_runs=3,
            benchmark_passed=True,
            is_active=True,
        )
        learner._candidates[learner._candidate_key("TEST", "ollama")] = [pending_prefpo]
        learner._candidates[learner._candidate_key("TEST", "default")] = [fallback]

        rec = learner.get_recommendation("ollama", "qwen", "action", section_name="TEST")

        assert rec is not None
        assert rec.value == "Approved default baseline"

    def test_promote_candidate_requires_benchmark_when_candidate_is_gated(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        candidate = PromptCandidate(
            section_name="TEST",
            provider="ollama",
            text="Pending PrefPO candidate",
            text_hash="pending",
            generation=1,
            parent_hash="parent",
            strategy_name="prefpo",
            requires_benchmark=True,
        )
        key = learner._candidate_key("TEST", "ollama")
        learner._candidates[key] = [candidate]

        with pytest.raises(ValueError, match="benchmark gating"):
            learner.promote_candidate(section_name="TEST", provider="ollama", text_hash="pending")

    def test_section_strategies_use_builtin_defaults(self, db):
        settings = SimpleNamespace(prompt_optimization=PromptOptimizationSettings(enabled=True))

        with patch("victor.config.settings.get_settings", return_value=settings):
            learner = PromptOptimizerLearner(name="test", db_connection=db)

        few_shot_names = [
            type(s).__name__ for s in learner._strategies_for_section("FEW_SHOT_EXAMPLES")
        ]
        asi_names = [
            type(s).__name__
            for s in learner._strategies_for_section("ASI_TOOL_EFFECTIVENESS_GUIDANCE")
        ]
        grounding_names = [
            type(s).__name__ for s in learner._strategies_for_section("GROUNDING_RULES")
        ]

        assert few_shot_names == ["MIPROv2Strategy"]
        assert asi_names == ["GEPAStrategy", "CoTDistillationStrategy"]
        assert grounding_names == ["GEPAStrategy"]

    def test_section_strategies_honor_config_overrides(self, db):
        settings = SimpleNamespace(
            prompt_optimization=PromptOptimizationSettings(
                enabled=True,
                default_strategies=["cot_distillation"],
                section_strategies={
                    "GROUNDING_RULES": ["gepa", "cot_distillation"],
                    "FEW_SHOT_EXAMPLES": [],
                },
            )
        )

        with patch("victor.config.settings.get_settings", return_value=settings):
            learner = PromptOptimizerLearner(name="test", db_connection=db)

        grounding_names = [
            type(s).__name__ for s in learner._strategies_for_section("GROUNDING_RULES")
        ]
        completion_names = [
            type(s).__name__ for s in learner._strategies_for_section("COMPLETION_GUIDANCE")
        ]

        assert grounding_names == ["GEPAStrategy", "CoTDistillationStrategy"]
        assert learner._strategies_for_section("FEW_SHOT_EXAMPLES") == []
        assert completion_names == ["CoTDistillationStrategy"]

    def test_section_strategies_support_prefpo_override(self, db):
        settings = SimpleNamespace(
            prompt_optimization=PromptOptimizationSettings(
                enabled=True,
                default_strategies=["gepa"],
                section_strategies={
                    "GROUNDING_RULES": ["prefpo"],
                },
            )
        )

        with patch("victor.config.settings.get_settings", return_value=settings):
            learner = PromptOptimizerLearner(name="test", db_connection=db)

        grounding_names = [
            type(s).__name__ for s in learner._strategies_for_section("GROUNDING_RULES")
        ]

        assert grounding_names == ["PrefPOStrategy"]

    def test_evolve_with_prefpo_creates_non_active_candidate(self, db):
        settings = SimpleNamespace(
            prompt_optimization=PromptOptimizationSettings(
                enabled=True,
                default_strategies=[],
                section_strategies={"GROUNDING_RULES": ["prefpo"]},
            )
        )
        traces = [
            ExecutionTrace(
                session_id=f"s{i}",
                task_type="action",
                provider="ollama",
                model="qwen",
                tool_calls=4,
                tool_failures={"file_not_found": 2, "edit_mismatch": 1},
                success=False,
                completion_score=0.3,
                tokens_used=700,
            )
            for i in range(5)
        ]

        with patch("victor.config.settings.get_settings", return_value=settings):
            learner = PromptOptimizerLearner(name="test", db_connection=db)

        with patch.object(learner, "_collect_learning_traces", return_value=traces):
            with patch.object(learner, "_enrich_traces_with_credit") as enrich_mock:
                candidate = learner.evolve(
                    "GROUNDING_RULES",
                    "Base responses on tool output only.",
                )

        assert candidate is not None
        assert candidate.is_active is False
        assert candidate.benchmark_passed is False
        assert candidate.strategy_name == "prefpo"
        assert candidate.requires_benchmark is True
        assert "Verify file paths with ls()" in candidate.text
        enrich_mock.assert_called_once()

    def test_evolve_falls_back_to_pareto_merge_when_mutation_noops(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db, use_pareto=True)
        key = learner._candidate_key("TEST", "ollama")
        frontier = ParetoFrontier()
        frontier.add_candidate("h1", "Prompt A", 1, {"task-a": 1.0})
        frontier.add_candidate("h2", "Prompt B", 2, {"task-b": 1.0})
        learner._pareto_frontiers[key] = frontier

        traces = [
            ExecutionTrace(
                session_id=f"s{i}",
                task_type="action",
                provider="ollama",
                model="qwen",
                tool_calls=3,
                tool_failures={},
                success=True,
                completion_score=0.9,
                tokens_used=300,
            )
            for i in range(5)
        ]
        merged_entry = ParetoEntry(
            text_hash="merged123",
            text="Merged prompt",
            generation=3,
            instance_scores={"task-a": 0.8, "task-b": 0.8},
        )

        with (
            patch.object(learner, "_collect_learning_traces", return_value=traces),
            patch.object(learner, "_enrich_traces_with_credit"),
            patch.object(learner, "_apply_section_strategies", return_value="base prompt"),
            patch.object(frontier, "attempt_merge", return_value=merged_entry) as merge_mock,
        ):
            candidate = learner.evolve("TEST", "base prompt", provider="ollama")

        assert candidate is not None
        assert candidate.text == "Merged prompt"
        assert candidate.text_hash == "merged123"
        assert candidate.strategy_chain.endswith("+merge")
        merge_mock.assert_called_once()

    def test_seed_from_evaluations_updates_only_matching_candidate_hash(self, db, tmp_path):
        learner = PromptOptimizerLearner(name="test", db_connection=db, use_pareto=True)
        key = learner._candidate_key("TEST", "ollama")
        candidate_a = PromptCandidate("TEST", "Prompt A", "hasha", 1, "p", provider="ollama")
        candidate_b = PromptCandidate("TEST", "Prompt B", "hashb", 1, "p", provider="ollama")
        learner._candidates[key] = [candidate_a, candidate_b]
        frontier = ParetoFrontier()
        frontier.add_candidate("hasha", "Prompt A", 1)
        frontier.add_candidate("hashb", "Prompt B", 1)
        learner._pareto_frontiers[key] = frontier

        eval_payload = {
            "config": {
                "model": "qwen",
                "prompt_candidate_hash": "hasha",
                "section_name": "TEST",
                "provider": "ollama",
            },
            "tasks": [
                {"task_id": "task-1", "status": "passed"},
                {"task_id": "task-2", "status": "failed"},
            ],
        }
        eval_path = tmp_path / "eval_swe_bench_test.json"
        eval_path.write_text(json.dumps(eval_payload))

        updated = learner.seed_from_evaluations(tmp_path)

        assert updated == 2
        entry_a = next(e for e in frontier.get_frontier() if e.text_hash == "hasha")
        entry_b = next(e for e in frontier.get_frontier() if e.text_hash == "hashb")
        assert entry_a.instance_scores == {"task-1::qwen": 1.0, "task-2::qwen": 0.0}
        assert entry_b.instance_scores == {}

    def test_build_rollout_experiment_config_uses_existing_ab_primitives(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        control = PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="ollama",
            text="Approved baseline prompt",
            text_hash="control",
            generation=1,
            parent_hash="parent",
            benchmark_score=0.88,
            benchmark_runs=4,
            benchmark_passed=True,
            is_active=True,
            strategy_name="gepa",
        )
        treatment = PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="ollama",
            text="Approved PrefPO prompt",
            text_hash="treat",
            generation=2,
            parent_hash="control",
            benchmark_score=0.93,
            benchmark_runs=3,
            benchmark_passed=True,
            strategy_name="prefpo",
            requires_benchmark=True,
        )
        key = learner._candidate_key("GROUNDING_RULES", "ollama")
        learner._candidates[key] = [control, treatment]

        config = learner.build_rollout_experiment_config(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="treat",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

        assert config.experiment_id.startswith("prompt_optimizer_grounding_rules_ollama_treat")
        assert config.control.type == VariantType.CONTROL
        assert config.control.config["text_hash"] == "control"
        assert config.treatment.type == VariantType.TREATMENT
        assert config.treatment.config["text_hash"] == "treat"
        assert config.traffic_split == 0.2
        assert config.min_samples_per_variant == 25

    def test_create_rollout_experiment_registers_and_starts_ab_test(self, db):
        learner = PromptOptimizerLearner(name="test", db_connection=db)
        coordinator = ExperimentCoordinator()
        control = PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="ollama",
            text="Approved baseline prompt",
            text_hash="control",
            generation=1,
            parent_hash="parent",
            benchmark_score=0.88,
            benchmark_runs=4,
            benchmark_passed=True,
            is_active=True,
            strategy_name="gepa",
        )
        treatment = PromptCandidate(
            section_name="GROUNDING_RULES",
            provider="ollama",
            text="Approved PrefPO prompt",
            text_hash="treat",
            generation=2,
            parent_hash="control",
            benchmark_score=0.93,
            benchmark_runs=3,
            benchmark_passed=True,
            strategy_name="prefpo",
            requires_benchmark=True,
        )
        key = learner._candidate_key("GROUNDING_RULES", "ollama")
        learner._candidates[key] = [control, treatment]

        experiment_id = learner.create_rollout_experiment(
            coordinator,
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="treat",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

        assert experiment_id in coordinator._experiments
        assert coordinator._status[experiment_id] == ExperimentStatus.RUNNING

    def test_enrich_traces_with_credit_uses_registered_service(self, learner):
        """Recent credit summary should enrich traces when service is in DI."""
        service = CreditTrackingService()
        service.get_tool_credit_summary = lambda: {
            "grep": {"avg_credit": 0.8, "total_credit": 1.6, "call_count": 2}
        }
        service.generate_agent_guidance = lambda: (
            "Agent execution credit (from recent team runs):\n"
            "- researcher_1: high effectiveness."
        )

        container = ServiceContainer()
        container.register_instance(CreditTrackingService, service)
        set_container(container)
        try:
            trace = ExecutionTrace(
                session_id="s1",
                provider="ollama",
                model="qwen",
                task_type="search",
                tool_calls=1,
                tool_failures={},
                completion_score=0.9,
                success=True,
                tokens_used=120,
                tool_call_details=[SimpleNamespace(tool_name="grep")],
            )
            learner._enrich_traces_with_credit([trace])
            assert trace.credit_signals[0]["tool_name"] == "grep"
            assert trace.credit_signals[0]["credit"] == 0.8
            assert "researcher_1" in (trace.agent_guidance or "")
        finally:
            reset_container()

    def test_enrich_traces_with_credit_filters_by_trace_session_id(self, learner):
        service = CreditTrackingService()
        service.record_tool_result("grep", True, 50.0, session_id="session-target")
        service.assign_turn_credit(agent_id="researcher_1")
        service.record_tool_result(
            "edit", False, 100.0, error="Mismatch", session_id="session-other"
        )
        service.assign_turn_credit(agent_id="executor_1")

        container = ServiceContainer()
        container.register_instance(CreditTrackingService, service)
        set_container(container)
        try:
            trace = ExecutionTrace(
                session_id="session-target",
                provider="ollama",
                model="qwen",
                task_type="search",
                tool_calls=2,
                tool_failures={},
                completion_score=0.9,
                success=True,
                tokens_used=120,
                tool_call_details=[
                    SimpleNamespace(tool_name="grep"),
                    SimpleNamespace(tool_name="edit"),
                ],
            )
            learner._enrich_traces_with_credit([trace])
            tool_names = [signal["tool_name"] for signal in trace.credit_signals]
            assert tool_names == ["grep"]
        finally:
            reset_container()

    def test_cot_distillation_skips_when_target_provider_is_best(self):
        from victor.framework.rl.learners.strategies.cot_distillation_strategy import (
            CoTDistillationStrategy,
        )

        strategy = CoTDistillationStrategy(min_source_score=0.7, min_score_gap=0.15)
        traces = [
            ExecutionTrace(
                session_id="s1",
                task_type="action",
                provider="anthropic",
                model="claude",
                tool_calls=5,
                tool_failures={},
                success=True,
                completion_score=0.92,
                tokens_used=300,
            ),
            ExecutionTrace(
                session_id="s2",
                task_type="action",
                provider="ollama",
                model="qwen",
                tool_calls=5,
                tool_failures={},
                success=True,
                completion_score=0.55,
                tokens_used=300,
            ),
        ]

        reflection = strategy.reflect(
            traces,
            "COMPLETION_GUIDANCE",
            "base prompt",
            provider="anthropic",
        )

        assert reflection == ""
