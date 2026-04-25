# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for GEPA-inspired prompt optimizer."""

import sqlite3
from types import SimpleNamespace

import pytest

from victor.core.container import ServiceContainer, reset_container, set_container
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.credit_tracking_service import CreditTrackingService
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

    def test_enrich_traces_with_credit_uses_registered_service(self, learner):
        """Recent credit summary should enrich traces when service is in DI."""
        service = CreditTrackingService()
        service.get_tool_credit_summary = lambda: {
            "grep": {"avg_credit": 0.8, "total_credit": 1.6, "call_count": 2}
        }

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
        finally:
            reset_container()
