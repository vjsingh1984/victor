# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for GEPA-inspired prompt optimizer."""

import sqlite3

import pytest

from victor.framework.rl.base import RLOutcome
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


class TestGEPAStrategy:
    """Tests for GEPAStrategy."""

    def test_reflect_produces_analysis(self):
        strategy = GEPAStrategy()
        traces = [
            ExecutionTrace("s1", "action", "ollama", "qwen", 5,
                           {"file_not_found": 3}, False, 0.3, 1000),
            ExecutionTrace("s2", "action", "ollama", "qwen", 3,
                           {}, True, 0.9, 800),
        ]
        reflection = strategy.reflect(traces, "TEST_SECTION", "current text")
        assert "2 execution traces" in reflection
        assert "file_not_found" in reflection

    def test_reflect_empty_traces(self):
        strategy = GEPAStrategy()
        reflection = strategy.reflect([], "TEST_SECTION", "current text")
        assert "0 execution traces" in reflection

    def test_mutate_adds_guidance_for_file_failures(self):
        strategy = GEPAStrategy()
        reflection = "file_not_found: 5 errors"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert "ls()" in result or "Verify file" in result

    def test_mutate_adds_guidance_for_edit_failures(self):
        strategy = GEPAStrategy()
        reflection = "edit mismatch: 3 errors"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert "old_str" in result or "editing" in result

    def test_mutate_no_change_when_no_failures(self):
        strategy = GEPAStrategy()
        reflection = "Success rate: 10/10 (100%)"
        result = strategy.mutate("Original.", reflection, "TEST")
        assert result == "Original."


class TestPromptOptimizerLearner:
    """Tests for PromptOptimizerLearner."""

    def test_init_creates_tables(self, learner):
        cursor = learner.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "agent_prompt_candidate" in tables

    def test_evolvable_sections(self, learner):
        assert "ASI_TOOL_EFFECTIVENESS_GUIDANCE" in learner.EVOLVABLE_SECTIONS
        assert "GROUNDING_RULES" in learner.EVOLVABLE_SECTIONS

    def test_get_recommendation_no_candidates(self, learner):
        rec = learner.get_recommendation(
            "ollama", "qwen", "action",
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

        rec = learner.get_recommendation(
            "ollama", "qwen", "action", section_name="TEST"
        )
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
        rec = learner.get_recommendation(
            "xai", "grok", "action", section_name="TEST"
        )
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
        assert PromptOptimizerLearner._categorize_failure(
            "File not found: foo.py"
        ) == "file_not_found"
        assert PromptOptimizerLearner._categorize_failure(
            "old_str not found in bar.py"
        ) == "edit_mismatch"
        assert PromptOptimizerLearner._categorize_failure(
            "Ambiguous match - found 2 times"
        ) == "edit_ambiguous"
        assert PromptOptimizerLearner._categorize_failure(
            "something else"
        ) == "other"
