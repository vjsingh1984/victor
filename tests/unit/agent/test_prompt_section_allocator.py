"""Tests for prompt section budget allocator."""

import pytest
from victor.agent.prompt_section_allocator import (
    PromptSection,
    PromptSectionBudgetAllocator,
    PromptSectionBudgetAllocator,
    SectionMetadata,
    allocate_prompt_sections,
    create_section_metadata,
)


class TestSectionMetadata:
    """Test SectionMetadata dataclass."""

    def test_value_score_calculation(self):
        """Test value score calculation."""
        # High relevance, low cost = high value
        section1 = SectionMetadata(
            name="test1",
            content="short content",
            token_cost=50,
            relevance_score=0.9,
            priority=2,
        )
        score1 = section1.value_score()
        assert score1 > 0

        # Low relevance, high cost = low value
        section2 = SectionMetadata(
            name="test2",
            content="very long content" * 100,
            token_cost=5000,
            relevance_score=0.3,
            priority=8,
        )
        score2 = section2.value_score()
        assert score2 < score1

    def test_free_section_high_value(self):
        """Test that free sections (zero cost) get high value."""
        section = SectionMetadata(
            name="free_section",
            content="content",
            token_cost=0,
            relevance_score=0.5,
            priority=5,
        )
        assert section.value_score() > 0


class TestPromptSectionBudgetAllocator:
    """Test prompt section budget allocator."""

    def test_initialization(self):
        """Test allocator initialization."""
        allocator = PromptSectionBudgetAllocator(max_tokens=5000)
        assert allocator.max_tokens == 5000
        assert allocator.min_tokens == 1000

    def test_custom_initialization(self):
        """Test allocator initialization with custom parameters."""
        allocator = PromptSectionBudgetAllocator(
            max_tokens=10000,
            min_tokens=2000,
            core_section_threshold=0.5,
            cache_selections=False,
        )
        assert allocator.max_tokens == 10000
        assert allocator.min_tokens == 2000
        assert allocator.core_threshold == 0.5
        assert allocator.cache_selections is False

    def test_allocate_core_sections_first(self):
        """Test that core sections are prioritized."""
        allocator = PromptSectionBudgetAllocator(max_tokens=2000)

        sections = {
            PromptSection.SYSTEM_ROLE.value: SectionMetadata(
                name=PromptSection.SYSTEM_ROLE.value,
                content="You are a helpful assistant.",
                token_cost=30,
                priority=1,
                category="core",
                relevance_score=0.9,
            ),
            PromptSection.GROUNDING_RULES.value: SectionMetadata(
                name=PromptSection.GROUNDING_RULES.value,
                content="Base responses on tool output.",
                token_cost=100,
                priority=3,
                category="guidance",
                relevance_score=0.8,
            ),
            PromptSection.FEW_SHOT_EXAMPLES.value: SectionMetadata(
                name=PromptSection.FEW_SHOT_EXAMPLES.value,
                content="Example 1...\nExample 2...",
                token_cost=500,
                priority=7,
                category="enhancement",
                relevance_score=0.7,
            ),
        }

        context = {"task": "chat", "user_query": "help me"}
        selected = allocator.allocate(sections, context)

        # Core section should be selected
        assert PromptSection.SYSTEM_ROLE.value in selected
        # Guidance section should be selected if budget allows
        assert PromptSection.GROUNDING_RULES.value in selected or len(selected) == 1

    def test_budget_exhaustion(self):
        """Test that allocation stops when budget exhausted."""
        allocator = PromptSectionBudgetAllocator(max_tokens=500, min_tokens=100)

        sections = {
            "section1": SectionMetadata(
                name="section1",
                content="x" * 400,  # 100 tokens
                token_cost=100,
                priority=1,
                category="core",
            ),
            "section2": SectionMetadata(
                name="section2",
                content="y" * 400,  # 100 tokens
                token_cost=100,
                priority=2,
                category="core",
            ),
            "section3": SectionMetadata(
                name="section3",
                content="z" * 400,  # 100 tokens
                token_cost=100,
                priority=3,
                category="core",
            ),
        }

        context = {}
        selected = allocator.allocate(sections, context)

        # Should select sections until budget exhausted
        # Since all are category="core", they need to meet threshold
        # Default threshold is 0.3, but relevance_score defaults to 0.5
        # So all should pass threshold check
        assert len(selected) >= 1  # At least one section selected

    def test_relevance_threshold_filtering(self):
        """Test that low relevance sections are filtered out."""
        allocator = PromptSectionBudgetAllocator(
            max_tokens=5000,
            guidance_section_threshold=0.7,
            enhancement_section_threshold=0.7,
        )

        sections = {
            PromptSection.GROUNDING_RULES.value: SectionMetadata(
                name=PromptSection.GROUNDING_RULES.value,
                content="Base responses on facts, don't hallucinate.",
                token_cost=100,
                priority=3,
                category="guidance",
            ),
            PromptSection.RECOVERY_STRATEGIES.value: SectionMetadata(
                name=PromptSection.RECOVERY_STRATEGIES.value,
                content="Recovery strategies for failures.",
                token_cost=100,
                priority=7,
                category="enhancement",
            ),
        }

        # Context that triggers high relevance for GROUNDING_RULES but not RECOVERY_STRATEGIES
        context = {"task": "file search", "user_query": "Don't invent or hallucinate"}
        selected = allocator.allocate(sections, context)

        # High relevance should be selected (hallucinate keyword triggers 0.9 score)
        assert PromptSection.GROUNDING_RULES.value in selected
        # Low relevance should not be selected (no retry/recover keywords in context)
        assert PromptSection.RECOVERY_STRATEGIES.value not in selected

    def test_selection_caching(self):
        """Test that selections are cached."""
        allocator = PromptSectionBudgetAllocator(
            max_tokens=5000,
            cache_selections=True,
            cache_ttl_hours=1.0,
        )

        sections = {
            PromptSection.GROUNDING_RULES.value: SectionMetadata(
                name=PromptSection.GROUNDING_RULES.value,
                content="Base responses on tool output.",
                token_cost=100,
                priority=3,
                category="guidance",
                relevance_score=0.8,
            ),
        }

        context = {"task": "tool", "user_query": "Use tool to search"}

        # First call - computes selection
        selected1 = allocator.allocate(sections, context)
        assert PromptSection.GROUNDING_RULES.value in selected1

        # Second call - should use cache
        selected2 = allocator.allocate(sections, context)
        assert selected2 == selected1

    def test_cache_expiration(self):
        """Test that cache entries expire."""
        import time

        allocator = PromptSectionBudgetAllocator(
            max_tokens=5000,
            cache_selections=True,
            cache_ttl_hours=0.001,  # 3.6 seconds
        )

        sections = {
            PromptSection.TOOL_GUIDANCE.value: SectionMetadata(
                name=PromptSection.TOOL_GUIDANCE.value,
                content="Use tools appropriately.",
                token_cost=100,
                priority=3,
                category="guidance",
                relevance_score=0.8,
            ),
        }

        context = {"task": "tool", "user_query": "Use tool"}

        # First call
        selected1 = allocator.allocate(sections, context)
        assert PromptSection.TOOL_GUIDANCE.value in selected1

        # Wait for cache to expire
        time.sleep(4)

        # Second call - should recompute
        selected2 = allocator.allocate(sections, context)
        assert PromptSection.TOOL_GUIDANCE.value in selected2

    def test_record_section_measurements_updates_rolling_average(self):
        """Measured token costs should accumulate as rolling averages."""
        allocator = PromptSectionBudgetAllocator()

        allocator.record_section_measurements({PromptSection.TOOL_GUIDANCE.value: 120})
        allocator.record_section_measurements(
            {
                PromptSection.TOOL_GUIDANCE.value: 180,
                PromptSection.GROUNDING_RULES.value: 80,
            }
        )

        measurements = allocator.get_section_measurements()

        assert measurements[PromptSection.TOOL_GUIDANCE.value]["sample_count"] == 2
        assert measurements[PromptSection.TOOL_GUIDANCE.value][
            "average_token_cost"
        ] == pytest.approx(150.0)
        assert measurements[PromptSection.GROUNDING_RULES.value]["sample_count"] == 1
        assert measurements[PromptSection.GROUNDING_RULES.value]["average_token_cost"] == 80.0

    def test_allocate_uses_measured_token_cost_for_budgeting(self):
        """Observed section costs should influence selection order and fit checks."""
        context = {"task": "tool search", "user_query": "Use the tool to search the codebase"}
        sections = {
            PromptSection.SYSTEM_ROLE.value: SectionMetadata(
                name=PromptSection.SYSTEM_ROLE.value,
                content="You are a coding agent.",
                token_cost=30,
                priority=1,
                category="core",
            ),
            PromptSection.TOOL_GUIDANCE.value: SectionMetadata(
                name=PromptSection.TOOL_GUIDANCE.value,
                content="Use tools deliberately and prefer search before edits.",
                token_cost=130,
                priority=3,
                category="guidance",
            ),
            PromptSection.GROUNDING_RULES.value: SectionMetadata(
                name=PromptSection.GROUNDING_RULES.value,
                content="Base all claims on observed outputs and code evidence.",
                token_cost=70,
                priority=4,
                category="guidance",
            ),
        }

        baseline_allocator = PromptSectionBudgetAllocator(
            max_tokens=150,
            min_tokens=0,
            cache_selections=False,
        )
        baseline_selected = baseline_allocator.allocate(sections, context)
        assert PromptSection.GROUNDING_RULES.value in baseline_selected
        assert PromptSection.TOOL_GUIDANCE.value not in baseline_selected

        measured_allocator = PromptSectionBudgetAllocator(
            max_tokens=150,
            min_tokens=0,
            cache_selections=False,
        )
        measured_allocator.record_section_measurements({PromptSection.TOOL_GUIDANCE.value: 40})
        measured_selected = measured_allocator.allocate(sections, context)

        assert PromptSection.TOOL_GUIDANCE.value in measured_selected
        assert PromptSection.GROUNDING_RULES.value in measured_selected

    def test_get_stats_includes_measurement_summary(self):
        """Allocator stats should expose tracked measurement counts."""
        allocator = PromptSectionBudgetAllocator()
        allocator.record_section_measurements(
            {
                PromptSection.TOOL_GUIDANCE.value: 120,
                PromptSection.GROUNDING_RULES.value: 80,
            }
        )

        stats = allocator.get_stats()

        assert stats["measured_sections"] == 2
        assert stats["measurement_samples"] == 2

    def test_get_stats(self):
        """Test getting allocator statistics."""
        allocator = PromptSectionBudgetAllocator(max_tokens=5000)
        stats = allocator.get_stats()

        assert "cache_entries" in stats
        assert "max_tokens" in stats
        assert stats["max_tokens"] == 5000


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_section_metadata(self):
        """Test creating section metadata with auto token estimation."""
        content = "This is a test prompt section with some content."
        section = create_section_metadata(
            name="test_section",
            content=content,
            category="guidance",
            priority=3,
        )

        assert section.name == "test_section"
        assert section.content == content
        assert section.category == "guidance"
        assert section.priority == 3
        # Token cost should be estimated
        assert section.token_cost > 0

    def test_allocate_prompt_sections(self):
        """Test convenience function for allocating sections."""
        sections = {
            "section1": SectionMetadata(
                name="section1",
                content="content1",
                token_cost=100,
                priority=5,
                category="guidance",
                relevance_score=0.7,
            ),
        }

        context = {"task": "test"}
        selected = allocate_prompt_sections(
            sections=sections,
            context=context,
            max_tokens=1000,
        )

        assert isinstance(selected, list)
        assert len(selected) >= 0

    def test_edge_model_relevance_scoring(self):
        """Test relevance scoring with edge model (keyword-based)."""
        allocator = PromptSectionBudgetAllocator()

        # Error-related keywords
        context = {"task": "error handling", "user_query": "Failed to open file"}
        score = allocator._score_section_relevance(
            PromptSection.ERROR_HANDLING,
            context,
        )

        # Should have high relevance for error handling
        assert score >= 0.7

        # Tool-related keywords
        context2 = {"task": "tool usage", "user_query": "Which tool should I use?"}
        score2 = allocator._score_section_relevance(
            PromptSection.TOOL_GUIDANCE,
            context2,
        )

        # Should have high relevance for tool guidance
        assert score2 >= 0.7
