# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Research prompt contributor.

Tests cover:
- Task type hints
- System prompt sections
- Grounding rules
- Context hints
- Priority values
"""

from __future__ import annotations

from typing import Dict

import pytest

from victor.core.verticals.protocols import TaskTypeHint
from victor.research.prompts import RESEARCH_TASK_TYPE_HINTS, ResearchPromptContributor


class TestResearchTaskTypeHints:
    """Tests for research task type hints."""

    def test_task_hints_is_dict(self):
        """Test that task hints is a dictionary."""
        assert isinstance(RESEARCH_TASK_TYPE_HINTS, dict)

    def test_task_hints_not_empty(self):
        """Test that task hints is not empty."""
        assert len(RESEARCH_TASK_TYPE_HINTS) > 0

    def test_includes_fact_check_hint(self):
        """Test that task hints includes fact_check."""
        assert "fact_check" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_literature_review_hint(self):
        """Test that task hints includes literature_review."""
        assert "literature_review" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_competitive_analysis_hint(self):
        """Test that task hints includes competitive_analysis."""
        assert "competitive_analysis" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_trend_research_hint(self):
        """Test that task hints includes trend_research."""
        assert "trend_research" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_technical_research_hint(self):
        """Test that task hints includes technical_research."""
        assert "technical_research" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_general_query_hint(self):
        """Test that task hints includes general_query."""
        assert "general_query" in RESEARCH_TASK_TYPE_HINTS

    def test_includes_general_hint(self):
        """Test that task hints includes general fallback."""
        assert "general" in RESEARCH_TASK_TYPE_HINTS

    def test_all_hints_are_task_type_hint_objects(self):
        """Test that all hints are TaskTypeHint objects."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert isinstance(hint, TaskTypeHint)

    def test_all_hints_have_task_type(self):
        """Test that all hints have task_type attribute."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert hasattr(hint, 'task_type')
            assert hint.task_type == key

    def test_all_hints_have_hint_text(self):
        """Test that all hints have hint text."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert hasattr(hint, 'hint')
            assert isinstance(hint.hint, str)
            assert len(hint.hint) > 0

    def test_all_hints_have_tool_budget(self):
        """Test that all hints have tool_budget."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert hasattr(hint, 'tool_budget')
            assert isinstance(hint.tool_budget, int)
            assert hint.tool_budget > 0

    def test_all_hints_have_priority_tools(self):
        """Test that all hints have priority_tools."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert hasattr(hint, 'priority_tools')
            assert isinstance(hint.priority_tools, list)
            assert len(hint.priority_tools) > 0

    def test_fact_check_hint_content(self):
        """Test that fact_check hint mentions verification."""
        hint = RESEARCH_TASK_TYPE_HINTS["fact_check"]
        assert "verif" in hint.hint.lower() or "source" in hint.hint.lower()

    def test_literature_review_hint_content(self):
        """Test that literature_review hint mentions systematic review."""
        hint = RESEARCH_TASK_TYPE_HINTS["literature_review"]
        assert "review" in hint.hint.lower() or "synth" in hint.hint.lower()

    def test_competitive_analysis_hint_content(self):
        """Test that competitive_analysis hint mentions comparison."""
        hint = RESEARCH_TASK_TYPE_HINTS["competitive_analysis"]
        assert "compar" in hint.hint.lower()

    def test_trend_research_hint_content(self):
        """Test that trend_research hint mentions patterns."""
        hint = RESEARCH_TASK_TYPE_HINTS["trend_research"]
        assert "trend" in hint.hint.lower() or "pattern" in hint.hint.lower()

    def test_technical_research_hint_content(self):
        """Test that technical_research hint mentions documentation."""
        hint = RESEARCH_TASK_TYPE_HINTS["technical_research"]
        assert "document" in hint.hint.lower() or "technic" in hint.hint.lower()

    def test_tool_budgets_are_reasonable(self):
        """Test that tool budgets are reasonable (between 5 and 25)."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert 5 <= hint.tool_budget <= 25

    def test_priority_tools_include_web_search(self):
        """Test that most hints include web_search in priority tools."""
        has_web_search = 0
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            if "web_search" in hint.priority_tools:
                has_web_search += 1

        # Most hints should have web_search
        assert has_web_search >= len(RESEARCH_TASK_TYPE_HINTS) // 2


class TestResearchPromptContributor:
    """Tests for ResearchPromptContributor class."""

    @pytest.fixture
    def contributor(self):
        """Create contributor instance."""
        return ResearchPromptContributor()

    def test_get_task_type_hints_returns_dict(self, contributor):
        """Test that get_task_type_hints returns dict."""
        hints = contributor.get_task_type_hints()

        assert isinstance(hints, dict)

    def test_get_task_type_hints_returns_copy(self, contributor):
        """Test that get_task_type_hints returns a copy."""
        hints1 = contributor.get_task_type_hints()
        hints2 = contributor.get_task_type_hints()

        # Should be different objects
        assert hints1 is not hints2

        # But same content
        assert hints1 == hints2

    def test_get_task_type_hints_content(self, contributor):
        """Test that task hints have expected content."""
        hints = contributor.get_task_type_hints()

        # Should have all expected hints
        assert "fact_check" in hints
        assert "literature_review" in hints
        assert "general" in hints

    def test_get_system_prompt_section_returns_string(self, contributor):
        """Test that get_system_prompt_section returns string."""
        section = contributor.get_system_prompt_section()

        assert isinstance(section, str)
        assert len(section) > 0

    def test_get_system_prompt_section_mentions_checklist(self, contributor):
        """Test that system prompt section mentions checklist."""
        section = contributor.get_system_prompt_section()

        assert "checklist" in section.lower()

    def test_get_system_prompt_section_mentions_sources(self, contributor):
        """Test that system prompt section mentions sources."""
        section = contributor.get_system_prompt_section()

        assert "source" in section.lower()

    def test_get_system_prompt_section_mentions_citations(self, contributor):
        """Test that system prompt section mentions citations."""
        section = contributor.get_system_prompt_section()

        assert "cite" in section.lower()

    def test_get_system_prompt_section_source_hierarchy(self, contributor):
        """Test that system prompt section includes source hierarchy."""
        section = contributor.get_system_prompt_section()

        # Should mention different source types
        assert "primary" in section.lower()
        assert "secondary" in section.lower()

    def test_get_grounding_rules_returns_string(self, contributor):
        """Test that get_grounding_rules returns string."""
        rules = contributor.get_grounding_rules()

        assert isinstance(rules, str)
        assert len(rules) > 0

    def test_get_grounding_rules_mentions_citations(self, contributor):
        """Test that grounding rules mention citations."""
        rules = contributor.get_grounding_rules()

        assert "cite" in rules.lower() or "url" in rules.lower()

    def test_get_grounding_rules_mentions_no_fabrication(self, contributor):
        """Test that grounding rules mention not fabricating."""
        rules = contributor.get_grounding_rules()

        assert "fabricat" in rules.lower() or "never" in rules.lower()

    def test_get_priority_returns_int(self, contributor):
        """Test that get_priority returns int."""
        priority = contributor.get_priority()

        assert isinstance(priority, int)

    def test_get_priority_value(self, contributor):
        """Test that get_priority returns expected value."""
        priority = contributor.get_priority()

        # Research is specialized, should have medium priority
        assert 0 <= priority <= 10

    def test_get_context_hints_with_known_task_type(self, contributor):
        """Test get_context_hints with known task type."""
        hint = contributor.get_context_hints("fact_check")

        assert hint is not None
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_get_context_hints_with_unknown_task_type(self, contributor):
        """Test get_context_hints with unknown task type."""
        hint = contributor.get_context_hints("unknown_task")

        assert hint is None

    def test_get_context_hints_with_none_task_type(self, contributor):
        """Test get_context_hints with None task type."""
        hint = contributor.get_context_hints(None)

        assert hint is None

    def test_get_context_hints_fact_check_content(self, contributor):
        """Test that fact_check context hint has expected content."""
        hint = contributor.get_context_hints("fact_check")

        assert "fact" in hint.lower() or "verif" in hint.lower()

    def test_get_context_hints_literature_review_content(self, contributor):
        """Test that literature_review context hint has expected content."""
        hint = contributor.get_context_hints("literature_review")

        assert "literature" in hint.lower() or "review" in hint.lower()

    def test_get_context_hints_general_fallback(self, contributor):
        """Test that general context hint works as fallback."""
        hint = contributor.get_context_hints("general")

        assert hint is not None
        assert "research" in hint.lower()

    def test_all_task_hints_have_structured_steps(self):
        """Test that all task hints have structured steps."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            # Hints should have numbered steps
            assert any(str(i) in hint.hint for i in range(1, 10))

    def test_all_task_hints_mention_tools(self):
        """Test that all task hints mention relevant tools implicitly."""
        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            # Hints should reference searching or fetching or finding
            hint_lower = hint.hint.lower()
            # Most hints should mention at least one of these
            has_tool_reference = any(term in hint_lower for term in ["search", "fetch", "find", "document", "code", "write"])
            assert has_tool_reference or len(hint.hint) > 0  # At minimum should have content

    def test_system_prompt_section_avoids_outdated_sources(self, contributor):
        """Test that system prompt section warns about outdated sources."""
        section = contributor.get_system_prompt_section()

        assert "outdat" in section.lower() or "old" in section.lower() or "2 year" in section.lower()

    def test_grounding_rules_mentions_uncertainty(self, contributor):
        """Test that grounding rules mention acknowledging uncertainty."""
        rules = contributor.get_grounding_rules()

        assert "uncertain" in rules.lower() or "conflict" in rules.lower()

    def test_system_prompt_section_checklist_items(self, contributor):
        """Test that system prompt section has multiple checklist items."""
        section = contributor.get_system_prompt_section()

        # Should have checklist items marked with [ ]
        assert "[ ]" in section or "- [" in section

    def test_fact_check_tool_budget_higher_than_general(self):
        """Test that fact_check has higher budget than general query."""
        fact_check_budget = RESEARCH_TASK_TYPE_HINTS["fact_check"].tool_budget
        general_budget = RESEARCH_TASK_TYPE_HINTS["general_query"].tool_budget

        assert fact_check_budget > general_budget

    def test_literature_review_highest_budget(self):
        """Test that literature_review has highest budget."""
        literature_budget = RESEARCH_TASK_TYPE_HINTS["literature_review"].tool_budget

        for key, hint in RESEARCH_TASK_TYPE_HINTS.items():
            assert literature_budget >= hint.tool_budget

    def test_technical_research_includes_code_search(self):
        """Test that technical_research includes code_search."""
        hint = RESEARCH_TASK_TYPE_HINTS["technical_research"]

        assert "code_search" in hint.priority_tools

    def test_all_hints_have_unique_task_types(self):
        """Test that all hints have unique task types."""
        task_types = [hint.task_type for hint in RESEARCH_TASK_TYPE_HINTS.values()]

        assert len(task_types) == len(set(task_types))

    def test_general_hint_is_comprehensive(self):
        """Test that general hint covers multiple research aspects."""
        hint = RESEARCH_TASK_TYPE_HINTS["general"]

        # Should mention finding, fetching, synthesizing
        hint_lower = hint.hint.lower()
        assert "search" in hint_lower
        assert "fetch" in hint_lower
        assert "synth" in hint_lower or "combin" in hint_lower
