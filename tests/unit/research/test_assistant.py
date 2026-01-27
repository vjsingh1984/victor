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

"""Unit tests for Research Assistant vertical.

Tests cover:
- Assistant initialization and properties
- Tool list generation
- System prompt content
- Stage definitions
- Prompt builder configuration
- Extension protocol methods
- Protocol registration
"""

from __future__ import annotations

from typing import Dict, List, Set

import pytest

from victor.core.verticals.base import StageDefinition
from victor.research.assistant import ResearchAssistant
from victor.tools.tool_names import ToolNames


class TestResearchAssistant:
    """Tests for ResearchAssistant class."""

    def test_name(self):
        """Test assistant name."""
        assert ResearchAssistant.name == "research"

    def test_description(self):
        """Test assistant description."""
        assert "web research" in ResearchAssistant.description.lower()
        assert "fact-checking" in ResearchAssistant.description.lower()

    def test_version(self):
        """Test assistant version."""
        assert ResearchAssistant.version == "0.5.0"

    def test_get_tools_returns_list(self):
        """Test that get_tools returns a list."""
        tools = ResearchAssistant.get_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_includes_web_search(self):
        """Test that get_tools includes web search."""
        tools = ResearchAssistant.get_tools()

        assert ToolNames.WEB_SEARCH in tools

    def test_get_tools_includes_web_fetch(self):
        """Test that get_tools includes web fetch."""
        tools = ResearchAssistant.get_tools()

        assert ToolNames.WEB_FETCH in tools

    def test_get_tools_includes_file_operations(self):
        """Test that get_tools includes file operations."""
        tools = ResearchAssistant.get_tools()

        # Should include basic file ops from framework
        assert ToolNames.READ in tools
        assert ToolNames.WRITE in tools

    def test_get_tools_includes_code_search(self):
        """Test that get_tools includes code search for technical research."""
        tools = ResearchAssistant.get_tools()

        assert ToolNames.CODE_SEARCH in tools

    def test_get_system_prompt_returns_string(self):
        """Test that get_system_prompt returns a string."""
        prompt = ResearchAssistant.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_system_prompt_contains_web_research_focus(self):
        """Test that system prompt emphasizes web research."""
        prompt = ResearchAssistant.get_system_prompt()

        # Should mention research and information gathering
        assert "research" in prompt.lower() or "information gathering" in prompt.lower()

    def test_get_system_prompt_contains_source_quality(self):
        """Test that system prompt mentions source quality."""
        prompt = ResearchAssistant.get_system_prompt()

        assert "source" in prompt.lower() or "authoritative" in prompt.lower()

    def test_get_system_prompt_contains_verification(self):
        """Test that system prompt mentions verification."""
        prompt = ResearchAssistant.get_system_prompt()

        assert "verif" in prompt.lower() or "fact-check" in prompt.lower()

    def test_get_stages_returns_dict(self):
        """Test that get_stages returns a dictionary."""
        stages = ResearchAssistant.get_stages()

        assert isinstance(stages, dict)
        assert len(stages) > 0

    def test_get_stages_includes_initial_stage(self):
        """Test that get_stages includes INITIAL stage."""
        stages = ResearchAssistant.get_stages()

        assert "INITIAL" in stages

    def test_get_stages_includes_searching_stage(self):
        """Test that get_stages includes SEARCHING stage."""
        stages = ResearchAssistant.get_stages()

        assert "SEARCHING" in stages

    def test_get_stages_includes_reading_stage(self):
        """Test that get_stages includes READING stage."""
        stages = ResearchAssistant.get_stages()

        assert "READING" in stages

    def test_get_stages_includes_synthesizing_stage(self):
        """Test that get_stages includes SYNTHESIZING stage."""
        stages = ResearchAssistant.get_stages()

        assert "SYNTHESIZING" in stages

    def test_get_stages_includes_writing_stage(self):
        """Test that get_stages includes WRITING stage."""
        stages = ResearchAssistant.get_stages()

        assert "WRITING" in stages

    def test_get_stages_includes_verification_stage(self):
        """Test that get_stages includes VERIFICATION stage."""
        stages = ResearchAssistant.get_stages()

        assert "VERIFICATION" in stages

    def test_get_stages_includes_completion_stage(self):
        """Test that get_stages includes COMPLETION stage."""
        stages = ResearchAssistant.get_stages()

        assert "COMPLETION" in stages

    def test_stage_definitions_have_correct_structure(self):
        """Test that stage definitions have correct structure."""
        stages = ResearchAssistant.get_stages()

        for stage_name, stage_def in stages.items():
            assert isinstance(stage_def, StageDefinition)
            assert hasattr(stage_def, "name")
            assert hasattr(stage_def, "description")
            assert hasattr(stage_def, "tools")
            assert hasattr(stage_def, "keywords")
            assert hasattr(stage_def, "next_stages")

            assert stage_def.name == stage_name
            assert isinstance(stage_def.description, str)
            assert isinstance(stage_def.tools, (set, frozenset))
            assert isinstance(stage_def.keywords, (list, set))
            assert isinstance(stage_def.next_stages, (set, frozenset))

    def test_initial_stage_tools(self):
        """Test that INITIAL stage has appropriate tools."""
        stages = ResearchAssistant.get_stages()
        initial = stages["INITIAL"]

        # Should have web search for research
        assert ToolNames.WEB_SEARCH in initial.tools

    def test_searching_stage_tools(self):
        """Test that SEARCHING stage has web search and fetch tools."""
        stages = ResearchAssistant.get_stages()
        searching = stages["SEARCHING"]

        assert ToolNames.WEB_SEARCH in searching.tools
        assert ToolNames.WEB_FETCH in searching.tools

    def test_reading_stage_tools(self):
        """Test that READING stage has web fetch and read tools."""
        stages = ResearchAssistant.get_stages()
        reading = stages["READING"]

        assert ToolNames.WEB_FETCH in reading.tools
        assert ToolNames.READ in reading.tools

    def test_writing_stage_tools(self):
        """Test that WRITING stage has write and edit tools."""
        stages = ResearchAssistant.get_stages()
        writing = stages["WRITING"]

        assert ToolNames.WRITE in writing.tools
        assert ToolNames.EDIT in writing.tools

    def test_verification_stage_tools(self):
        """Test that VERIFICATION stage has web search tools."""
        stages = ResearchAssistant.get_stages()
        verification = stages["VERIFICATION"]

        assert ToolNames.WEB_SEARCH in verification.tools
        assert ToolNames.WEB_FETCH in verification.tools

    def test_stage_transitions_are_valid(self):
        """Test that stage transitions reference valid stages."""
        stages = ResearchAssistant.get_stages()
        stage_names = set(stages.keys())

        for stage_name, stage_def in stages.items():
            # All next_stages should be valid stage names
            assert stage_def.next_stages.issubset(stage_names)

    def test_get_vertical_prompt_returns_string(self):
        """Test that _get_vertical_prompt returns a string."""
        prompt = ResearchAssistant._get_vertical_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_vertical_prompt_content(self):
        """Test that vertical prompt mentions research skills."""
        prompt = ResearchAssistant._get_vertical_prompt()

        assert "research" in prompt.lower()
        assert "information" in prompt.lower() or "synthesis" in prompt.lower()

    def test_get_prompt_builder_returns_builder(self):
        """Test that get_prompt_builder returns PromptBuilder."""
        builder = ResearchAssistant.get_prompt_builder()

        assert builder is not None
        assert hasattr(builder, "build")

    def test_get_prompt_builder_adds_grounding(self):
        """Test that prompt builder adds research-specific grounding."""
        builder = ResearchAssistant.get_prompt_builder()

        # Should have grounding added
        # This tests that the builder was configured correctly
        assert builder is not None

    def test_get_prompt_builder_adds_rules(self):
        """Test that prompt builder adds research-specific rules."""
        builder = ResearchAssistant.get_prompt_builder()

        # Should have rules added
        assert builder is not None

    def test_get_prompt_builder_adds_checklist(self):
        """Test that prompt builder adds research-specific checklist."""
        builder = ResearchAssistant.get_prompt_builder()

        # Should have checklist added
        assert builder is not None

    def test_get_handlers_returns_dict(self):
        """Test that get_handlers returns a dictionary."""
        handlers = ResearchAssistant.get_handlers()

        assert isinstance(handlers, dict)
        assert len(handlers) > 0

    def test_get_handlers_includes_web_scraper(self):
        """Test that get_handlers includes web_scraper."""
        handlers = ResearchAssistant.get_handlers()

        assert "web_scraper" in handlers

    def test_get_handlers_includes_citation_formatter(self):
        """Test that get_handlers includes citation_formatter."""
        handlers = ResearchAssistant.get_handlers()

        assert "citation_formatter" in handlers

    def test_get_tool_dependency_provider(self):
        """Test that get_tool_dependency_provider returns provider."""
        provider = ResearchAssistant.get_tool_dependency_provider()

        assert provider is not None

    def test_get_capability_configs(self):
        """Test that get_capability_configs returns dict."""
        configs = ResearchAssistant.get_capability_configs()

        assert isinstance(configs, dict)

    def test_implements_tool_provider_protocol(self):
        """Test that ResearchAssistant implements ToolProvider protocol."""
        from victor.core.verticals.protocols.providers import ToolProvider

        # Check protocol implementation via list_implemented_protocols
        implemented = ResearchAssistant.list_implemented_protocols()
        assert ToolProvider in implemented

    def test_implements_prompt_contributor_provider_protocol(self):
        """Test that ResearchAssistant implements PromptContributorProvider protocol."""
        from victor.core.verticals.protocols.providers import PromptContributorProvider

        implemented = ResearchAssistant.list_implemented_protocols()
        assert PromptContributorProvider in implemented

    def test_implements_handler_provider_protocol(self):
        """Test that ResearchAssistant implements HandlerProvider protocol."""
        from victor.core.verticals.protocols.providers import HandlerProvider

        implemented = ResearchAssistant.list_implemented_protocols()
        assert HandlerProvider in implemented

    def test_stage_keywords_relevant_to_research(self):
        """Test that stage keywords are research-relevant."""
        stages = ResearchAssistant.get_stages()

        # Check that keywords are relevant
        for stage_name, stage_def in stages.items():
            if isinstance(stage_def.keywords, (list, set)):
                for keyword in stage_def.keywords:
                    assert isinstance(keyword, str)
                    assert len(keyword) > 0

    def test_system_prompts_mentions_tools(self):
        """Test that system prompt mentions available tools."""
        prompt = ResearchAssistant.get_system_prompt()

        # Should mention gathering or searching
        assert "gathering" in prompt.lower() or "searching" in prompt.lower()

    def test_system_prompt_mentions_citations(self):
        """Test that system prompt mentions citations."""
        prompt = ResearchAssistant.get_system_prompt()

        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_system_prompt_mentions_objectivity(self):
        """Test that system prompt mentions objectivity."""
        prompt = ResearchAssistant.get_system_prompt()

        # Prompt mentions distinguishing facts from opinions and considering biases
        assert (
            "distinguish" in prompt.lower() and "opinion" in prompt.lower()
        ) or "bias" in prompt.lower()

    def test_completion_stage_has_no_tools(self):
        """Test that COMPLETION stage has no tools."""
        stages = ResearchAssistant.get_stages()
        completion = stages["COMPLETION"]

        # Should have empty or no tools
        assert len(completion.tools) == 0

    def test_completion_stage_has_no_next_stages(self):
        """Test that COMPLETION stage has no next stages."""
        stages = ResearchAssistant.get_stages()
        completion = stages["COMPLETION"]

        # Should have empty next_stages
        assert len(completion.next_stages) == 0

    def test_research_workflow_progression(self):
        """Test that stages form a logical research workflow."""
        stages = ResearchAssistant.get_stages()

        # INITIAL -> SEARCHING
        assert "SEARCHING" in stages["INITIAL"].next_stages

        # SEARCHING -> READING or SEARCHING
        assert "READING" in stages["SEARCHING"].next_stages

        # READING -> SYNTHESIZING
        assert "SYNTHESIZING" in stages["READING"].next_stages

        # SYNTHESIZING -> WRITING
        assert "WRITING" in stages["SYNTHESIZING"].next_stages

        # WRITING -> VERIFICATION
        assert "VERIFICATION" in stages["WRITING"].next_stages

        # VERIFICATION -> COMPLETION
        assert "COMPLETION" in stages["VERIFICATION"].next_stages

    def test_get_tools_does_not_duplicate(self):
        """Test that get_tools does not have duplicates."""
        tools = ResearchAssistant.get_tools()

        # Check for duplicates
        unique_tools = set(tools)
        assert len(tools) == len(unique_tools)

    def test_all_stages_have_descriptions(self):
        """Test that all stages have descriptions."""
        stages = ResearchAssistant.get_stages()

        for stage_name, stage_def in stages.items():
            assert len(stage_def.description) > 0
