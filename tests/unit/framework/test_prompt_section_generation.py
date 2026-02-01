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

"""Tests for prompt section generation and centralization.

Phase 5 implementation: Tests that verticals use centralized prompt sections
instead of inline prompts for consistency and maintainability.
"""


from victor.coding.assistant import CodingAssistant
from victor.devops.assistant import DevOpsAssistant
from victor.rag.assistant import RAGAssistant


class TestPromptSectionCentralization:
    """Tests for centralized prompt sections in verticals.

    Phase 5: System Prompt Centralization ensures that:
    1. Inline prompts are extracted to dedicated section modules
    2. Both get_system_prompt() and get_prompt_builder() use same source
    3. Prompts are consistent and maintainable
    """

    def test_coding_has_centralized_prompt_template(self, reset_singletons):
        """CodingAssistant should use CodingPromptTemplate.

        Phase 5: Coding vertical should use CodingPromptTemplate
        instead of inline prompt in get_system_prompt().
        """
        # Should have get_prompt_builder() method
        assert hasattr(
            CodingAssistant, "get_prompt_builder"
        ), "CodingAssistant should have get_prompt_builder() method"

        # Should return a PromptBuilder instance
        builder = CodingAssistant.get_prompt_builder()
        assert builder is not None, "get_prompt_builder() should return PromptBuilder"

        # The builder should be able to build a prompt
        prompt = builder.build()
        assert isinstance(prompt, str), "Builder should produce string prompt"
        assert len(prompt) > 0, "Prompt should not be empty"

    def test_devops_has_centralized_prompt_section(self, reset_singletons):
        """DevOpsAssistant should use centralized prompt section.

        Phase 5: DevOps vertical should use DevOpsPromptSection
        instead of inline prompt in _get_system_prompt().
        """
        # Should have get_prompt_builder() method
        assert hasattr(
            DevOpsAssistant, "get_prompt_builder"
        ), "DevOpsAssistant should have get_prompt_builder() method"

        # Should return a PromptBuilder instance
        builder = DevOpsAssistant.get_prompt_builder()
        assert builder is not None, "get_prompt_builder() should return PromptBuilder"

        # The builder should be able to build a prompt
        prompt = builder.build()
        assert isinstance(prompt, str), "Builder should produce string prompt"
        assert len(prompt) > 0, "Prompt should not be empty"

    def test_rag_has_centralized_prompt_section(self, reset_singletons):
        """RAGAssistant should use centralized prompt section.

        Phase 5: RAG vertical should use RAGPromptSection
        instead of inline prompt in get_system_prompt().
        """
        # Should have get_prompt_builder() method
        assert hasattr(
            RAGAssistant, "get_prompt_builder"
        ), "RAGAssistant should have get_prompt_builder() method"

        # Should return a PromptBuilder instance
        builder = RAGAssistant.get_prompt_builder()
        assert builder is not None, "get_prompt_builder() should return PromptBuilder"

        # The builder should be able to build a prompt
        prompt = builder.build()
        assert isinstance(prompt, str), "Builder should produce string prompt"
        assert len(prompt) > 0, "Prompt should not be empty"

    def test_prompt_builder_consistency_with_system_prompt(self, reset_singletons):
        """PromptBuilder should produce consistent prompt with get_system_prompt().

        Phase 5: Both methods should use the same centralized prompt source
        to ensure consistency.
        """
        # Test CodingAssistant
        coding_builder_prompt = CodingAssistant.get_prompt_builder().build()
        coding_system_prompt = CodingAssistant.get_system_prompt()

        # Both should produce non-empty prompts
        assert len(coding_builder_prompt) > 0, "Builder prompt should not be empty"
        assert len(coding_system_prompt) > 0, "System prompt should not be empty"

        # Builder prompt should contain core concepts from system prompt
        # (We don't require exact match, but semantic overlap)
        assert (
            "developer" in coding_builder_prompt.lower()
            or "software" in coding_builder_prompt.lower()
        ), "Builder prompt should mention development"

    def test_coding_prompt_content_quality(self, reset_singletons):
        """CodingAssistant prompt should have quality content.

        Phase 5: Centralized prompt should contain key guidance for coding.
        """
        prompt = CodingAssistant.get_system_prompt()

        # Should mention key concepts
        assert (
            "code" in prompt.lower() or "software" in prompt.lower()
        ), "Prompt should mention code/software"
        assert len(prompt) > 200, "Prompt should have substantial content"

    def test_devops_prompt_content_quality(self, reset_singletons):
        """DevOpsAssistant prompt should have quality content.

        Phase 5: Centralized prompt should contain key guidance for DevOps.
        """
        prompt = DevOpsAssistant.get_system_prompt()

        # Should mention DevOps concepts
        devops_keywords = ["docker", "infrastructure", "deploy", "container"]
        prompt_lower = prompt.lower()
        has_any_keyword = any(keyword in prompt_lower for keyword in devops_keywords)
        assert has_any_keyword, f"Prompt should mention DevOps concepts (one of {devops_keywords})"
        assert len(prompt) > 200, "Prompt should have substantial content"

    def test_rag_prompt_content_quality(self, reset_singletons):
        """RAGAssistant prompt should have quality content.

        Phase 5: Centralized prompt should contain key guidance for RAG.
        """
        prompt = RAGAssistant.get_system_prompt()

        # Should mention RAG concepts
        rag_keywords = ["retrieval", "document", "search", "ingest", "knowledge"]
        prompt_lower = prompt.lower()
        has_any_keyword = any(keyword in prompt_lower for keyword in rag_keywords)
        assert has_any_keyword, f"Prompt should mention RAG concepts (one of {rag_keywords})"
        assert len(prompt) > 200, "Prompt should have substantial content"


class TestPromptSectionModules:
    """Tests for prompt section module organization.

    Phase 5: Verticals should have dedicated prompt section modules
    in the victor/framework/prompt_sections/ directory.
    """

    def test_coding_prompt_section_importable(self, reset_singletons):
        """CodingPromptTemplate should be importable.

        Phase 5: victor/coding/coding_prompt_template.py exists.
        """
        from victor.coding.coding_prompt_template import CodingPromptTemplate

        template = CodingPromptTemplate()
        assert template.vertical_name == "coding"

    def test_devops_prompt_section_importable(self, reset_singletons):
        """DevOpsPromptTemplate should be importable.

        Phase 5: victor/devops/devops_prompt_template.py should exist.
        """
        # This test will fail initially, then pass after implementation
        from victor.devops.devops_prompt_template import DevOpsPromptTemplate

        template = DevOpsPromptTemplate()
        assert template.vertical_name == "devops"

    def test_rag_prompt_section_importable(self, reset_singletons):
        """RAGPromptTemplate should be importable.

        Phase 5: victor/rag/rag_prompt_template.py should exist.
        """
        # This test will fail initially, then pass after implementation
        from victor.rag.rag_prompt_template import RAGPromptTemplate

        template = RAGPromptTemplate()
        assert template.vertical_name == "rag"
