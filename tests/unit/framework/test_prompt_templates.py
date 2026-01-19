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

"""Unit tests for victor.framework.prompts.prompt_templates module."""

import pytest

from victor.framework.prompts.prompt_templates import (
    ANALYSIS_WORKFLOW_TEMPLATE,
    BUG_FIX_WORKFLOW_TEMPLATE,
    BENCHMARK_IDENTITY_TEMPLATE,
    CODE_GENERATION_WORKFLOW_TEMPLATE,
    CODE_QUALITY_CHECKLIST_TEMPLATE,
    CODING_GUIDELINES_TEMPLATE,
    CODING_IDENTITY_TEMPLATE,
    CODING_PITFALLS_TEMPLATE,
    DATA_ANALYSIS_IDENTITY_TEMPLATE,
    DATA_QUALITY_CHECKLIST_TEMPLATE,
    DATA_ANALYSIS_PITFALLS_TEMPLATE,
    DEVOPS_GUIDELINES_TEMPLATE,
    DEVOPS_IDENTITY_TEMPLATE,
    DEVOPS_PITFALLS_TEMPLATE,
    RAG_IDENTITY_TEMPLATE,
    RESEARCH_GUIDELINES_TEMPLATE,
    RESEARCH_IDENTITY_TEMPLATE,
    RESEARCH_QUALITY_CHECKLIST_TEMPLATE,
    SECURITY_CHECKLIST_TEMPLATE,
    TemplateBuilder,
    TOOL_USAGE_CODING_TEMPLATE,
    TOOL_USAGE_DATA_ANALYSIS_TEMPLATE,
    TOOL_USAGE_DEVOPS_TEMPLATE,
    TOOL_USAGE_RAG_TEMPLATE,
    TOOL_USAGE_RESEARCH_TEMPLATE,
)


class TestIdentityTemplates:
    """Tests for identity templates."""

    def test_coding_identity_template(self):
        """Test coding identity template."""
        identity = CODING_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "software development" in identity
        assert "semantic search" in identity
        assert "LSP integration" in identity
        assert "Git operations" in identity

    def test_coding_identity_with_variable(self):
        """Test coding identity template with variable substitution."""
        identity = CODING_IDENTITY_TEMPLATE.format(additional_capabilities="- Custom capability")
        assert "Custom capability" in identity
        assert "{additional_capabilities}" not in identity

    def test_devops_identity_template(self):
        """Test DevOps identity template."""
        identity = DEVOPS_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "DevOps" in identity
        assert "Docker" in identity
        assert "Kubernetes" in identity
        assert "Terraform" in identity

    def test_research_identity_template(self):
        """Test research identity template."""
        identity = RESEARCH_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "research" in identity
        assert "Web search" in identity
        assert "citation" in identity

    def test_data_analysis_identity_template(self):
        """Test data analysis identity template."""
        identity = DATA_ANALYSIS_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "data analysis" in identity
        assert "Statistical analysis" in identity
        assert "visualization" in identity

    def test_benchmark_identity_template(self):
        """Test benchmark identity template."""
        identity = BENCHMARK_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "benchmark" in identity
        assert "evaluation" in identity

    def test_rag_identity_template(self):
        """Test RAG identity template."""
        identity = RAG_IDENTITY_TEMPLATE
        assert "Victor" in identity
        assert "knowledge retrieval" in identity
        assert "Document ingestion" in identity


class TestToolUsageTemplates:
    """Tests for tool usage templates."""

    def test_coding_tool_usage_template(self):
        """Test coding tool usage template."""
        usage = TOOL_USAGE_CODING_TEMPLATE
        assert "semantic_code_search" in usage
        assert "code_search" in usage
        assert "edit" in usage
        assert "write" in usage

    def test_research_tool_usage_template(self):
        """Test research tool usage template."""
        usage = TOOL_USAGE_RESEARCH_TEMPLATE
        assert "web_search" in usage
        assert "web_fetch" in usage
        assert "cite sources" in usage
        assert "URLs" in usage

    def test_devops_tool_usage_template(self):
        """Test DevOps tool usage template."""
        usage = TOOL_USAGE_DEVOPS_TEMPLATE
        assert "docker-compose config" in usage
        assert "terraform validate" in usage
        assert "infrastructure as code principles" in usage

    def test_data_analysis_tool_usage_template(self):
        """Test data analysis tool usage template."""
        usage = TOOL_USAGE_DATA_ANALYSIS_TEMPLATE
        assert "pd.read_csv" in usage
        assert "df.info" in usage
        assert "plt.plot" in usage
        assert "sns.heatmap" in usage

    def test_rag_tool_usage_template(self):
        """Test RAG tool usage template."""
        usage = TOOL_USAGE_RAG_TEMPLATE
        assert "rag_query" in usage
        assert "rag_search" in usage
        assert "rag_ingest" in usage
        assert "Citation Format" in usage


class TestChecklistTemplates:
    """Tests for checklist templates."""

    def test_security_checklist_template(self):
        """Test security checklist template."""
        checklist = SECURITY_CHECKLIST_TEMPLATE
        assert "Security Checklist" in checklist
        assert "- [ ] No hardcoded secrets" in checklist
        assert "- [ ] Using least-privilege IAM" in checklist
        assert "Network traffic encrypted in transit" in checklist

    def test_code_quality_checklist_template(self):
        """Test code quality checklist template."""
        checklist = CODE_QUALITY_CHECKLIST_TEMPLATE
        assert "Code Quality Checklist" in checklist
        assert "- [ ] Code follows existing style" in checklist
        assert "- [ ] Tests pass" in checklist
        assert "- [ ] No regressions" in checklist

    def test_research_quality_checklist_template(self):
        """Test research quality checklist template."""
        checklist = RESEARCH_QUALITY_CHECKLIST_TEMPLATE
        assert "Research Quality Checklist" in checklist
        assert "- [ ] All claims have cited sources" in checklist
        assert "Sources are authoritative" in checklist
        assert "- [ ] URLs" in checklist

    def test_data_quality_checklist_template(self):
        """Test data quality checklist template."""
        checklist = DATA_QUALITY_CHECKLIST_TEMPLATE
        assert "Data Quality Checklist" in checklist
        assert "- [ ] Missing values" in checklist
        assert "- [ ] Outliers" in checklist
        assert "- [ ] Code is reproducible" in checklist


class TestGuidelinesTemplates:
    """Tests for guidelines templates."""

    def test_coding_guidelines_template(self):
        """Test coding guidelines template."""
        guidelines = CODING_GUIDELINES_TEMPLATE
        assert "Guidelines:" in guidelines
        assert "Understand before modifying" in guidelines
        assert "Incremental changes" in guidelines
        assert "Verify changes" in guidelines

    def test_devops_guidelines_template(self):
        """Test DevOps guidelines template."""
        guidelines = DEVOPS_GUIDELINES_TEMPLATE
        assert "Infrastructure as Code" in guidelines
        assert "Immutable infrastructure" in guidelines
        assert "Automation first" in guidelines

    def test_research_guidelines_template(self):
        """Test research guidelines template."""
        guidelines = RESEARCH_GUIDELINES_TEMPLATE
        assert "Source quality" in guidelines
        assert "Multiple perspectives" in guidelines
        assert "Verification" in guidelines


class TestPitfallsTemplates:
    """Tests for pitfalls templates."""

    def test_coding_pitfalls_template(self):
        """Test coding pitfalls template."""
        pitfalls = CODING_PITFALLS_TEMPLATE
        assert "Common Pitfalls" in pitfalls
        assert "Reading too much" in pitfalls
        assert "Large refactors" in pitfalls
        assert "Ignoring tests" in pitfalls

    def test_devops_pitfalls_template(self):
        """Test DevOps pitfalls template."""
        pitfalls = DEVOPS_PITFALLS_TEMPLATE
        assert "Docker" in pitfalls
        assert "Kubernetes" in pitfalls
        assert "Terraform" in pitfalls
        assert "`latest` tag" in pitfalls

    def test_data_analysis_pitfalls_template(self):
        """Test data analysis pitfalls template."""
        pitfalls = DATA_ANALYSIS_PITFALLS_TEMPLATE
        assert "Not checking data quality" in pitfalls
        assert "Ignoring missing values" in pitfalls
        assert "Confusion correlation" in pitfalls


class TestWorkflowTemplates:
    """Tests for workflow templates."""

    def test_bug_fix_workflow_template(self):
        """Test bug fix workflow template."""
        workflow = BUG_FIX_WORKFLOW_TEMPLATE
        assert "Bug Fix Workflow" in workflow
        assert "UNDERSTAND" in workflow
        assert "FIX" in workflow
        assert "VERIFY" in workflow
        assert "max 5 file reads" in workflow

    def test_code_generation_workflow_template(self):
        """Test code generation workflow template."""
        workflow = CODE_GENERATION_WORKFLOW_TEMPLATE
        assert "Code Generation Workflow" in workflow
        assert "UNDERSTAND" in workflow
        assert "IMPLEMENT" in workflow
        assert "VERIFY" in workflow

    def test_analysis_workflow_template(self):
        """Test analysis workflow template."""
        workflow = ANALYSIS_WORKFLOW_TEMPLATE
        assert "Analysis Workflow" in workflow
        assert "EXPLORE" in workflow
        assert "ANALYZE" in workflow
        assert "SYNTHESIZE" in workflow
        assert "REPORT" in workflow


class TestTemplateBuilder:
    """Tests for TemplateBuilder."""

    def test_add_identity(self):
        """Test adding identity section."""
        prompt = TemplateBuilder().add_identity(CODING_IDENTITY_TEMPLATE).build()
        assert "Victor" in prompt
        assert "software development" in prompt

    def test_add_tool_usage(self):
        """Test adding tool usage section."""
        prompt = TemplateBuilder().add_tool_usage(TOOL_USAGE_CODING_TEMPLATE).build()
        assert "semantic_code_search" in prompt
        assert "code_search" in prompt

    def test_add_guidelines(self):
        """Test adding guidelines section."""
        prompt = TemplateBuilder().add_guidelines(CODING_GUIDELINES_TEMPLATE).build()
        assert "Guidelines:" in prompt
        assert "Understand before modifying" in prompt

    def test_add_checklist(self):
        """Test adding checklist section."""
        prompt = TemplateBuilder().add_checklist(SECURITY_CHECKLIST_TEMPLATE).build()
        assert "Security Checklist" in prompt
        assert "- [ ] No hardcoded secrets" in prompt

    def test_add_pitfalls(self):
        """Test adding pitfalls section."""
        prompt = TemplateBuilder().add_pitfalls(CODING_PITFALLS_TEMPLATE).build()
        assert "Common Pitfalls" in prompt
        assert "Reading too much" in prompt

    def test_add_workflow(self):
        """Test adding workflow section."""
        prompt = TemplateBuilder().add_workflow(BUG_FIX_WORKFLOW_TEMPLATE).build()
        assert "Bug Fix Workflow" in prompt
        assert "UNDERSTAND" in prompt

    def test_add_custom_section(self):
        """Test adding custom section."""
        custom_content = "## Custom Section\n\nCustom content here"
        prompt = TemplateBuilder().add_section("custom", custom_content).build()
        assert "Custom Section" in prompt
        assert "Custom content here" in prompt

    def test_variable_substitution(self):
        """Test variable substitution in templates."""
        prompt = (
            TemplateBuilder()
            .add_identity(CODING_IDENTITY_TEMPLATE)
            .add_variable("additional_capabilities", "- Custom capability")
            .build()
        )
        assert "Custom capability" in prompt
        assert "{additional_capabilities}" not in prompt

    def test_multiple_variables(self):
        """Test multiple variable substitutions."""
        template = "Var1: {var1}, Var2: {var2}"
        prompt = (
            TemplateBuilder()
            .add_section("test", template)
            .add_variable("var1", "value1")
            .add_variable("var2", "value2")
            .build()
        )
        assert "value1" in prompt
        assert "value2" in prompt
        assert "{var1}" not in prompt
        assert "{var2}" not in prompt

    def test_build_complete_prompt(self):
        """Test building complete prompt with multiple sections."""
        prompt = (
            TemplateBuilder()
            .add_identity(CODING_IDENTITY_TEMPLATE)
            .add_tool_usage(TOOL_USAGE_CODING_TEMPLATE)
            .add_guidelines(CODING_GUIDELINES_TEMPLATE)
            .add_checklist(CODE_QUALITY_CHECKLIST_TEMPLATE)
            .add_pitfalls(CODING_PITFALLS_TEMPLATE)
            .build()
        )
        assert "Victor" in prompt
        assert "semantic_code_search" in prompt
        assert "Guidelines:" in prompt
        assert "Code Quality Checklist" in prompt
        assert "Common Pitfalls" in prompt

    def test_empty_builder(self):
        """Test building with empty builder."""
        prompt = TemplateBuilder().build()
        assert prompt == ""

    def test_section_ordering(self):
        """Test that sections are added in order."""
        prompt = (
            TemplateBuilder()
            .add_section("section1", "Content 1")
            .add_section("section2", "Content 2")
            .add_section("section3", "Content 3")
            .build()
        )
        lines = prompt.split("\n\n")
        assert "Content 1" in lines[0]
        assert "Content 2" in lines[1]
        assert "Content 3" in lines[2]


class TestTemplateFormatting:
    """Tests for template formatting and structure."""

    def test_templates_are_strings(self):
        """Test that all templates are strings."""
        templates = [
            CODING_IDENTITY_TEMPLATE,
            DEVOPS_IDENTITY_TEMPLATE,
            RESEARCH_IDENTITY_TEMPLATE,
            DATA_ANALYSIS_IDENTITY_TEMPLATE,
            BENCHMARK_IDENTITY_TEMPLATE,
            RAG_IDENTITY_TEMPLATE,
            TOOL_USAGE_CODING_TEMPLATE,
            TOOL_USAGE_RESEARCH_TEMPLATE,
            TOOL_USAGE_DEVOPS_TEMPLATE,
            TOOL_USAGE_DATA_ANALYSIS_TEMPLATE,
            TOOL_USAGE_RAG_TEMPLATE,
            SECURITY_CHECKLIST_TEMPLATE,
            CODE_QUALITY_CHECKLIST_TEMPLATE,
            RESEARCH_QUALITY_CHECKLIST_TEMPLATE,
            DATA_QUALITY_CHECKLIST_TEMPLATE,
            CODING_GUIDELINES_TEMPLATE,
            DEVOPS_GUIDELINES_TEMPLATE,
            RESEARCH_GUIDELINES_TEMPLATE,
            CODING_PITFALLS_TEMPLATE,
            DEVOPS_PITFALLS_TEMPLATE,
            DATA_ANALYSIS_PITFALLS_TEMPLATE,
            BUG_FIX_WORKFLOW_TEMPLATE,
            CODE_GENERATION_WORKFLOW_TEMPLATE,
            ANALYSIS_WORKFLOW_TEMPLATE,
        ]
        for template in templates:
            assert isinstance(template, str)
            assert len(template) > 0

    def test_templates_no_trailing_whitespace(self):
        """Test that templates don't have trailing whitespace."""
        templates = [
            CODING_IDENTITY_TEMPLATE,
            CODING_GUIDELINES_TEMPLATE,
            SECURITY_CHECKLIST_TEMPLATE,
        ]
        for template in templates:
            # Check that lines don't end with spaces
            lines = template.split("\n")
            for line in lines:
                assert line == line.rstrip(), f"Line has trailing whitespace: {line!r}"
