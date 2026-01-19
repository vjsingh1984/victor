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

"""Victor Framework Prompt Library.

This package provides reusable prompt patterns, templates, and builders
that eliminate duplication across verticals.

Key Modules:
- common_prompts: Base classes and builders for prompt construction
- prompt_templates: Pre-built templates for common patterns

Usage:
    from victor.framework.prompts import (
        GroundingRulesBuilder,
        TaskHintBuilder,
        CODING_IDENTITY_TEMPLATE,
        TemplateBuilder,
    )

    # Build grounding rules
    grounding = GroundingRulesBuilder().extended().build()

    # Build task hint
    hint = TaskHintBuilder() \\
        .for_task_type("edit") \\
        .with_guidance("[EDIT] Read first, then modify") \\
        .build()

    # Compose templates
    prompt = TemplateBuilder() \\
        .add_identity(CODING_IDENTITY_TEMPLATE) \\
        .add_guidelines(CODING_GUIDELINES_TEMPLATE) \\
        .build()
"""

from victor.framework.prompts.common_prompts import (
    AnalysisWorkflow,
    BugFixWorkflow,
    CodeCreationWorkflow,
    ChecklistBuilder,
    GroundingMode,
    GroundingRulesBuilder,
    SafetyRulesBuilder,
    SystemPromptBuilder,
    TaskCategory,
    TaskHint,
    TaskHintBuilder,
    WorkflowTemplate,
)

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

__all__ = [
    # Common prompts
    "GroundingMode",
    "TaskCategory",
    "GroundingRulesBuilder",
    "TaskHintBuilder",
    "TaskHint",
    "SystemPromptBuilder",
    "ChecklistBuilder",
    "SafetyRulesBuilder",
    "WorkflowTemplate",
    "BugFixWorkflow",
    "CodeCreationWorkflow",
    "AnalysisWorkflow",
    # Prompt templates
    "TemplateBuilder",
    "CODING_IDENTITY_TEMPLATE",
    "DEVOPS_IDENTITY_TEMPLATE",
    "RESEARCH_IDENTITY_TEMPLATE",
    "DATA_ANALYSIS_IDENTITY_TEMPLATE",
    "BENCHMARK_IDENTITY_TEMPLATE",
    "RAG_IDENTITY_TEMPLATE",
    "TOOL_USAGE_CODING_TEMPLATE",
    "TOOL_USAGE_RESEARCH_TEMPLATE",
    "TOOL_USAGE_DEVOPS_TEMPLATE",
    "TOOL_USAGE_DATA_ANALYSIS_TEMPLATE",
    "TOOL_USAGE_RAG_TEMPLATE",
    "SECURITY_CHECKLIST_TEMPLATE",
    "CODE_QUALITY_CHECKLIST_TEMPLATE",
    "RESEARCH_QUALITY_CHECKLIST_TEMPLATE",
    "DATA_QUALITY_CHECKLIST_TEMPLATE",
    "CODING_GUIDELINES_TEMPLATE",
    "DEVOPS_GUIDELINES_TEMPLATE",
    "RESEARCH_GUIDELINES_TEMPLATE",
    "CODING_PITFALLS_TEMPLATE",
    "DEVOPS_PITFALLS_TEMPLATE",
    "DATA_ANALYSIS_PITFALLS_TEMPLATE",
    "BUG_FIX_WORKFLOW_TEMPLATE",
    "CODE_GENERATION_WORKFLOW_TEMPLATE",
    "ANALYSIS_WORKFLOW_TEMPLATE",
]
