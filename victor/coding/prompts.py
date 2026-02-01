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

"""Coding-specific prompt contributions.

This module provides task type hints and system prompt sections
specific to software development tasks. These are injected into
the framework via the PromptContributorProtocol.

Common patterns are now imported from victor.framework.prompts to reduce
duplication and maintain consistency across verticals.
"""

from __future__ import annotations


from victor.core.verticals.protocols import PromptContributorProtocol, TaskTypeHint
from victor.core.vertical_types import StandardTaskHints
from victor.framework.prompts import (
    GroundingRulesBuilder,
    SystemPromptBuilder,
    TOOL_USAGE_CODING_TEMPLATE,
)


# Task-type-specific prompt hints for coding tasks
# These guide the model's approach based on detected task type
CODING_TASK_TYPE_HINTS: dict[str, TaskTypeHint] = {
    "code_generation": TaskTypeHint(
        task_type="code_generation",
        hint="[GENERATE] Write code directly. No exploration needed. Complete implementation.",
        tool_budget=3,
        priority_tools=["write"],
    ),
    "create_simple": TaskTypeHint(
        task_type="create_simple",
        hint="[CREATE] Write file immediately. Skip codebase exploration. One tool call max.",
        tool_budget=2,
        priority_tools=["write"],
    ),
    "create": TaskTypeHint(
        task_type="create",
        hint="[CREATE+CONTEXT] Read 1-2 relevant files, then create. Follow existing patterns.",
        tool_budget=5,
        priority_tools=["read", "write"],
    ),
    "edit": TaskTypeHint(
        task_type="edit",
        hint="[EDIT] Read target file first, then modify. Focused changes only.",
        tool_budget=5,
        priority_tools=["read", "edit"],
    ),
    "search": TaskTypeHint(
        task_type="search",
        hint="[SEARCH] Use grep/ls for exploration. Summarize after 2-4 calls.",
        tool_budget=6,
        priority_tools=["grep", "ls", "code_search"],
    ),
    "action": TaskTypeHint(
        task_type="action",
        hint="[ACTION] Execute git/test/build operations. Multiple tool calls allowed. Continue until complete.",
        tool_budget=15,
        priority_tools=["shell", "git", "test"],
    ),
    "analysis_deep": TaskTypeHint(
        task_type="analysis_deep",
        hint="[ANALYSIS] Thorough codebase exploration. Read all relevant modules. Comprehensive output.",
        tool_budget=25,
        priority_tools=["read", "grep", "code_search", "ls"],
    ),
    "analyze": TaskTypeHint(
        task_type="analyze",
        hint="[ANALYZE] Examine code carefully. Read related files. Structured findings.",
        tool_budget=12,
        priority_tools=["read", "grep", "symbol"],
    ),
    "design": TaskTypeHint(
        task_type="design",
        hint="""[ARCHITECTURE] For architecture/component questions:
USE STRUCTURED GRAPH FIRST:
- Call architecture_summary to get module pagerank/centrality with edge_counts + 2–3 callsites (runtime-only). Avoid ad-hoc graph/find hops unless data is missing.
- Keep modules vs symbols separate; cite CALLS/INHERITS/IMPORTS counts and callsites (file:line) per hotspot.
- Prefer runtime code; ignore tests/venv/build outputs unless explicitly requested.
DOC-FIRST STRATEGY (mandatory order):
1. FIRST: Read architecture docs if they exist:
   - read_file CLAUDE.md, .victor/init.md, README.md, ARCHITECTURE.md
   - These contain component lists, named implementations, and key relationships
2. SECOND: Explore implementation directories systematically:
   - list_directory on src/, lib/, engines/, impls/, modules/, core/, services/
   - Directory names under impls/ or engines/ are often named implementations
   - Look for ALL-CAPS directory/file names - these are typically named engines/components
3. THIRD: Read key implementation files for each component found
4. FOURTH: Look for benchmark/test files (benches/, *_bench*, *_test*) for performance insights

DISCOVERY PATTERNS - Look for:
- Named implementations: Directories with ALL-CAPS names (engines, stores, protocols)
- Factories/registries: Files named *_factory.*, *_registry.*, mod.rs, index.ts
- Core abstractions: base.py, interface.*, trait definitions
- Configuration: *.yaml, *.toml in config/ directories

Output requirements:
- Use discovered component names (not generic descriptions like "storage module")
- Include file:line references (e.g., "src/engines/impl.rs:42")
- Verify improvements reference ACTUAL code patterns (grep first)
Use 15-20 tool calls minimum. Prioritize by architectural importance.""",
        tool_budget=25,
        priority_tools=["read", "ls", "grep", "code_search"],
    ),
    "refactor": TaskTypeHint(
        task_type="refactor",
        hint="[REFACTOR] Analyze code structure first. Use refactoring tools. Verify with tests.",
        tool_budget=15,
        priority_tools=["read", "rename", "test"],
    ),
    "debug": TaskTypeHint(
        task_type="debug",
        hint="[DEBUG] Read error context. Trace execution flow. Find root cause before fixing.",
        tool_budget=12,
        priority_tools=["read", "grep", "shell"],
    ),
    "test": TaskTypeHint(
        task_type="test",
        hint="[TEST] Run tests first. Analyze failures. Fix issues incrementally.",
        tool_budget=10,
        priority_tools=["test", "read", "edit"],
    ),
    "general": TaskTypeHint(
        task_type="general",
        hint="[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.",
        tool_budget=8,
        priority_tools=["read", "grep", "ls"],
    ),
    "bug_fix": TaskTypeHint(
        task_type="bug_fix",
        hint="""[BUG FIX] Resolve a GitHub issue or bug report. CRITICAL WORKFLOW:

PHASE 1 - UNDERSTAND (max 5 file reads):
1. Read the file(s) mentioned in the error traceback/issue
2. Read related imports and dependencies (1-2 files max)
3. Identify the root cause from the code

PHASE 2 - FIX (MANDATORY after Phase 1):
4. Use edit_file or write_file to make the fix
5. The fix should be minimal and surgical - only change what's necessary
6. If the issue suggests a fix (e.g., "add quiet=True"), implement exactly that

PHASE 3 - VERIFY (optional):
7. If tests exist, run them to verify the fix

CRITICAL RULES:
- DO NOT read more than 5-7 files before making an edit
- After reading the traceback/error location, you have enough context to edit
- Prefer SMALL, FOCUSED changes over large refactors
- If unsure, make the minimal fix that addresses the reported issue
- Say "Fix applied" when done editing

ANTI-PATTERNS TO AVOID:
- Reading the entire codebase before editing
- Exploring tangential files not in the error trace
- Waiting for "perfect understanding" before acting
- Re-reading files you've already read""",
        tool_budget=12,
        priority_tools=["read", "edit", "test", "shell"],
    ),
    "issue_resolution": TaskTypeHint(
        task_type="issue_resolution",
        hint="[ISSUE] Same as bug_fix - resolve GitHub issue with focused edits after minimal exploration.",
        tool_budget=12,
        priority_tools=["read", "edit", "test", "shell"],
    ),
}

# Merge with standard task hints to provide common defaults across verticals
CODING_TASK_TYPE_HINTS = StandardTaskHints.merge_with(CODING_TASK_TYPE_HINTS)


# Coding-specific grounding rules (now using framework builders)
CODING_GROUNDING_RULES = GroundingRulesBuilder().minimal().build()

CODING_GROUNDING_EXTENDED = GroundingRulesBuilder().extended().build()


# Coding-specific system prompt section (now using framework templates)
CODING_SYSTEM_PROMPT_SECTION = (
    SystemPromptBuilder()
    .with_tool_usage(TOOL_USAGE_CODING_TEMPLATE)
    .with_guidelines(
        [
            "Understand before modifying: Always read and understand code before making changes",
            "Incremental changes: Make small, focused changes rather than large rewrites",
            "Verify changes: Run tests or validation after modifications",
            "Explain reasoning: Briefly explain your approach when making non-trivial changes",
            "Preserve style: Match existing code style and patterns",
            "Handle errors gracefully: If something fails, diagnose and recover",
        ]
    )
    .build()
)


class CodingPromptContributor(PromptContributorProtocol):
    """Prompt contributor for coding vertical.

    Provides coding-specific task type hints and system prompt sections
    for integration with the framework's prompt builder.
    """

    def __init__(self, use_extended_grounding: bool = False):
        """Initialize the prompt contributor.

        Args:
            use_extended_grounding: Whether to use extended grounding rules
                                   (typically for local models)
        """
        self._use_extended_grounding = use_extended_grounding

    def get_task_type_hints(self) -> dict[str, TaskTypeHint]:
        """Get coding-specific task type hints.

        Returns:
            Dict mapping task types to their hints
        """
        return CODING_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Get coding-specific system prompt section.

        Returns:
            System prompt text for coding tasks
        """
        return CODING_SYSTEM_PROMPT_SECTION

    def get_grounding_rules(self) -> str:
        """Get coding-specific grounding rules.

        Returns:
            Grounding rules text
        """
        if self._use_extended_grounding:
            return CODING_GROUNDING_EXTENDED
        return CODING_GROUNDING_RULES

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value (coding is primary, so high priority)
        """
        return 10


def get_task_type_hint(task_type: str) -> str:
    """Get prompt hint for a specific task type.

    Convenience function for backward compatibility.

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    hint = CODING_TASK_TYPE_HINTS.get(task_type.lower())
    return hint.hint if hint else ""


__all__ = [
    "CodingPromptContributor",
    "CODING_TASK_TYPE_HINTS",
    "CODING_GROUNDING_RULES",
    "CODING_GROUNDING_EXTENDED",
    "CODING_SYSTEM_PROMPT_SECTION",
    "get_task_type_hint",
]
