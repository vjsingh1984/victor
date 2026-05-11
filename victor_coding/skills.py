"""Coding skills — composable expertise units for the coding vertical.

Each SkillDefinition binds a focused prompt fragment with the tools
needed to execute a specific coding task pattern.

Usage:
    from victor_coding.skills import CODING_SKILLS

    # Register with SkillRegistry
    for skill in CODING_SKILLS:
        registry.register(skill)

    # Or via CodingAssistant.get_skills()
"""

from __future__ import annotations

from victor_sdk.skills import SkillDefinition


debug_test_failure = SkillDefinition(
    name="debug_test_failure",
    description="Diagnose and fix a failing test by reading the test, tracing the code path, and applying a targeted fix",
    category="coding",
    prompt_fragment=(
        "SKILL: debug_test_failure\n"
        "1. Read the failing test to understand the assertion\n"
        "2. Run the test to see the exact error output\n"
        "3. Trace the code path from test → source to find the bug\n"
        "4. Apply a minimal fix to the source (not the test)\n"
        "5. Re-run the test to confirm it passes\n"
        "6. Run related tests to check for regressions"
    ),
    required_tools=["read", "shell", "edit"],
    optional_tools=["grep", "code_search", "ls"],
    tags=frozenset({"debug", "test", "fix", "tdd"}),
    max_tool_calls=25,
)

code_review = SkillDefinition(
    name="code_review",
    description="Review code changes for correctness, style, security issues, and suggest improvements",
    category="coding",
    prompt_fragment=(
        "SKILL: code_review\n"
        "1. Read the diff or changed files to understand the scope\n"
        "2. Check for correctness: logic errors, edge cases, error handling\n"
        "3. Check for style: naming, structure, consistency with codebase\n"
        "4. Check for security: injection, auth, secrets, OWASP top 10\n"
        "5. Check for performance: unnecessary allocations, N+1 queries\n"
        "6. Provide actionable feedback with specific line references"
    ),
    required_tools=["read", "grep"],
    optional_tools=["code_search", "shell", "ls"],
    tags=frozenset({"review", "quality", "security"}),
    max_tool_calls=15,
)

implement_feature = SkillDefinition(
    name="implement_feature",
    description="Implement a new feature by reading existing patterns, writing code, and verifying with tests",
    category="coding",
    prompt_fragment=(
        "SKILL: implement_feature\n"
        "1. Read related files to understand existing patterns and conventions\n"
        "2. Plan the implementation: what files to create/modify\n"
        "3. Write the implementation following existing code style\n"
        "4. Write or update tests for the new feature\n"
        "5. Run tests to verify the implementation\n"
        "6. Run linting to ensure code quality"
    ),
    required_tools=["read", "edit", "write", "shell"],
    optional_tools=["grep", "code_search", "ls", "git"],
    tags=frozenset({"feature", "implement", "create", "write"}),
    max_tool_calls=30,
)

refactor_code = SkillDefinition(
    name="refactor_code",
    description="Refactor code to improve structure, readability, or performance without changing behavior",
    category="coding",
    prompt_fragment=(
        "SKILL: refactor_code\n"
        "1. Read the target code thoroughly to understand current behavior\n"
        "2. Identify the refactoring pattern (extract method, rename, etc.)\n"
        "3. Run existing tests to establish baseline (must all pass)\n"
        "4. Apply refactoring in small, incremental steps\n"
        "5. Run tests after each step to verify behavior is preserved\n"
        "6. Verify no regressions with full test suite"
    ),
    required_tools=["read", "edit", "shell"],
    optional_tools=["grep", "code_search", "git"],
    tags=frozenset({"refactor", "cleanup", "improve", "restructure"}),
    max_tool_calls=25,
)

explore_codebase = SkillDefinition(
    name="explore_codebase",
    description="Explore and understand a codebase's architecture, patterns, and key components",
    category="coding",
    prompt_fragment=(
        "SKILL: explore_codebase\n"
        "1. Start with project structure: ls the root and key directories\n"
        "2. Read entry points, main modules, and configuration files\n"
        "3. Use grep/code_search to find key patterns and abstractions\n"
        "4. Trace the main data/control flow through the system\n"
        "5. Identify architectural patterns (MVC, DI, event-driven, etc.)\n"
        "6. Summarize findings: modules, boundaries, dependencies, hot paths"
    ),
    required_tools=["read", "ls", "grep"],
    optional_tools=["code_search"],
    tags=frozenset({"explore", "understand", "architecture", "analysis"}),
    max_tool_calls=20,
)


CODING_SKILLS = [
    debug_test_failure,
    code_review,
    implement_feature,
    refactor_code,
    explore_codebase,
]
