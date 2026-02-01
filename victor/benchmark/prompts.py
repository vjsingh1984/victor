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

"""Benchmark-specific prompt contributions.

This module provides task type hints and system prompt sections
specific to benchmark evaluation tasks. These are injected into
the framework via the PromptContributorProtocol.
"""

from __future__ import annotations


from victor.core.verticals.protocols import PromptContributorProtocol, TaskTypeHint


# Task-type-specific prompt hints for benchmark tasks
# These guide the model's approach based on detected task type
BENCHMARK_TASK_TYPE_HINTS: dict[str, TaskTypeHint] = {
    "swe_bench_issue": TaskTypeHint(
        task_type="swe_bench_issue",
        hint="""[SWE-BENCH ISSUE] Resolve GitHub issue with minimal changes.

CRITICAL WORKFLOW:
1. UNDERSTAND: Read issue description and error traceback carefully (1-2 files)
2. LOCATE: Use code_search to find the exact location of the bug
3. FIX: Make surgical changes - only modify what's necessary
4. VERIFY: Run provided tests to confirm the fix works

RULES:
- Read ONLY the files mentioned in the traceback/issue (max 5 files)
- Make MINIMAL, FOCUSED changes - don't refactor
- Follow existing code patterns and style
- Generate clean patches using git diff
- Run tests to verify your solution works

You are evaluated on:
- Correctness: Does the fix pass all tests?
- Minimality: Did you make the smallest necessary change?
- Efficiency: Did you find and fix the bug quickly?""",
        tool_budget=20,
        priority_tools=["read", "code_search", "edit", "test", "git"],
    ),
    "code_generation": TaskTypeHint(
        task_type="code_generation",
        hint="""[CODE GENERATION] Write code to solve a programming problem.

WORKFLOW:
1. UNDERSTAND: Read the problem statement carefully
2. IMPLEMENT: Write clean, correct code that solves the problem
3. VERIFY: Test with provided examples or edge cases

RULES:
- Write complete, runnable implementations
- Follow language best practices and idioms
- Include necessary imports
- Handle edge cases appropriately
- Add brief comments for complex logic only

You are evaluated on:
- Correctness: Does your code solve the problem?
- Quality: Is your code clean and well-structured?
- Efficiency: Does your code run within time limits?""",
        tool_budget=5,
        priority_tools=["write", "read", "shell"],
    ),
    "function_completion": TaskTypeHint(
        task_type="function_completion",
        hint="""[FUNCTION COMPLETION] Complete the function body to pass tests.

WORKFLOW:
1. READ: Read the incomplete function and understand its signature
2. IMPLEMENT: Write the function body based on docstring/tests
3. TEST: Run tests to verify your implementation

RULES:
- Keep the function signature unchanged
- Implement logic based on docstring description
- Consider edge cases and error conditions
- Write idiomatic code for the language

You are evaluated on:
- Correctness: Does your implementation pass all tests?
- Completeness: Does it handle all required cases?""",
        tool_budget=3,
        priority_tools=["read", "edit", "test"],
    ),
    "bug_fixing": TaskTypeHint(
        task_type="bug_fixing",
        hint="""[BUG FIXING] Identify and fix bugs in existing code.

WORKFLOW:
1. ANALYZE: Read the buggy code and understand expected behavior
2. DEBUG: Identify the root cause of the bug
3. FIX: Make minimal changes to fix the bug
4. VERIFY: Run tests to confirm the fix works

RULES:
- Make surgical fixes - only change what's necessary
- Preserve existing logic and style
- Add comments explaining the fix if it's subtle
- Test edge cases after fixing

You are evaluated on:
- Correctness: Does the fix resolve the bug?
- Minimality: Did you make the smallest necessary change?""",
        tool_budget=8,
        priority_tools=["read", "edit", "test", "shell"],
    ),
    "code_review": TaskTypeHint(
        task_type="code_review",
        hint="""[CODE REVIEW] Analyze code for issues and improvements.

WORKFLOW:
1. READ: Read the code carefully
2. ANALYZE: Identify bugs, style issues, and improvements
3. REPORT: Provide structured feedback with specific examples

FOCUS AREAS:
- Correctness: Logic errors, edge cases
- Style: Formatting, naming, idiomatic usage
- Performance: Inefficient algorithms or data structures
- Security: Potential vulnerabilities
- Maintainability: Code clarity and complexity

You are evaluated on:
- Thoroughness: Did you catch all issues?
- Accuracy: Are your findings valid?
- Actionability: Is your feedback specific and helpful?""",
        tool_budget=15,
        priority_tools=["read", "grep", "code_search"],
    ),
    "test_generation": TaskTypeHint(
        task_type="test_generation",
        hint="""[TEST GENERATION] Write comprehensive tests for code.

WORKFLOW:
1. UNDERSTAND: Read the code to understand its behavior
2. DESIGN: Design test cases covering functionality and edge cases
3. IMPLEMENT: Write clear, maintainable tests
4. VERIFY: Run tests to ensure they pass

RULES:
- Test normal cases and edge cases
- Test error conditions and boundary values
- Use descriptive test names
- Follow the testing framework's conventions
- Keep tests independent and fast

You are evaluated on:
- Coverage: Do tests cover the code thoroughly?
- Quality: Are tests well-written and maintainable?
- Effectiveness: Do tests catch real bugs?""",
        tool_budget=12,
        priority_tools=["read", "write", "test", "shell"],
    ),
    "passk_sampling": TaskTypeHint(
        task_type="passk_sampling",
        hint="""[PASS@K SAMPLING] Generate multiple solution candidates.

GOAL: Generate k distinct, high-quality solution candidates.

APPROACH:
1. UNDERSTAND: Read the problem statement
2. DIVERSIFY: Generate solutions with different approaches:
   - Alternative algorithms or data structures
   - Different implementation styles
   - Varying levels of optimization
3. VERIFY: Test each candidate independently

RULES:
- Each candidate must be a complete solution
- Explore legitimate algorithmic alternatives
- Avoid trivial variations (renaming variables)
- Focus on correct, diverse implementations

You are evaluated on:
- Pass@k: What fraction of candidates pass tests?
- Diversity: Are solutions meaningfully different?
- Quality: Are all solutions well-written?""",
        tool_budget=50,
        priority_tools=["read", "write", "test", "shell"],
    ),
    "benchmark_analysis": TaskTypeHint(
        task_type="benchmark_analysis",
        hint="""[BENCHMARK ANALYSIS] Analyze benchmark results and performance.

WORKFLOW:
1. COLLECT: Gather benchmark metrics and results
2. ANALYZE: Identify patterns, trends, and anomalies
3. REPORT: Provide insights with data-backed recommendations

METRICS TO ANALYZE:
- Pass rates and success rates
- Execution time and latency
- Token usage and cost
- Tool call efficiency
- Error patterns

You are evaluated on:
- Accuracy: Are your insights correct?
- Depth: Did you identify meaningful patterns?
- Actionability: Are your recommendations useful?""",
        tool_budget=10,
        priority_tools=["read", "grep", "shell"],
    ),
}


# Benchmark-specific grounding rules
BENCHMARK_GROUNDING_RULES = """
GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Quote code exactly from tool output. If more info needed, call another tool.

BENCHMARK EVALUATION: Your responses will be evaluated for correctness and quality.
Focus on precision and accuracy in all interactions.
""".strip()


# Extended grounding for benchmark evaluation
BENCHMARK_GROUNDING_EXTENDED = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

BENCHMARK EVALUATION CONTEXT:
- You are being evaluated on correctness and efficiency
- Minimal, focused changes are preferred over large refactors
- Test verification is critical for confirming solutions
- Precision matters more than speed

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS AND POOR EVALUATION SCORES.
""".strip()


# Benchmark-specific system prompt section
BENCHMARK_SYSTEM_PROMPT_SECTION = """
BENCHMARK EXECUTION GUIDELINES:

Task Understanding:
- Read the problem/issue description carefully before taking action
- Identify the specific requirements and constraints
- Understand what success looks like (passing tests, fixing bug, etc.)

Code Analysis:
- Use semantic_code_search for conceptual queries ("authentication logic")
- Use code_search for exact patterns ("def authenticate")
- Read files mentioned in error traces first
- Focus on relevant files, avoid exploring entire codebase

Code Modification:
- Make MINIMAL, FOCUSED changes - only modify what's necessary
- Use edit for surgical changes to existing code
- Use write only for new files or complete rewrites
- Follow existing code style and patterns
- Preserve existing functionality unless directed otherwise

Verification:
- ALWAYS run tests after making changes
- Verify fixes with the actual test suite
- Check for regressions in related functionality
- Use git diff to review your changes before committing

Performance Optimization:
- Prioritize correctness over performance optimization
- Only optimize if explicitly requested
- Profile before optimizing to identify actual bottlenecks

You are being evaluated on:
1. Correctness: Does your solution pass tests / solve the problem?
2. Minimality: Did you make the smallest necessary change?
3. Efficiency: Did you find the solution quickly with minimal tool calls?
4. Quality: Is your code clean and maintainable?
""".strip()


class BenchmarkPromptContributor(PromptContributorProtocol):
    """Prompt contributor for benchmark vertical.

    Provides benchmark-specific task type hints and system prompt sections
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
        """Get benchmark-specific task type hints.

        Returns:
            Dict mapping task types to their hints
        """
        return BENCHMARK_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Get benchmark-specific system prompt section.

        Returns:
            System prompt text for benchmark tasks
        """
        return BENCHMARK_SYSTEM_PROMPT_SECTION

    def get_grounding_rules(self) -> str:
        """Get benchmark-specific grounding rules.

        Returns:
            Grounding rules text
        """
        if self._use_extended_grounding:
            return BENCHMARK_GROUNDING_EXTENDED
        return BENCHMARK_GROUNDING_RULES

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value (benchmark is domain-specific, so medium-high priority)
        """
        return 8


def get_task_type_hint(task_type: str) -> str:
    """Get prompt hint for a specific task type.

    Convenience function for backward compatibility.

    Args:
        task_type: The detected task type (e.g., "swe_bench_issue", "code_generation")

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    hint = BENCHMARK_TASK_TYPE_HINTS.get(task_type.lower())
    return hint.hint if hint else ""


__all__ = [
    "BenchmarkPromptContributor",
    "BENCHMARK_TASK_TYPE_HINTS",
    "BENCHMARK_GROUNDING_RULES",
    "BENCHMARK_GROUNDING_EXTENDED",
    "BENCHMARK_SYSTEM_PROMPT_SECTION",
    "get_task_type_hint",
]
