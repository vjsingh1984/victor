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

"""Framework-based safety rules for Benchmark vertical.

This module provides factory functions to register benchmark-specific safety rules
with the framework SafetyEnforcer, following the same pattern as Coding, DevOps,
RAG, Research, and DataAnalysis verticals.

Safety rules cover:
- Repository isolation (prevent modifying non-benchmark repos)
- Resource limits (timeout, cost, token usage)
- Test isolation (prevent running tests on production systems)
- Data privacy (prevent uploading benchmark data externally)
"""

from typing import Optional

from victor.framework.config import SafetyEnforcer, SafetyLevel, SafetyRule


def create_benchmark_repository_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_outside_workspace: bool = True,
    protected_repositories: Optional[list[str]] = None,
    block_git_operations_outside_workspace: bool = True,
) -> None:
    """Register benchmark repository safety rules.

    These rules ensure that benchmark operations don't modify repositories
    outside the designated benchmark workspace.

    Args:
        enforcer: SafetyEnforcer instance to register rules with
        block_outside_workspace: Block operations outside benchmark workspace
        protected_repositories: List of repo paths to protect (default: common production paths)
        block_git_operations_outside_workspace: Block git ops outside workspace

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
        from victor.benchmark.safety import create_benchmark_repository_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_benchmark_repository_safety_rules(
            enforcer,
            protected_repositories=["/production", "/main", "/master"]
        )
    """
    if protected_repositories is None:
        protected_repositories = [
            "/production",
            "/prod",
            "/main",
            "/master",
            "release",
            "deploy",
            " main",  # Match "git push origin main"
            " master",  # Match "git push origin master"
        ]

    if block_outside_workspace:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_outside_workspace",
                description="Block benchmark operations outside designated workspace",
                check_fn=lambda op: (
                    any(
                        protected in op.lower()
                        for protected in protected_repositories
                    )
                    and any(
                        tool in op.lower()
                        for tool in ["write", "edit", "delete", "modify", "git"]
                    )
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never allow benchmarking on production repos
            )
        )

    if block_git_operations_outside_workspace:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_git_operations_outside_workspace",
                description="Block git operations (push, commit, force) outside benchmark workspace",
                check_fn=lambda op: any(
                    git_op in op.lower()
                    for git_op in [
                        "git push",
                        "git commit",
                        "git force",
                        "git --force",
                        "git apply",
                    ]
                )
                and any(
                    protected in op.lower()
                    for protected in protected_repositories
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never allow git ops on protected repos
            )
        )


def create_benchmark_resource_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_excessive_timeouts: bool = True,
    max_timeout_seconds: int = 600,
    block_unlimited_budgets: bool = True,
    warn_expensive_operations: bool = True,
) -> None:
    """Register benchmark resource limit safety rules.

    These rules prevent benchmark operations from consuming excessive resources
    (time, tokens, cost) that could impact system performance or cost.

    Args:
        enforcer: SafetyEnforcer instance to register rules with
        block_excessive_timeouts: Block operations with very high timeouts
        max_timeout_seconds: Maximum allowed timeout in seconds (default: 600)
        block_unlimited_budgets: Block unlimited tool budgets
        warn_expensive_operations: Warn about expensive operations

    Example:
        from victor.benchmark.safety import create_benchmark_resource_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_benchmark_resource_safety_rules(
            enforcer,
            max_timeout_seconds=300,
            block_unlimited_budgets=True
        )
    """
    if block_excessive_timeouts:
        # Capture max_timeout_seconds in default arg for lambda
        def check_excessive_timeout(op: str, max_seconds=max_timeout_seconds) -> bool:
            """Check if operation has timeout exceeding max_seconds."""
            # Check for timeout keywords
            if not any(phrase in op.lower() for phrase in ["timeout", "time_limit", "max_time", "duration"]):
                return False

            # Extract all numbers from the operation
            import re
            numbers = re.findall(r'\d+', op)
            for num_str in numbers:
                try:
                    num = int(num_str)
                    if num > max_seconds:
                        return True
                except ValueError:
                    continue
            return False

        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_excessive_timeouts",
                description=f"Block benchmark operations with timeouts exceeding {max_timeout_seconds}s",
                check_fn=check_excessive_timeout,
                level=SafetyLevel.MEDIUM,
                allow_override=True,  # Allow for special cases
            )
        )

    if block_unlimited_budgets:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_unlimited_budgets",
                description="Block unlimited tool budgets in benchmark runs",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "tool_budget=-1",
                        "unlimited budget",
                        "no limit",
                        "budget=999",
                        "budget=1000",
                    ]
                ),
                level=SafetyLevel.MEDIUM,
                allow_override=True,  # Allow for controlled experiments
            )
        )

    if warn_expensive_operations:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_warn_expensive_operations",
                description="Warn about expensive operations in benchmark runs",
                check_fn=lambda op: any(
                    phrase in op.lower()
                    for phrase in [
                        "model=gpt-4",
                        "model=claude-opus",
                        "temperature=1.0",
                        "max_tokens=10000",
                        "max_tokens=20000",
                    ]
                ),
                level=SafetyLevel.LOW,
                allow_override=True,  # Just warn
            )
        )


def create_benchmark_test_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_production_test_runs: bool = True,
    block_destructive_tests: bool = True,
    protected_environments: Optional[list[str]] = None,
) -> None:
    """Register benchmark test execution safety rules.

    These rules prevent benchmark test runs from executing on production
    systems or running destructive tests.

    Args:
        enforcer: SafetyEnforcer instance to register rules with
        block_production_test_runs: Block test runs on production environments
        block_destructive_tests: Block tests that modify data/state
        protected_environments: List of environment names to protect

    Example:
        from victor.benchmark.safety import create_benchmark_test_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_benchmark_test_safety_rules(
            enforcer,
            protected_environments=["production", "staging", "prod"]
        )
    """
    if protected_environments is None:
        protected_environments = [
            "production",
            "prod",
            "staging",
            "live",
            "release",
        ]

    if block_production_test_runs:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_production_test_runs",
                description="Block running benchmark tests on production environments",
                check_fn=lambda op: (
                    any(
                        env in op.lower()
                        for env in protected_environments
                    )
                    and any(
                        test_op in op.lower()
                        for test_op in [
                            "test",
                            "pytest",
                            "run tests",
                            "execute tests",
                            "verify",
                        ]
                    )
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never test on production
            )
        )

    if block_destructive_tests:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_destructive_tests",
                description="Block destructive tests that modify data or state",
                check_fn=lambda op: any(
                    destructive in op.lower()
                    for destructive in [
                        "drop table",
                        "delete from",
                        "truncate",
                        "remove database",
                        "clear cache",
                        "flush",
                        "reset",
                        "wipe",
                    ]
                )
                and "test" in op.lower(),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never allow destructive tests in benchmarks
            )
        )


def create_benchmark_data_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_external_uploads: bool = True,
    block_task_data_leaks: bool = True,
    block_solution_sharing: bool = True,
) -> None:
    """Register benchmark data privacy safety rules.

    These rules prevent benchmark task data, solutions, or results from
    being shared externally or uploaded to external services.

    Args:
        enforcer: SafetyEnforcer instance to register rules with
        block_external_uploads: Block uploading benchmark data externally
        block_task_data_leaks: Block leaking benchmark task data
        block_solution_sharing: Block sharing benchmark solutions externally

    Example:
        from victor.benchmark.safety import create_benchmark_data_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_benchmark_data_safety_rules(
            enforcer,
            block_external_uploads=True,
            block_solution_sharing=True
        )
    """
    if block_external_uploads:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_external_uploads",
                description="Block uploading benchmark data to external services",
                check_fn=lambda op: (
                    # Check for upload/send operation
                    any(
                        upload_op in op.lower()
                        for upload_op in [
                            "upload",
                            "upload to",
                            "send to",
                            "post to",
                            "share with",
                            "sync to",
                        ]
                    )
                    and any(
                        external in op.lower()
                        for external in [
                            "s3",
                            "gcs",
                            "azure",
                            "dropbox",
                            "google drive",
                            "api",
                            "external",
                        ]
                    )
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never leak benchmark data
            )
        )

    if block_task_data_leaks:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_task_data_leaks",
                description="Block leaking benchmark task data or prompts",
                check_fn=lambda op: (
                    # Check for leak operation
                    any(
                        leak_op in op.lower()
                        for leak_op in [
                            "log task",
                            "print prompt",
                            "export task",
                            "export",
                            "save task",
                            "share task",
                        ]
                    )
                    and any(
                        data in op.lower()
                        for data in [
                            "benchmark",
                            "task_id",
                            "task description",
                            "problem statement",
                        ]
                    )
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never leak task data
            )
        )

    if block_solution_sharing:
        enforcer.add_rule(
            SafetyRule(
                name="benchmark_block_solution_sharing",
                description="Block sharing benchmark solutions externally",
                check_fn=lambda op: (
                    # Check for share operation with solution
                    any(
                        share_op in op.lower()
                        for share_op in [
                            "share solution",
                            "share",
                            "publish solution",
                            "upload solution",
                            "post solution",
                            "export solution",
                        ]
                    )
                    and (
                        # Check for benchmark name OR "solution" keyword
                        any(
                            benchmark in op.lower()
                            for benchmark in [
                                "swe-bench",
                                "humaneval",
                                "mbpp",
                                "benchmark",
                            ]
                        )
                        or "solution" in op.lower()
                    )
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,  # Never share benchmark solutions
            )
        )


def create_all_benchmark_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    repository_rules: bool = True,
    resource_rules: bool = True,
    test_rules: bool = True,
    data_rules: bool = True,
    **kwargs,
) -> None:
    """Register all benchmark safety rules.

    Convenience function to register all benchmark-specific safety rules
    with a single call.

    Args:
        enforcer: SafetyEnforcer instance to register rules with
        repository_rules: Register repository isolation rules
        resource_rules: Register resource limit rules
        test_rules: Register test isolation rules
        data_rules: Register data privacy rules
        **kwargs: Additional arguments passed to specific rule functions
            (e.g., protected_repositories, max_timeout_seconds, protected_environments)

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
        from victor.benchmark.safety import create_all_benchmark_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_benchmark_safety_rules(
            enforcer,
            protected_repositories=["/production", "/main"],
            max_timeout_seconds=300
        )
    """
    # Extract kwargs for each rule function
    repository_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {
            "block_outside_workspace",
            "protected_repositories",
            "block_git_operations_outside_workspace",
        }
    }

    resource_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {
            "block_excessive_timeouts",
            "max_timeout_seconds",
            "block_unlimited_budgets",
            "warn_expensive_operations",
        }
    }

    test_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {
            "block_production_test_runs",
            "block_destructive_tests",
            "protected_environments",
        }
    }

    data_kwargs = {
        k: v for k, v in kwargs.items()
        if k in {
            "block_external_uploads",
            "block_task_data_leaks",
            "block_solution_sharing",
        }
    }

    if repository_rules:
        create_benchmark_repository_safety_rules(enforcer, **repository_kwargs)

    if resource_rules:
        create_benchmark_resource_safety_rules(enforcer, **resource_kwargs)

    if test_rules:
        create_benchmark_test_safety_rules(enforcer, **test_kwargs)

    if data_rules:
        create_benchmark_data_safety_rules(enforcer, **data_kwargs)


__all__ = [
    "create_benchmark_repository_safety_rules",
    "create_benchmark_resource_safety_rules",
    "create_benchmark_test_safety_rules",
    "create_benchmark_data_safety_rules",
    "create_all_benchmark_safety_rules",
]
