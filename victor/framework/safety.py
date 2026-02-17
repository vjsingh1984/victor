"""Framework-level safety rule factories.

Reusable safety rule factories for git operations and file protection.
These are domain-agnostic and can be used by any vertical that works
with git or the filesystem.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.framework.safety import create_git_safety_rules, create_file_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_git_safety_rules(enforcer)
    create_file_safety_rules(enforcer)

    allowed, reason = enforcer.check_operation("git push --force origin main")
    if not allowed:
        print(f"Blocked: {reason}")
"""

from __future__ import annotations

from victor.framework.config import SafetyEnforcer, SafetyRule, SafetyLevel


def create_git_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    protected_branches: list[str] | None = None,
) -> None:
    """Register git-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_force_push: Block force push to protected branches
        block_main_push: Block direct push to main/master
        require_tests_before_commit: Require tests to pass before commit
        protected_branches: List of protected branch names
            (default: ["main", "master", "develop"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(
            enforcer,
            block_force_push=True,
            block_main_push=True,
            protected_branches=["main", "master", "production"]
        )
    """
    protected = protected_branches or ["main", "master", "develop"]

    if block_force_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_force_push",
                description="Block force push to protected branches",
                check_fn=lambda op: "git push" in op
                and "--force" in op
                and any(branch in op for branch in protected),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_main_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_main_push",
                description="Block direct push to main/master branches (use PRs)",
                check_fn=lambda op: "git push" in op
                and any(
                    f" origin {branch}" in op or f" upstream {branch}" in op
                    for branch in protected[:2]
                ),  # main, master only
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if require_tests_before_commit:
        enforcer.add_rule(
            SafetyRule(
                name="git_require_tests_before_commit",
                description="Require tests to pass before committing",
                check_fn=lambda op: "git commit" in op and "--no-verify" not in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_file_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_commands: bool = True,
    protected_patterns: list[str] | None = None,
) -> None:
    """Register file operation safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_commands: Block commands like rm -rf, git clean -fdx
        protected_patterns: Glob patterns for protected files
            (default: [".env", "secrets", "credential", "password"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_safety_rules(
            enforcer,
            block_destructive_commands=True,
            protected_patterns=["**/.env", "**/secrets*", "**/config/prod/*"]
        )
    """
    protected = protected_patterns or [".env", "secrets", "credential", "password"]

    if block_destructive_commands:
        enforcer.add_rule(
            SafetyRule(
                name="file_block_destructive",
                description="Block destructive file commands (rm -rf, git clean -fdx)",
                check_fn=lambda op: any(
                    cmd in op
                    for cmd in [
                        "rm -rf /",
                        "rm -rf /*",
                        "git clean -fdx",
                        "del /q /s",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

        enforcer.add_rule(
            SafetyRule(
                name="file_block_protected_modification",
                description="Block modification of protected files (.env, secrets, credentials)",
                check_fn=lambda op: any(
                    pattern in op.lower() for pattern in [p.lower() for p in protected]
                )
                and any(cmd in op for cmd in ["write(", "write_file(", "edit("]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )
