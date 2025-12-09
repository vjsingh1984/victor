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

"""Git tool with AI-powered commit messages and smart operations.

This tool provides:
1. Unified git operations (status, diff, stage, commit, log, branch)
2. AI-generated commit messages based on diff analysis
3. PR creation and management
4. Conflict detection and resolution help
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from victor.config.timeouts import ProcessTimeouts
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority, ToolConfig
from victor.tools.decorators import tool

# Global state for AI provider (DEPRECATED: use ToolConfig via context instead)
_provider = None
_model: Optional[str] = None


def set_git_provider(provider, model: Optional[str] = None) -> None:
    """Set the global provider and model for git AI operations.

    DEPRECATED: Use ToolConfig via context instead. This function is kept
    for backward compatibility but will be removed in a future version.

    Args:
        provider: LLM provider for AI-generated messages
        model: Model to use for message generation
    """
    global _provider, _model
    _provider = provider
    _model = model


def _get_provider_and_model(context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Optional[str]]:
    """Get provider and model from context or fall back to globals.

    Args:
        context: Tool execution context

    Returns:
        Tuple of (provider, model) - provider may be None if not configured
    """
    # First try to get from context (preferred method)
    config = ToolConfig.from_context(context) if context else None
    if config and config.provider:
        return config.provider, config.model

    # Fall back to global state (deprecated)
    return _provider, _model


def _run_git(*args: str, env_overrides: Optional[Dict[str, str]] = None) -> Tuple[bool, str, str]:
    """Run git command with optional environment variable overrides.

    Args:
        *args: Git command arguments
        env_overrides: Optional dictionary of environment variables to set for this command.
            Useful for setting GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL, etc.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        # Prepare environment
        cmd_env = None
        if env_overrides:
            cmd_env = os.environ.copy()
            cmd_env.update(env_overrides)

        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=ProcessTimeouts.GIT,
            env=cmd_env,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Git command timed out"
    except Exception as e:
        return False, "", str(e)


@tool(
    category="git",
    priority=Priority.HIGH,  # Frequently used for version control
    access_mode=AccessMode.MIXED,  # Reads repo state and writes commits
    danger_level=DangerLevel.MEDIUM,  # Repository modifications
    # Registry-driven metadata for tool selection and cache management
    stages=["execution", "verification", "completion"],  # Conversation stages where relevant
    task_types=["action", "analysis"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.MIXED,  # Can both read and write
    progress_params=["operation", "files", "branch"],  # Params indicating different operations
    keywords=[
        "git",
        "commit",
        "stage",
        "diff",
        "status",
        "branch",
        "log",
        "version control",
        "repository",
        "author",
        "changes",
        "history",
    ],
    use_cases=[
        "checking repository status",
        "viewing file changes and diffs",
        "staging files for commit",
        "committing changes with custom authorship",
        "viewing commit history",
        "creating and switching branches",
    ],
    examples=[
        "show git status",
        "what files have changed",
        "stage all changes",
        "commit with message 'fix bug'",
        "commit as John Doe john@example.com",
        "show last 5 commits",
        "create new branch feature/auth",
    ],
    priority_hints=[
        "Use for all git version control operations",
        "Supports custom author name and email for commits",
        "Can stage individual files or all changes",
    ],
)
async def git(
    operation: str,
    files: Optional[List[str]] = None,
    message: Optional[str] = None,
    branch: Optional[str] = None,
    staged: bool = False,
    limit: int = 10,
    options: Optional[Dict[str, Any]] = None,
    author_name: Optional[str] = None,
    author_email: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified git operations: status, diff, stage, commit, log, branch.

    Operations: status, diff (staged=True for staged), stage (files or all),
    commit (message required), log (limit), branch (list/create/switch).
    Supports custom author_name/author_email for commits.
    """
    if not operation:
        return {"success": False, "error": "Missing required parameter: operation"}

    if options is None:
        options = {}

    # Status operation
    if operation == "status":
        success, stdout, stderr = _run_git("status", "--short", "--branch")

        if not success:
            return {"success": False, "output": "", "error": stderr}

        # Also get longer status for summary
        _, long_status, _ = _run_git("status")

        return {
            "success": True,
            "output": f"Short status:\n{stdout}\n\nFull status:\n{long_status}",
            "error": "",
        }

    # Diff operation
    elif operation == "diff":
        args = ["diff"]
        if staged:
            args.append("--staged")

        if files:
            args.extend(["--"] + files)

        success, stdout, stderr = _run_git(*args)

        if not success:
            return {"success": False, "output": "", "error": stderr}

        if not stdout:
            return {
                "success": True,
                "output": "No changes to show" if not staged else "No staged changes",
                "error": "",
            }

        return {"success": True, "output": stdout, "error": ""}

    # Stage operation
    elif operation == "stage":
        if not files:
            # Stage all changes
            success, stdout, stderr = _run_git("add", ".")
        else:
            # Stage specific files
            success, stdout, stderr = _run_git("add", *files)

        if not success:
            return {"success": False, "output": "", "error": stderr}

        # Get updated status
        _, status, _ = _run_git("status", "--short")

        return {
            "success": True,
            "output": f"Files staged successfully\n\nStatus:\n{status}",
            "error": "",
        }

    # Commit operation
    elif operation == "commit":
        if not message:
            return {
                "success": False,
                "output": "",
                "error": "Commit message required. Use message parameter.",
            }

        # Build environment overrides for custom authorship
        env_overrides: Dict[str, str] = {}
        if author_name:
            env_overrides["GIT_AUTHOR_NAME"] = author_name
            env_overrides["GIT_COMMITTER_NAME"] = author_name
        if author_email:
            env_overrides["GIT_AUTHOR_EMAIL"] = author_email
            env_overrides["GIT_COMMITTER_EMAIL"] = author_email

        # Commit with message and optional author override
        success, stdout, stderr = _run_git(
            "commit", "-m", message, env_overrides=env_overrides if env_overrides else None
        )

        if not success:
            return {"success": False, "output": "", "error": stderr}

        author_info = ""
        if author_name or author_email:
            author_info = f" (as {author_name or 'default'} <{author_email or 'default'}>)"

        return {
            "success": True,
            "output": f"Committed successfully{author_info}:\n{stdout}",
            "error": "",
        }

    # Log operation
    elif operation == "log":
        success, stdout, stderr = _run_git(
            "log", f"-{limit}", "--pretty=format:%h - %s (%an, %ar)", "--graph"
        )

        if not success:
            return {"success": False, "output": "", "error": stderr}

        return {"success": True, "output": stdout, "error": ""}

    # Branch operation
    elif operation == "branch":
        if not branch:
            # List branches
            success, stdout, stderr = _run_git("branch", "-a")
        else:
            # Create or switch to branch
            # Try to switch first
            success, stdout, stderr = _run_git("checkout", branch)

            if not success and "did not match" in stderr:
                # Branch doesn't exist, create it
                success, stdout, stderr = _run_git("checkout", "-b", branch)

        if not success:
            return {"success": False, "output": "", "error": stderr}

        return {"success": True, "output": stdout, "error": ""}

    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: status, diff, stage, commit, log, branch",
        }


@tool(
    category="git",
    priority=Priority.MEDIUM,  # Task-specific AI operation
    access_mode=AccessMode.READONLY,  # Only reads diff, doesn't commit
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=["commit message", "ai", "generate", "suggest", "conventional commit"],
    mandatory_keywords=["generate commit message", "suggest commit"],  # Force inclusion
    task_types=["generation", "git"],  # Classification-aware selection
    use_cases=["generating commit messages", "creating conventional commits"],
    examples=["suggest a commit message", "generate commit message for staged changes"],
)
async def commit_msg(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate an AI-powered commit message from staged changes.

    Analyzes the staged diff and generates a conventional commit message
    using the configured LLM provider. The generated message follows the
    conventional commit format: type(scope): subject

    Types used: feat, fix, docs, style, refactor, test, chore

    Args:
        context: Tool execution context (injected by decorator, do not pass manually)

    Returns:
        Dictionary containing:
        - success: bool - Whether message generation succeeded
        - output: str - The generated commit message
        - error: str - Error message if failed (empty on success)

    Note:
        Requires staged changes (git add) before calling this tool.
        Uses the configured LLM provider for message generation.
    """
    provider, model = _get_provider_and_model(context)
    if not provider:
        return {
            "success": False,
            "output": "",
            "error": "No LLM provider available for AI generation",
        }

    # Get staged diff
    success, diff, stderr = _run_git("diff", "--staged")

    if not success:
        return {"success": False, "output": "", "error": stderr}

    if not diff:
        return {"success": False, "output": "", "error": "No staged changes to analyze"}

    # Get list of changed files
    _, files, _ = _run_git("diff", "--staged", "--name-only")

    # Prepare prompt for LLM
    prompt = f"""Analyze these git changes and generate a concise, conventional commit message.

Changed files:
{files}

Diff:
{diff[:5000]}  # Limit diff size

Follow conventional commit format:
<type>(<scope>): <subject>

Types: feat, fix, docs, style, refactor, test, chore
Subject: Present tense, lowercase, no period, max 50 chars

Generate ONLY the commit message, nothing else."""

    try:
        # Call LLM
        from victor.providers.base import Message

        response = await provider.complete(
            model=model or "default",
            messages=[Message(role="user", content=prompt)],
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=200,
        )

        message = response.content.strip()

        # Clean up message
        message = message.replace('"', "").replace("'", "")
        if message.startswith("Commit message:"):
            message = message.replace("Commit message:", "").strip()

        return {"success": True, "output": message, "error": ""}

    except Exception as e:
        return {"success": False, "output": "", "error": f"AI generation failed: {str(e)}"}


@tool(
    category="git",
    priority=Priority.MEDIUM,  # Task-specific GitHub operation
    access_mode=AccessMode.NETWORK,  # Pushes to remote, creates PR via API
    danger_level=DangerLevel.LOW,  # PRs can be closed/reverted
    keywords=["pull request", "pr", "github", "merge request", "create pr"],
    mandatory_keywords=["create pr", "pull request", "open pr"],  # Force inclusion
    task_types=["action", "git"],  # Classification-aware selection
    use_cases=["creating pull requests", "opening PRs on GitHub"],
    examples=[
        "create a pull request",
        "open PR to main branch",
        "create pr with title 'Add feature'",
    ],
)
async def pr(
    pr_title: Optional[str] = None,
    pr_description: Optional[str] = None,
    base_branch: str = "main",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a GitHub pull request with auto-generated or custom content.

    Creates a pull request using GitHub CLI (gh). Automatically pushes the
    current branch to origin if needed. If title or description are not
    provided and an AI provider is configured, generates them from the
    commit history and diff.

    Args:
        pr_title: Pull request title. If None and AI is available, generates
            from commit messages. Otherwise defaults to "Merge {branch} into {base}".
            Example: "Add user authentication feature"
        pr_description: Pull request body/description. If None and AI is available,
            generates from diff. Otherwise uses a default description.
            Example: "This PR adds OAuth2 authentication support..."
        base_branch: Target branch to merge into. Default: "main".
            Example: "develop" or "release/v2.0"
        context: Tool execution context (injected by decorator, do not pass manually)

    Returns:
        Dictionary containing:
        - success: bool - Whether PR creation succeeded
        - output: str - PR URL and creation confirmation
        - error: str - Error message if failed (empty on success)

    Note:
        Requires GitHub CLI (gh) to be installed and authenticated.
        Install with: brew install gh (macOS) or see https://cli.github.com/
    """
    provider, model = _get_provider_and_model(context)

    # Get current branch
    success, current_branch, stderr = _run_git("branch", "--show-current")
    if not success:
        return {"success": False, "output": "", "error": stderr}

    current_branch = current_branch.strip()

    # If no title/description and AI available, generate them
    if (not pr_title or not pr_description) and provider:
        # Get diff from base branch
        success, diff, stderr = _run_git("diff", f"{base_branch}...HEAD")

        if success and diff:
            # Get commit log
            _, log, _ = _run_git("log", f"{base_branch}..HEAD", "--pretty=format:- %s")

            # Generate PR content
            prompt = f"""Generate a pull request title and description for these changes.

Current branch: {current_branch}
Base branch: {base_branch}

Recent commits:
{log}

Diff summary:
{diff[:3000]}

Generate:
1. A concise title (max 60 chars)
2. A detailed description with:
   - Summary of changes
   - Why these changes were made
   - Any breaking changes or migration notes

Format:
TITLE: <title>
DESCRIPTION:
<description>
"""

            try:
                from victor.providers.base import Message

                response = await provider.complete(
                    model=model or "default",
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.5,
                    max_tokens=500,
                )

                content = response.content.strip()

                # Parse response
                if "TITLE:" in content and "DESCRIPTION:" in content:
                    parts = content.split("DESCRIPTION:")
                    title_part = parts[0].replace("TITLE:", "").strip()
                    desc_part = parts[1].strip()

                    if not pr_title:
                        pr_title = title_part
                    if not pr_description:
                        pr_description = desc_part

            except Exception:
                # AI generation failed, continue with manual
                pass

    # Use defaults if still not provided
    if not pr_title:
        pr_title = f"Merge {current_branch} into {base_branch}"

    if not pr_description:
        pr_description = "Automatically generated PR description"

    # Push branch first
    success, stdout, stderr = _run_git("push", "--set-upstream", "origin", current_branch)

    if not success:
        return {"success": False, "output": "", "error": f"Failed to push branch: {stderr}"}

    # Create PR with gh CLI
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                base_branch,
                "--head",
                current_branch,
                "--title",
                pr_title,
                "--body",
                pr_description,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return {
                "success": True,
                "output": f"PR created successfully!\n\n{result.stdout}",
                "error": "",
            }
        else:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to create PR: {result.stderr}",
            }

    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": "GitHub CLI (gh) not found. Install with: brew install gh",
        }


@tool(
    category="git",
    priority=Priority.MEDIUM,  # Context-specific conflict resolution
    access_mode=AccessMode.READONLY,  # Only analyzes, doesn't modify
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=["merge conflict", "conflict", "resolve", "rebase", "merge"],
    use_cases=["analyzing merge conflicts", "resolving git conflicts", "conflict resolution help"],
    examples=["analyze conflicts", "show merge conflicts", "help resolve conflicts"],
)
async def conflicts(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze merge conflicts and provide resolution guidance.

    Detects files with merge conflicts (marked as UU in git status) and provides
    detailed information about each conflict, including previews of conflict
    markers and step-by-step resolution instructions.

    Args:
        context: Tool execution context (injected by decorator, do not pass manually)

    Returns:
        Dictionary containing:
        - success: bool - Whether analysis succeeded
        - output: str - Conflict analysis with file list, conflict counts,
            conflict previews, and resolution steps
        - error: str - Error message if failed (empty on success)

    Note:
        Call this after a failed merge or rebase to understand what needs
        to be resolved. After manual resolution, stage files with git add
        and continue with git merge --continue or git rebase --continue.
    """
    # Get list of conflicted files
    success, status, stderr = _run_git("status", "--short")

    if not success:
        return {"success": False, "output": "", "error": stderr}

    # Find conflicted files (marked with UU)
    conflicted = [line.split()[-1] for line in status.split("\n") if line.startswith("UU")]

    if not conflicted:
        return {"success": True, "output": "No merge conflicts detected", "error": ""}

    # Analyze each conflicted file
    analysis = [f"Found {len(conflicted)} conflicted file(s):\n"]

    for file in conflicted:
        analysis.append(f"\n{file}:")

        # Read file to show conflict markers
        try:
            with open(file) as f:
                content = f.read()

            # Count conflict markers
            conflict_count = content.count("<<<<<<< ")

            analysis.append(f"   {conflict_count} conflict(s) in file")

            # Show first conflict context
            if "<<<<<<< " in content:
                start = content.find("<<<<<<< ")
                end = content.find(">>>>>>> ", start)
                if end != -1:
                    conflict_section = content[start : end + 50]
                    analysis.append(f"   First conflict preview:\n   {conflict_section[:200]}...")

        except Exception as e:
            analysis.append(f"   Error reading file: {e}")

    # If AI available, get resolution suggestions
    provider, _ = _get_provider_and_model(context)
    if provider:
        analysis.append("\n\nAI-generated resolution suggestions:")
        analysis.append("   (Using LLM to analyze conflicts...)")
        # TODO: Implement AI conflict resolution suggestions

    analysis.append("\n\nTo resolve:")
    analysis.append("1. Edit conflicted files manually")
    analysis.append("2. Remove conflict markers (<<<<<<, =======, >>>>>>>)")
    analysis.append("3. Stage resolved files: git add <file>")
    analysis.append("4. Continue: git merge --continue or git rebase --continue")

    return {"success": True, "output": "\n".join(analysis), "error": ""}


