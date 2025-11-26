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

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.decorators import tool

# Global state for AI provider
_provider = None
_model: Optional[str] = None


def set_git_provider(provider, model: Optional[str] = None) -> None:
    """Set the global provider and model for git AI operations.

    Args:
        provider: LLM provider for AI-generated messages
        model: Model to use for message generation
    """
    global _provider, _model
    _provider = provider
    _model = model


def _run_git(*args: str) -> Tuple[bool, str, str]:
    """Run git command.

    Args:
        *args: Git command arguments

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Git command timed out"
    except Exception as e:
        return False, "", str(e)


@tool
async def git(
    operation: str,
    files: Optional[List[str]] = None,
    message: Optional[str] = None,
    branch: Optional[str] = None,
    staged: bool = False,
    limit: int = 10,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Unified git operations tool.

    Performs common git operations (status, diff, stage, commit, log, branch)
    through a single unified interface. Consolidates basic git functionality.

    Args:
        operation: Git operation to perform. Options: "status", "diff", "stage",
            "commit", "log", "branch".
        files: List of file paths (for stage operation).
        message: Commit message (for commit operation).
        branch: Branch name (for branch operation).
        staged: Show staged changes for diff (default: False).
        limit: Number of commits for log (default: 10).
        options: Additional operation-specific options.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Operation result
        - error: Error message if failed

    Examples:
        # Show repository status
        git(operation="status")

        # Show unstaged changes
        git(operation="diff")

        # Show staged changes
        git(operation="diff", staged=True)

        # Stage specific files
        git(operation="stage", files=["src/main.py", "tests/test.py"])

        # Stage all changes
        git(operation="stage")

        # Commit with message
        git(operation="commit", message="Fix authentication bug")

        # Show commit history
        git(operation="log", limit=20)

        # List branches
        git(operation="branch")

        # Create/switch to branch
        git(operation="branch", branch="feature/new-feature")
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
            "error": ""
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
                "error": ""
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
            "error": ""
        }

    # Commit operation
    elif operation == "commit":
        if not message:
            return {
                "success": False,
                "output": "",
                "error": "Commit message required. Use message parameter."
            }

        # Commit with message
        success, stdout, stderr = _run_git("commit", "-m", message)

        if not success:
            return {"success": False, "output": "", "error": stderr}

        return {
            "success": True,
            "output": f"Committed successfully:\n{stdout}",
            "error": ""
        }

    # Log operation
    elif operation == "log":
        success, stdout, stderr = _run_git(
            "log",
            f"-{limit}",
            "--pretty=format:%h - %s (%an, %ar)",
            "--graph"
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
            "error": f"Unknown operation: {operation}. Valid operations: status, diff, stage, commit, log, branch"
        }


@tool
async def git_suggest_commit() -> Dict[str, Any]:
    """Generate AI commit message from staged changes.

    Analyzes the staged diff and generates a conventional commit message
    using the configured LLM provider.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Generated commit message
        - error: Error message if failed
    """
    if not _provider:
        return {
            "success": False,
            "output": "",
            "error": "No LLM provider available for AI generation"
        }

    # Get staged diff
    success, diff, stderr = _run_git("diff", "--staged")

    if not success:
        return {"success": False, "output": "", "error": stderr}

    if not diff:
        return {
            "success": False,
            "output": "",
            "error": "No staged changes to analyze"
        }

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

        response = await _provider.complete(
            model=_model or "default",
            messages=[Message(role="user", content=prompt)],
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=200
        )

        message = response.content.strip()

        # Clean up message
        message = message.replace('"', '').replace("'", "")
        if message.startswith("Commit message:"):
            message = message.replace("Commit message:", "").strip()

        return {
            "success": True,
            "output": message,
            "error": ""
        }

    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"AI generation failed: {str(e)}"
        }


@tool
async def git_create_pr(
    pr_title: Optional[str] = None,
    pr_description: Optional[str] = None,
    base_branch: str = "main"
) -> Dict[str, Any]:
    """Create a pull request with auto-generated content.

    Creates a pull request using GitHub CLI. If title or description
    are not provided and AI is available, generates them from the commits.

    Args:
        pr_title: PR title. If None, auto-generated
        pr_description: PR description. If None, auto-generated
        base_branch: Base branch for PR (default: "main")

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: PR creation result with URL
        - error: Error message if failed
    """
    # Get current branch
    success, current_branch, stderr = _run_git("branch", "--show-current")
    if not success:
        return {"success": False, "output": "", "error": stderr}

    current_branch = current_branch.strip()

    # If no title/description and AI available, generate them
    if (not pr_title or not pr_description) and _provider:
        # Get diff from base branch
        success, diff, stderr = _run_git("diff", f"{base_branch}...HEAD")

        if success and diff:
            # Get commit log
            _, log, _ = _run_git(
                "log",
                f"{base_branch}..HEAD",
                "--pretty=format:- %s"
            )

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

                response = await _provider.complete(
                    model=_model or "default",
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.5,
                    max_tokens=500
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
    success, stdout, stderr = _run_git(
        "push", "--set-upstream", "origin", current_branch
    )

    if not success:
        return {
            "success": False,
            "output": "",
            "error": f"Failed to push branch: {stderr}"
        }

    # Create PR with gh CLI
    try:
        result = subprocess.run(
            [
                "gh", "pr", "create",
                "--base", base_branch,
                "--head", current_branch,
                "--title", pr_title,
                "--body", pr_description
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {
                "success": True,
                "output": f"PR created successfully!\n\n{result.stdout}",
                "error": ""
            }
        else:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to create PR: {result.stderr}"
            }

    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": "GitHub CLI (gh) not found. Install with: brew install gh"
        }


@tool
async def git_analyze_conflicts() -> Dict[str, Any]:
    """Analyze merge conflicts and provide resolution guidance.

    Detects conflicted files and provides information about the conflicts,
    including conflict markers and resolution steps.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Conflict analysis and resolution guidance
        - error: Error message if failed
    """
    # Get list of conflicted files
    success, status, stderr = _run_git("status", "--short")

    if not success:
        return {"success": False, "output": "", "error": stderr}

    # Find conflicted files (marked with UU)
    conflicted = [
        line.split()[-1]
        for line in status.split("\n")
        if line.startswith("UU")
    ]

    if not conflicted:
        return {
            "success": True,
            "output": "No merge conflicts detected",
            "error": ""
        }

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
                    conflict_section = content[start:end+50]
                    analysis.append(f"   First conflict preview:\n   {conflict_section[:200]}...")

        except Exception as e:
            analysis.append(f"   Error reading file: {e}")

    # If AI available, get resolution suggestions
    if _provider:
        analysis.append("\n\nAI-generated resolution suggestions:")
        analysis.append("   (Using LLM to analyze conflicts...)")
        # TODO: Implement AI conflict resolution suggestions

    analysis.append("\n\nTo resolve:")
    analysis.append("1. Edit conflicted files manually")
    analysis.append("2. Remove conflict markers (<<<<<<, =======, >>>>>>>)")
    analysis.append("3. Stage resolved files: git add <file>")
    analysis.append("4. Continue: git merge --continue or git rebase --continue")

    return {
        "success": True,
        "output": "\n".join(analysis),
        "error": ""
    }


# Keep class for backward compatibility
class GitTool:
    """Deprecated: Use git and git_* functions instead.

    This class is kept for backward compatibility but will be removed
    in a future version. Use the decorator-based git and git_* functions instead.
    """

    def __init__(self, provider=None, model: Optional[str] = None):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "GitTool class is deprecated. Use git and git_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        set_git_provider(provider, model)
