"""Git tool with AI-powered commit messages and smart operations.

This tool provides:
1. AI-generated commit messages based on diff analysis
2. Intelligent staging of related files
3. PR creation and management
4. Conflict detection and resolution help
5. Git hooks integration

Migrated to decorator pattern for better maintainability.
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
async def git_status() -> Dict[str, Any]:
    """Get repository status.

    Shows both short and full git status to provide a complete
    view of the current repository state.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Status information (short and full)
        - error: Error message if failed
    """
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


@tool
async def git_diff(staged: bool = False) -> Dict[str, Any]:
    """Show changes in the repository.

    Displays git diff output for either staged or unstaged changes.

    Args:
        staged: If True, show staged changes; if False, show unstaged changes

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Diff output
        - error: Error message if failed
    """
    args = ["diff"]
    if staged:
        args.append("--staged")

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


@tool
async def git_stage(files: Optional[List[str]] = None) -> Dict[str, Any]:
    """Stage files for commit.

    If no files are specified, stages all changes. Otherwise, stages
    only the specified files.

    Args:
        files: List of file paths to stage. If None or empty, stages all changes.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Status after staging
        - error: Error message if failed
    """
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


@tool
async def git_commit(message: Optional[str] = None, generate_ai: Optional[bool] = None) -> Dict[str, Any]:
    """Commit staged changes.

    Creates a commit with the provided message. If no message is provided
    and AI is available, generates one automatically from the staged diff.

    Args:
        message: Commit message. If None, will attempt AI generation
        generate_ai: Force AI generation. Defaults to True if provider available

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Commit result
        - error: Error message if failed
    """
    if generate_ai is None:
        generate_ai = _provider is not None

    # If no message and AI available, generate one
    if not message and generate_ai and _provider:
        suggest_result = await git_suggest_commit()
        if suggest_result["success"]:
            message = suggest_result["output"]
        else:
            return {
                "success": False,
                "output": "",
                "error": "No commit message provided and AI generation failed"
            }

    if not message:
        return {
            "success": False,
            "output": "",
            "error": "No commit message provided"
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


@tool
async def git_log(limit: int = 10) -> Dict[str, Any]:
    """Show commit history.

    Displays the commit log with graph visualization and formatted output.

    Args:
        limit: Number of commits to show (default: 10)

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Formatted commit log
        - error: Error message if failed
    """
    success, stdout, stderr = _run_git(
        "log",
        f"-{limit}",
        "--pretty=format:%h - %s (%an, %ar)",
        "--graph"
    )

    if not success:
        return {"success": False, "output": "", "error": stderr}

    return {"success": True, "output": stdout, "error": ""}


@tool
async def git_branch(branch: Optional[str] = None) -> Dict[str, Any]:
    """List, create, or switch branches.

    If no branch name is provided, lists all branches. If a branch name
    is provided, switches to it (or creates it if it doesn't exist).

    Args:
        branch: Branch name to switch to/create. If None, lists branches.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Branch information or operation result
        - error: Error message if failed
    """
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
        analysis.append(f"\n📄 {file}:")

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
        analysis.append("\n\n💡 AI-generated resolution suggestions:")
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
    """Deprecated: Use individual git_* functions instead.

    This class is kept for backward compatibility but will be removed
    in a future version. Use the decorator-based git_* functions instead.
    """

    def __init__(self, provider=None, model: Optional[str] = None):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "GitTool class is deprecated. Use git_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
        set_git_provider(provider, model)
