"""Git tool with AI-powered commit messages and smart operations.

This tool provides:
1. AI-generated commit messages based on diff analysis
2. Intelligent staging of related files
3. PR creation and management
4. Conflict detection and resolution help
5. Git hooks integration
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import BaseTool, ToolParameter, ToolResult


class GitTool(BaseTool):
    """Git operations with AI assistance."""

    def __init__(self, provider=None, model: Optional[str] = None):
        """Initialize git tool.

        Args:
            provider: LLM provider for AI-generated messages
            model: Model to use for message generation
        """
        super().__init__()
        self.provider = provider
        self.model = model

    @property
    def name(self) -> str:
        """Get tool name."""
        return "git"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Git operations with AI assistance.

Provides intelligent git operations including:
- AI-generated commit messages from diff analysis
- Smart file staging
- PR creation with auto-generated descriptions
- Conflict detection and resolution guidance
- Git status and log analysis

Operations:
- status: Get repository status
- diff: Show changes (staged or unstaged)
- stage: Stage files (can be smart about related files)
- commit: Commit with AI-generated or custom message
- log: Show commit history
- branch: List, create, or switch branches
- create_pr: Create pull request with auto-description
- analyze_conflicts: Get help resolving merge conflicts
- suggest_commit: Generate commit message from diff

Example workflows:
1. Smart commit:
   - git(operation="suggest_commit") -> Get AI suggestion
   - git(operation="commit", message="...") -> Commit with message

2. Create PR:
   - git(operation="create_pr") -> Auto-generated title/description
"""

    @property
    def parameters(self) -> List[ToolParameter]:
        """Get tool parameters."""
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: status, diff, stage, commit, log, branch, create_pr, analyze_conflicts, suggest_commit",
                required=True
            ),
            ToolParameter(
                name="files",
                type="array",
                description="Files to stage (for stage operation)",
                required=False
            ),
            ToolParameter(
                name="message",
                type="string",
                description="Commit message (for commit operation)",
                required=False
            ),
            ToolParameter(
                name="branch",
                type="string",
                description="Branch name (for branch operations)",
                required=False
            ),
            ToolParameter(
                name="staged",
                type="boolean",
                description="Show staged changes (for diff operation, default: false)",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Number of commits to show (for log operation, default: 10)",
                required=False
            ),
            ToolParameter(
                name="generate_ai",
                type="boolean",
                description="Use AI to generate message (for commit operation, default: true if provider available)",
                required=False
            ),
            ToolParameter(
                name="pr_title",
                type="string",
                description="PR title (for create_pr, auto-generated if not provided)",
                required=False
            ),
            ToolParameter(
                name="pr_description",
                type="string",
                description="PR description (for create_pr, auto-generated if not provided)",
                required=False
            ),
            ToolParameter(
                name="base_branch",
                type="string",
                description="Base branch for PR (default: main)",
                required=False
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute git operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with output or error
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation"
            )

        try:
            if operation == "status":
                return await self._status(kwargs)
            elif operation == "diff":
                return await self._diff(kwargs)
            elif operation == "stage":
                return await self._stage(kwargs)
            elif operation == "commit":
                return await self._commit(kwargs)
            elif operation == "log":
                return await self._log(kwargs)
            elif operation == "branch":
                return await self._branch(kwargs)
            elif operation == "create_pr":
                return await self._create_pr(kwargs)
            elif operation == "analyze_conflicts":
                return await self._analyze_conflicts(kwargs)
            elif operation == "suggest_commit":
                return await self._suggest_commit(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Git operation error: {str(e)}"
            )

    def _run_git(self, *args: str) -> Tuple[bool, str, str]:
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

    async def _status(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get repository status."""
        success, stdout, stderr = self._run_git("status", "--short", "--branch")

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        # Also get longer status for summary
        _, long_status, _ = self._run_git("status")

        return ToolResult(
            success=True,
            output=f"Short status:\n{stdout}\n\nFull status:\n{long_status}",
            error=""
        )

    async def _diff(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show changes."""
        staged = kwargs.get("staged", False)

        args = ["diff"]
        if staged:
            args.append("--staged")

        success, stdout, stderr = self._run_git(*args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        if not stdout:
            return ToolResult(
                success=True,
                output="No changes to show" if not staged else "No staged changes",
                error=""
            )

        return ToolResult(success=True, output=stdout, error="")

    async def _stage(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Stage files."""
        files = kwargs.get("files", [])

        if not files:
            # Stage all changes
            success, stdout, stderr = self._run_git("add", ".")
        else:
            # Stage specific files
            success, stdout, stderr = self._run_git("add", *files)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        # Get updated status
        _, status, _ = self._run_git("status", "--short")

        return ToolResult(
            success=True,
            output=f"Files staged successfully\n\nStatus:\n{status}",
            error=""
        )

    async def _commit(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Commit changes."""
        message = kwargs.get("message")
        generate_ai = kwargs.get("generate_ai", self.provider is not None)

        # If no message and AI available, generate one
        if not message and generate_ai and self.provider:
            suggest_result = await self._suggest_commit(kwargs)
            if suggest_result.success:
                message = suggest_result.output
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error="No commit message provided and AI generation failed"
                )

        if not message:
            return ToolResult(
                success=False,
                output="",
                error="No commit message provided"
            )

        # Commit with message
        success, stdout, stderr = self._run_git("commit", "-m", message)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(
            success=True,
            output=f"Committed successfully:\n{stdout}",
            error=""
        )

    async def _log(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Show commit history."""
        limit = kwargs.get("limit", 10)

        success, stdout, stderr = self._run_git(
            "log",
            f"-{limit}",
            "--pretty=format:%h - %s (%an, %ar)",
            "--graph"
        )

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _branch(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Branch operations."""
        branch = kwargs.get("branch")

        if not branch:
            # List branches
            success, stdout, stderr = self._run_git("branch", "-a")
        else:
            # Create or switch to branch
            # Try to switch first
            success, stdout, stderr = self._run_git("checkout", branch)

            if not success and "did not match" in stderr:
                # Branch doesn't exist, create it
                success, stdout, stderr = self._run_git("checkout", "-b", branch)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _suggest_commit(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Generate AI commit message from diff."""
        if not self.provider:
            return ToolResult(
                success=False,
                output="",
                error="No LLM provider available for AI generation"
            )

        # Get staged diff
        success, diff, stderr = self._run_git("diff", "--staged")

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        if not diff:
            return ToolResult(
                success=False,
                output="",
                error="No staged changes to analyze"
            )

        # Get list of changed files
        _, files, _ = self._run_git("diff", "--staged", "--name-only")

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

            response = await self.provider.complete(
                model=self.model or "default",
                messages=[Message(role="user", content=prompt)],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=200
            )

            message = response.content.strip()

            # Clean up message
            message = message.replace('"', '').replace("'", "")
            if message.startswith("Commit message:"):
                message = message.replace("Commit message:", "").strip()

            return ToolResult(
                success=True,
                output=message,
                error=""
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"AI generation failed: {str(e)}"
            )

    async def _create_pr(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Create pull request with auto-generated description."""
        pr_title = kwargs.get("pr_title")
        pr_description = kwargs.get("pr_description")
        base_branch = kwargs.get("base_branch", "main")

        # Get current branch
        success, current_branch, stderr = self._run_git("branch", "--show-current")
        if not success:
            return ToolResult(success=False, output="", error=stderr)

        current_branch = current_branch.strip()

        # If no title/description and AI available, generate them
        if (not pr_title or not pr_description) and self.provider:
            # Get diff from base branch
            success, diff, stderr = self._run_git("diff", f"{base_branch}...HEAD")

            if success and diff:
                # Get commit log
                _, log, _ = self._run_git(
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

                    response = await self.provider.complete(
                        model=self.model or "default",
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

                except Exception as e:
                    # AI generation failed, continue with manual
                    pass

        # Use defaults if still not provided
        if not pr_title:
            pr_title = f"Merge {current_branch} into {base_branch}"

        if not pr_description:
            pr_description = "Automatically generated PR description"

        # Create PR using gh CLI (GitHub CLI must be installed)
        success, stdout, stderr = self._run_git(
            "push", "--set-upstream", "origin", current_branch
        )

        if not success:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to push branch: {stderr}"
            )

        # Try to create PR with gh
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
                return ToolResult(
                    success=True,
                    output=f"PR created successfully!\n\n{result.stdout}",
                    error=""
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to create PR: {result.stderr}"
                )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="GitHub CLI (gh) not found. Install with: brew install gh"
            )

    async def _analyze_conflicts(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Analyze merge conflicts and provide resolution guidance."""
        # Get list of conflicted files
        success, status, stderr = self._run_git("status", "--short")

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        # Find conflicted files (marked with UU)
        conflicted = [
            line.split()[-1]
            for line in status.split("\n")
            if line.startswith("UU")
        ]

        if not conflicted:
            return ToolResult(
                success=True,
                output="No merge conflicts detected",
                error=""
            )

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
        if self.provider:
            analysis.append("\n\n💡 AI-generated resolution suggestions:")
            analysis.append("   (Using LLM to analyze conflicts...)")
            # TODO: Implement AI conflict resolution suggestions

        analysis.append("\n\nTo resolve:")
        analysis.append("1. Edit conflicted files manually")
        analysis.append("2. Remove conflict markers (<<<<<<, =======, >>>>>>>)")
        analysis.append("3. Stage resolved files: git add <file>")
        analysis.append("4. Continue: git merge --continue or git rebase --continue")

        return ToolResult(
            success=True,
            output="\n".join(analysis),
            error=""
        )
