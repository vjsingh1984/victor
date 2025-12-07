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

"""Tests for git_tool module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from victor.tools.git_tool import git


class TestGitTool:
    """Tests for git function."""

    @pytest.mark.asyncio
    async def test_git_invalid_operation(self):
        """Test git with invalid operation."""
        result = await git(operation="invalid_op")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_git_status(self):
        """Test git status operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "clean", "")
            result = await git(operation="status")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_diff(self):
        """Test git diff operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "diff output", "")
            result = await git(operation="diff")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_log(self):
        """Test git log operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "commit logs", "")
            result = await git(operation="log", limit=5)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_branch(self):
        """Test git branch operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "* main", "")
            result = await git(operation="branch")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_stage_all(self):
        """Test git stage operation (add all)."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")
            result = await git(operation="stage")
            assert result["success"] is True
            mock.assert_called()

    @pytest.mark.asyncio
    async def test_git_stage_files(self):
        """Test git stage with specific files."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")
            result = await git(operation="stage", files=["test.py", "other.py"])
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_commit(self):
        """Test git commit operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "committed", "")
            result = await git(operation="commit", message="test commit")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_commit_no_message(self):
        """Test git commit without message fails."""
        result = await git(operation="commit")
        assert result["success"] is False
        assert "message" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_git_diff_staged(self):
        """Test git diff with staged option."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "staged diff", "")
            result = await git(operation="diff", staged=True)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_diff_with_files(self):
        """Test git diff with specific files."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "file diff", "")
            result = await git(operation="diff", files=["test.py"])
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_branch_create(self):
        """Test git branch create operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")
            result = await git(operation="branch", branch="new-feature")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_log_with_limit(self):
        """Test git log with custom limit."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "log entries", "")
            result = await git(operation="log", limit=20)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_missing_operation(self):
        """Test git with missing operation."""
        result = await git(operation="")
        assert result["success"] is False
        assert "operation" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_git_status_failure(self):
        """Test git status when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "not a git repository")
            result = await git(operation="status")
            assert result["success"] is False
            assert "not a git repository" in result["error"]

    @pytest.mark.asyncio
    async def test_git_diff_no_changes(self):
        """Test git diff when no changes."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")
            result = await git(operation="diff")
            assert result["success"] is True
            assert "No changes" in result["output"]

    @pytest.mark.asyncio
    async def test_git_diff_staged_no_changes(self):
        """Test git diff --staged when no staged changes."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")
            result = await git(operation="diff", staged=True)
            assert result["success"] is True
            assert "No staged changes" in result["output"]

    @pytest.mark.asyncio
    async def test_git_diff_failure(self):
        """Test git diff when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error running diff")
            result = await git(operation="diff")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_git_stage_failure(self):
        """Test git stage when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error staging")
            result = await git(operation="stage")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_git_commit_with_custom_author(self):
        """Test git commit with custom author name and email."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "committed", "")
            result = await git(
                operation="commit",
                message="test commit",
                author_name="Custom Author",
                author_email="custom@example.com",
            )
            assert result["success"] is True
            assert "Custom Author" in result["output"]
            assert "custom@example.com" in result["output"]
            # Verify env_overrides were passed
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("env_overrides") is not None
            assert call_kwargs["env_overrides"]["GIT_AUTHOR_NAME"] == "Custom Author"
            assert call_kwargs["env_overrides"]["GIT_AUTHOR_EMAIL"] == "custom@example.com"
            assert call_kwargs["env_overrides"]["GIT_COMMITTER_NAME"] == "Custom Author"
            assert call_kwargs["env_overrides"]["GIT_COMMITTER_EMAIL"] == "custom@example.com"

    @pytest.mark.asyncio
    async def test_git_commit_with_author_name_only(self):
        """Test git commit with only author name."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "committed", "")
            result = await git(
                operation="commit", message="test commit", author_name="Custom Author"
            )
            assert result["success"] is True
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("env_overrides") is not None
            assert call_kwargs["env_overrides"]["GIT_AUTHOR_NAME"] == "Custom Author"
            assert "GIT_AUTHOR_EMAIL" not in call_kwargs["env_overrides"]

    @pytest.mark.asyncio
    async def test_git_commit_with_author_email_only(self):
        """Test git commit with only author email."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "committed", "")
            result = await git(
                operation="commit", message="test commit", author_email="custom@example.com"
            )
            assert result["success"] is True
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("env_overrides") is not None
            assert call_kwargs["env_overrides"]["GIT_AUTHOR_EMAIL"] == "custom@example.com"
            assert "GIT_AUTHOR_NAME" not in call_kwargs["env_overrides"]

    @pytest.mark.asyncio
    async def test_git_commit_without_custom_author(self):
        """Test git commit without custom author uses default."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "committed", "")
            result = await git(operation="commit", message="test commit")
            assert result["success"] is True
            call_kwargs = mock.call_args.kwargs
            # env_overrides should be None when no author specified
            assert call_kwargs.get("env_overrides") is None

    @pytest.mark.asyncio
    async def test_git_commit_failure(self):
        """Test git commit when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "nothing to commit")
            result = await git(operation="commit", message="test")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_git_log_failure(self):
        """Test git log when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error")
            result = await git(operation="log")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_git_branch_failure(self):
        """Test git branch when git fails."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error")
            result = await git(operation="branch")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_git_branch_create_when_not_exist(self):
        """Test git branch creates new branch when doesn't exist."""
        with patch("victor.tools.git_tool._run_git") as mock:
            # First call fails (checkout fails), second call creates branch
            mock.side_effect = [
                (False, "", "error: pathspec did not match any file"),
                (True, "", ""),
            ]
            result = await git(operation="branch", branch="new-branch")
            assert result["success"] is True
            assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_git_with_options(self):
        """Test git with options parameter."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "output", "")
            result = await git(operation="status", options={"verbose": True})
            assert result["success"] is True


class TestRunGit:
    """Tests for _run_git helper function."""

    def test_run_git_success(self):
        """Test _run_git success case."""
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0
            mock.return_value.stdout = "output"
            mock.return_value.stderr = ""
            success, stdout, stderr = _run_git("status")
            assert success is True
            assert stdout == "output"

    def test_run_git_failure(self):
        """Test _run_git failure case."""
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 1
            mock.return_value.stdout = ""
            mock.return_value.stderr = "error"
            success, stdout, stderr = _run_git("invalid")
            assert success is False
            assert stderr == "error"

    def test_run_git_timeout(self):
        """Test _run_git timeout handling."""
        import subprocess
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.side_effect = subprocess.TimeoutExpired("git", 30)
            success, stdout, stderr = _run_git("status")
            assert success is False
            assert "timed out" in stderr.lower()

    def test_run_git_with_env_overrides(self):
        """Test _run_git with environment overrides."""
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0
            mock.return_value.stdout = "output"
            mock.return_value.stderr = ""

            env_overrides = {
                "GIT_AUTHOR_NAME": "Custom Author",
                "GIT_AUTHOR_EMAIL": "custom@example.com",
            }
            success, stdout, stderr = _run_git("commit", "-m", "test", env_overrides=env_overrides)
            assert success is True

            # Verify subprocess.run was called with the correct env
            call_kwargs = mock.call_args.kwargs
            assert "env" in call_kwargs
            assert call_kwargs["env"]["GIT_AUTHOR_NAME"] == "Custom Author"
            assert call_kwargs["env"]["GIT_AUTHOR_EMAIL"] == "custom@example.com"

    def test_run_git_without_env_overrides(self):
        """Test _run_git without environment overrides passes None env."""
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.return_value.returncode = 0
            mock.return_value.stdout = "output"
            mock.return_value.stderr = ""

            success, stdout, stderr = _run_git("status")
            assert success is True

            # Verify subprocess.run was called with env=None
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs.get("env") is None

    def test_run_git_exception(self):
        """Test _run_git exception handling."""
        from victor.tools.git_tool import _run_git

        with patch("subprocess.run") as mock:
            mock.side_effect = Exception("unexpected error")
            success, stdout, stderr = _run_git("status")
            assert success is False
            assert "unexpected error" in stderr


class TestSetGitProvider:
    """Tests for set_git_provider function."""

    def test_set_git_provider(self):
        """Test setting git provider."""
        from victor.tools.git_tool import set_git_provider
        import victor.tools.git_tool as git_module

        mock_provider = object()
        set_git_provider(mock_provider, "test-model")
        assert git_module._provider is mock_provider
        assert git_module._model == "test-model"

    def test_set_git_provider_without_model(self):
        """Test setting git provider without model."""
        from victor.tools.git_tool import set_git_provider
        import victor.tools.git_tool as git_module

        mock_provider = object()
        set_git_provider(mock_provider)
        assert git_module._provider is mock_provider
        assert git_module._model is None


class TestGitSuggestCommit:
    """Tests for git_suggest_commit function."""

    @pytest.mark.asyncio
    async def test_no_provider(self):
        """Test git_suggest_commit without provider."""
        from victor.tools.git_tool import git_suggest_commit
        import victor.tools.git_tool as git_module

        # Save and clear provider
        original = git_module._provider
        git_module._provider = None
        try:
            result = await git_suggest_commit()
            assert result["success"] is False
            assert "provider" in result["error"].lower()
        finally:
            git_module._provider = original

    @pytest.mark.asyncio
    async def test_diff_failure(self):
        """Test git_suggest_commit when diff fails."""
        from victor.tools.git_tool import git_suggest_commit, set_git_provider
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        set_git_provider(mock_provider, "test-model")

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error getting diff")
            result = await git_suggest_commit()
            assert result["success"] is False
            assert "error getting diff" in result["error"]

    @pytest.mark.asyncio
    async def test_no_staged_changes(self):
        """Test git_suggest_commit with no staged changes."""
        from victor.tools.git_tool import git_suggest_commit, set_git_provider
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        set_git_provider(mock_provider, "test-model")

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "", "")  # Empty diff
            result = await git_suggest_commit()
            assert result["success"] is False
            assert "No staged changes" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        """Test successful commit message generation."""
        from victor.tools.git_tool import git_suggest_commit, set_git_provider
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "feat(api): add new endpoint"
        mock_provider.complete = AsyncMock(return_value=mock_response)
        set_git_provider(mock_provider, "test-model")

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.side_effect = [
                (True, "+ new line\n- old line", ""),  # diff --staged
                (True, "test.py\napi.py", ""),  # diff --staged --name-only
            ]
            result = await git_suggest_commit()
            assert result["success"] is True
            assert "feat(api)" in result["output"]

    @pytest.mark.asyncio
    async def test_llm_error(self):
        """Test git_suggest_commit when LLM fails."""
        from victor.tools.git_tool import git_suggest_commit, set_git_provider
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("LLM error"))
        set_git_provider(mock_provider, "test-model")

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.side_effect = [
                (True, "+ new line", ""),  # diff --staged
                (True, "test.py", ""),  # diff --staged --name-only
            ]
            result = await git_suggest_commit()
            assert result["success"] is False
            assert "AI generation failed" in result["error"]


class TestGitCreatePR:
    """Tests for git_create_pr function."""

    @pytest.mark.asyncio
    async def test_get_branch_failure(self):
        """Test git_create_pr when getting branch fails."""
        from victor.tools.git_tool import git_create_pr

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "not a git repo")
            result = await git_create_pr()
            assert result["success"] is False
            assert "not a git repo" in result["error"]

    @pytest.mark.asyncio
    async def test_push_failure(self):
        """Test git_create_pr when push fails."""
        from victor.tools.git_tool import git_create_pr
        import victor.tools.git_tool as git_module

        # Clear provider to skip AI generation
        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock:
                mock.side_effect = [
                    (True, "feature-branch\n", ""),  # branch --show-current
                    (False, "", "push rejected"),  # push
                ]
                result = await git_create_pr()
                assert result["success"] is False
                assert "push" in result["error"].lower()
        finally:
            git_module._provider = original

    @pytest.mark.asyncio
    async def test_gh_not_found(self):
        """Test git_create_pr when gh CLI not found."""
        from victor.tools.git_tool import git_create_pr
        import victor.tools.git_tool as git_module

        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock_git:
                mock_git.side_effect = [
                    (True, "feature-branch\n", ""),  # branch --show-current
                    (True, "", ""),  # push
                ]
                with patch("subprocess.run") as mock_run:
                    mock_run.side_effect = FileNotFoundError()
                    result = await git_create_pr()
                    assert result["success"] is False
                    assert "gh" in result["error"].lower()
        finally:
            git_module._provider = original

    @pytest.mark.asyncio
    async def test_pr_creation_success(self):
        """Test successful PR creation."""
        from victor.tools.git_tool import git_create_pr
        import victor.tools.git_tool as git_module

        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock_git:
                mock_git.side_effect = [
                    (True, "feature-branch\n", ""),  # branch --show-current
                    (True, "", ""),  # push
                ]
                with patch("subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = "https://github.com/user/repo/pull/1"
                    mock_run.return_value = mock_result
                    result = await git_create_pr(pr_title="Test PR", pr_description="Test desc")
                    assert result["success"] is True
                    assert "PR created" in result["output"]
        finally:
            git_module._provider = original

    @pytest.mark.asyncio
    async def test_pr_creation_failure(self):
        """Test PR creation failure."""
        from victor.tools.git_tool import git_create_pr
        import victor.tools.git_tool as git_module

        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock_git:
                mock_git.side_effect = [
                    (True, "feature-branch\n", ""),  # branch
                    (True, "", ""),  # push
                ]
                with patch("subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.returncode = 1
                    mock_result.stderr = "PR already exists"
                    mock_run.return_value = mock_result
                    result = await git_create_pr(pr_title="Test", pr_description="Desc")
                    assert result["success"] is False
                    assert "PR already exists" in result["error"]
        finally:
            git_module._provider = original


class TestGitAnalyzeConflicts:
    """Tests for git_analyze_conflicts function."""

    @pytest.mark.asyncio
    async def test_status_failure(self):
        """Test git_analyze_conflicts when status fails."""
        from victor.tools.git_tool import git_analyze_conflicts

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (False, "", "error")
            result = await git_analyze_conflicts()
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_no_conflicts(self):
        """Test git_analyze_conflicts with no conflicts."""
        from victor.tools.git_tool import git_analyze_conflicts

        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, " M test.py\n", "")
            result = await git_analyze_conflicts()
            assert result["success"] is True
            assert "No merge conflicts" in result["output"]

    @pytest.mark.asyncio
    async def test_with_conflicts(self):
        """Test git_analyze_conflicts with conflicts."""
        from victor.tools.git_tool import git_analyze_conflicts
        import victor.tools.git_tool as git_module

        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock:
                mock.return_value = (True, "UU conflict.py\n", "")
                with patch("builtins.open") as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = (
                        "<<<<<<< HEAD\nour content\n=======\ntheir content\n>>>>>>> branch\n"
                    )
                    result = await git_analyze_conflicts()
                    assert result["success"] is True
                    assert "1 conflicted file" in result["output"]
        finally:
            git_module._provider = original

    @pytest.mark.asyncio
    async def test_file_read_error(self):
        """Test git_analyze_conflicts when file read fails."""
        from victor.tools.git_tool import git_analyze_conflicts
        import victor.tools.git_tool as git_module

        original = git_module._provider
        git_module._provider = None
        try:
            with patch("victor.tools.git_tool._run_git") as mock:
                mock.return_value = (True, "UU missing.py\n", "")
                with patch("builtins.open") as mock_open:
                    mock_open.side_effect = FileNotFoundError("file not found")
                    result = await git_analyze_conflicts()
                    assert result["success"] is True
                    assert "Error reading file" in result["output"]
        finally:
            git_module._provider = original
