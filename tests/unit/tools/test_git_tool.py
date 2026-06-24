# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for the unified ``git`` command tool.

Covers the bash-style parser, delegation to ``victor_devops.git`` when the
package is present, the shell fallback when it is absent, AI commit message
generation, and PR creation.
"""

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.unified.git_tool import (
    _fallback_command,
    _git_kwarg_map,
    create_git_parser,
    git_tool,
)
from victor.tools.unified.parser import split_command


def _parse(cmd: str) -> argparse.Namespace:
    """Parse using the real shlex-aware splitter (handles quoted messages)."""
    parser = create_git_parser()
    parts = split_command(cmd)
    if parts and parts[0] == "git":
        parts = parts[1:]
    return parser.parse_args(parts)


class TestGitParser:
    """Subcommand parsing."""

    def test_status(self):
        assert _parse("git status").subcommand == "status"

    def test_diff_staged_with_files(self):
        ns = _parse("git diff --staged a.py b.py")
        assert ns.subcommand == "diff"
        assert ns.staged is True
        assert ns.files == ["a.py", "b.py"]

    def test_log_limit(self):
        ns = _parse("git log -n 25")
        assert ns.subcommand == "log"
        assert ns.limit == 25

    def test_stage_and_add_alias(self):
        assert _parse("git stage a.py").subcommand == "stage"
        assert _parse("git add a.py").subcommand == "add"

    def test_commit_message_and_ai(self):
        ns = _parse('git commit -m "fix it"')
        assert ns.subcommand == "commit"
        assert ns.message == "fix it"
        assert ns.ai is False
        ns_ai = _parse("git commit --ai")
        assert ns_ai.ai is True

    def test_push_options(self):
        ns = _parse("git push origin main --force --tags")
        assert ns.subcommand == "push"
        assert ns.remote == "origin"
        assert ns.branch == "main"
        assert ns.force is True
        assert ns.tags is True

    def test_branch_optional_name(self):
        assert _parse("git branch").name is None
        assert _parse("git branch feat/x").name == "feat/x"

    def test_pr_subcommand(self):
        ns = _parse("git pr --title T --base develop")
        assert ns.subcommand == "pr"
        assert ns.title == "T"
        assert ns.base == "develop"


class TestKwargMap:
    """Bash args -> ``victor_devops.git(operation=...)`` kwargs."""

    def test_status(self):
        op, kwargs = _git_kwarg_map(_parse("git status"))
        assert op == "status" and kwargs == {}

    def test_diff_maps_staged_and_files(self):
        op, kwargs = _git_kwarg_map(_parse("git diff --staged a.py"))
        assert op == "diff"
        assert kwargs == {"staged": True, "files": ["a.py"]}

    def test_log_maps_limit(self):
        op, kwargs = _git_kwarg_map(_parse("git log -n 5"))
        assert op == "log" and kwargs == {"limit": 5}

    def test_add_alias_normalizes_to_stage(self):
        op, kwargs = _git_kwarg_map(_parse("git add a.py b.py"))
        assert op == "stage"
        assert kwargs == {"files": ["a.py", "b.py"]}

    def test_commit_maps_message_and_author(self):
        op, kwargs = _git_kwarg_map(
            _parse('git commit -m "msg" --author-name Bot --author-email b@x.io')
        )
        assert op == "commit"
        assert kwargs["message"] == "msg"
        assert kwargs["author_name"] == "Bot"
        assert kwargs["author_email"] == "b@x.io"

    def test_push_maps_options(self):
        op, kwargs = _git_kwarg_map(_parse("git push origin main --force"))
        assert op == "push"
        assert kwargs["branch"] == "main"
        assert kwargs["options"] == {"remote": "origin", "force": True}


class TestFallbackCommand:
    """Shell fallback command construction (used when devops is absent)."""

    def test_status_is_readonly(self):
        sub, argv, ro = _fallback_command(_parse("git status"))
        assert sub == "status" and ro is True

    def test_stage_uses_add_and_dot(self):
        sub, argv, ro = _fallback_command(_parse("git stage"))
        assert sub == "add" and argv == ["."] and ro is False

    def test_stage_specific_files(self):
        sub, argv, ro = _fallback_command(_parse("git stage a.py b.py"))
        assert sub == "add" and argv == ["a.py", "b.py"] and ro is False

    def test_commit_requires_message(self):
        with pytest.raises(ValueError, match="commit message required"):
            _fallback_command(_parse("git commit"))

    def test_commit_message_built(self):
        sub, argv, ro = _fallback_command(_parse('git commit -m "fix"'))
        assert sub == "commit" and argv == ["-m", "fix"] and ro is False

    def test_ai_only_ops_require_devops(self):
        with pytest.raises(ValueError, match="victor-devops"):
            _fallback_command(_parse("git commit_msg"))
        with pytest.raises(ValueError, match="victor-devops"):
            _fallback_command(_parse("git pr --title t"))


class TestDelegationPath:
    """When ``victor_devops.git`` is resolvable, the dispatcher delegates."""

    @pytest.mark.asyncio
    async def test_status_delegates_to_devops(self):
        mock_git = AsyncMock(return_value={"success": True, "output": "clean tree"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git status")
        mock_git.assert_awaited_once_with(operation="status")
        assert result == "clean tree"

    @pytest.mark.asyncio
    async def test_commit_delegates_with_message(self):
        mock_git = AsyncMock(return_value={"success": True, "output": "Committed"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool('git commit -m "fix: x"')
        mock_git.assert_awaited_once_with(operation="commit", message="fix: x", author_name=None, author_email=None)
        assert "Committed" in result

    @pytest.mark.asyncio
    async def test_failed_devops_result_is_formatted_as_error(self):
        mock_git = AsyncMock(return_value={"success": False, "error": "no upstream"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git push")
        assert "### ❌ ERROR" in result
        assert "no upstream" in result


class TestFallbackPath:
    """When devops is absent, the dispatcher falls back to the shell tool."""

    @pytest.mark.asyncio
    async def test_status_uses_shell_when_devops_absent(self):
        mock_shell = AsyncMock(return_value={"success": True, "stdout": "## develop"})
        with (
            patch(
                "victor.tools.unified.git_tool.resolve_vertical_callable",
                return_value=(None, None),
            ),
            patch("victor.tools.bash.shell", mock_shell),
        ):
            result = await git_tool("git status")
        mock_shell.assert_awaited_once()
        called_cmd = mock_shell.call_args.kwargs["cmd"]
        assert called_cmd.startswith("git status")
        assert mock_shell.call_args.kwargs["readonly"] is True
        assert "## develop" in result

    @pytest.mark.asyncio
    async def test_commit_msg_without_devops_returns_error(self):
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await git_tool("git commit_msg")
        assert "### ❌ ERROR" in result
        assert "victor-devops" in result


class TestAiCommit:
    """AI commit message generation flow."""

    @pytest.mark.asyncio
    async def test_ai_commit_generates_then_commits(self):
        mock_git = AsyncMock(
            side_effect=[
                {"success": True, "message": "feat: add x"},  # commit_msg
                {"success": True, "output": "Committed"},  # commit
            ]
        )
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_git, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool("git commit --ai")
        assert mock_git.await_count == 2
        # First call generates the message.
        first_call = mock_git.await_args_list[0]
        assert first_call.kwargs["operation"] == "commit_msg"
        # Second call commits with the generated message.
        second_call = mock_git.await_args_list[1]
        assert second_call.kwargs == {
            "operation": "commit",
            "message": "feat: add x",
            "author_name": None,
            "author_email": None,
        }
        assert "feat: add x" in result
        assert "Committed" in result

    @pytest.mark.asyncio
    async def test_ai_commit_requires_devops(self):
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await git_tool("git commit --ai")
        assert "### ❌ ERROR" in result
        assert "victor-devops" in result


class TestPrAndEdgeCases:
    """PR creation and edge-case handling."""

    @pytest.mark.asyncio
    async def test_pr_delegates_to_pr_callable(self):
        mock_pr = AsyncMock(return_value={"success": True, "output": "PR #42 created"})
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(mock_pr, "victor_devops.tools.git_tool"),
        ):
            result = await git_tool('git pr --title "Add auth" --base develop')
        mock_pr.assert_awaited_once_with(pr_title="Add auth", base_branch="develop")
        assert "PR #42 created" in result

    @pytest.mark.asyncio
    async def test_pr_without_devops_returns_error(self):
        with patch(
            "victor.tools.unified.git_tool.resolve_vertical_callable",
            return_value=(None, None),
        ):
            result = await git_tool("git pr --title T")
        assert "### ❌ ERROR" in result

    @pytest.mark.asyncio
    async def test_no_subcommand_returns_help(self):
        result = await git_tool("git")
        assert "### ❌ ERROR" in result
        assert "usage" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_subcommand_returns_error(self):
        result = await git_tool("git bogus")
        assert "### ❌ ERROR" in result

    @pytest.mark.asyncio
    async def test_tool_registered_with_canonical_name(self):
        assert git_tool.Tool.name == "git"
