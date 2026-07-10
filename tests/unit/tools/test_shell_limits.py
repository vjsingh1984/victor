"""Tests for separate stdout/stderr output limits on the shell tool.

Recovered from the orphaned root-level ``test_shell_limits.py`` (which was never
collected by pytest because ``testpaths = ["tests"]``). The hardcoded
``sys.path`` hack and ``__main__`` runner were removed so pytest's async auto
mode picks it up cleanly. These assertions are unique coverage: no other test in
``tests/unit/tools`` checks ``stdout_limit`` / ``stderr_limit`` behaviour.
"""

import pytest

from victor.tools.bash import shell


@pytest.mark.asyncio
async def test_separate_limits():
    """Unlimited output (None) preserves all lines."""

    result = await shell(
        cmd="echo 'test'; echo 'test2'; echo 'test3'",
        stdout_limit=None,
        stderr_limit=None,
    )
    assert "[truncated]" not in result["stdout"]
    assert result["stdout_lines"] == 3


@pytest.mark.asyncio
async def test_single_stdout_limit():
    """A stdout_limit caps stdout_lines and marks truncation."""

    result = await shell(
        cmd="echo 'line1'; echo 'line2'; echo 'line3'",
        stdout_limit=2,
    )
    assert "truncated" in result["stdout"].lower() or result.get("truncated", False)
    assert result["stdout_lines"] == 2


@pytest.mark.asyncio
async def test_separate_stdout_stderr_limits():
    """stdout and stderr limits are applied independently."""

    result = await shell(
        cmd="echo 'stdout'; >&2 echo 'stderr1'; >&2 echo 'stderr2'",
        stdout_limit=1,
        stderr_limit=10,
    )
    assert result["stdout_lines"] == 1
    assert result["stderr_lines"] >= 2


@pytest.mark.asyncio
async def test_none_is_unlimited():
    """stdout_limit=None preserves arbitrarily large output."""

    result = await shell(
        cmd="seq 1 100",  # 100 lines
        stdout_limit=None,
    )
    assert result["stdout_lines"] == 100


@pytest.mark.asyncio
async def test_default_limits_present():
    """Default invocation still returns stdout_lines / stderr_lines keys."""

    result = await shell(cmd="echo 'test'")
    assert "stdout_lines" in result
    assert "stderr_lines" in result
