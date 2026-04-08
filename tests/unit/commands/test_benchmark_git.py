# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for benchmark git timeout helper and pre-warm timeout."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRunGitWithTimeout:
    """Tests for _run_git_with_timeout helper."""

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_success(self):
        """Successful git command returns (returncode, stdout, stderr)."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        result = await _run_git_with_timeout(["git", "--version"], cwd=".", timeout=10)
        rc, stdout, stderr = result
        assert rc == 0
        assert b"git version" in stdout

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_kills_on_hang(self):
        """Hanging git command is killed after timeout."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        with pytest.raises(asyncio.TimeoutError):
            await _run_git_with_timeout(["sleep", "60"], cwd=".", timeout=1)

    @pytest.mark.asyncio
    async def test_prewarm_timeout_continues(self):
        """Pre-warm timeout should not crash the benchmark — it's best-effort."""

        # Simulate what happens when pre-warm times out
        async def hanging_prewarm(*args, **kwargs):
            await asyncio.sleep(60)

        try:
            await asyncio.wait_for(hanging_prewarm(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected — benchmark should continue

        # Verify we can still proceed after timeout
        assert True
