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

"""Tests for Git Safety Middleware consolidation (Phase 2.2).

Tests that coding GitSafetyMiddleware:
1. Uses framework implementation
2. Maintains backward compatibility
"""

import pytest

from victor.coding.middleware import GitSafetyMiddleware as CodingGitSafetyMiddleware
from victor.framework.middleware.framework import GitSafetyMiddleware as FrameworkGitSafetyMiddleware


class TestGitSafetyConsolidation:
    """Test Git Safety Middleware consolidation."""

    def test_coding_uses_framework_implementation(self):
        """coding GitSafetyMiddleware should inherit from framework."""
        # The coding version should be a subclass of the framework version
        assert issubclass(CodingGitSafetyMiddleware, FrameworkGitSafetyMiddleware)

    def test_backward_compatibility_init(self):
        """Coding middleware should accept original init parameters."""
        # Original parameters should still work
        middleware = CodingGitSafetyMiddleware(
            block_dangerous=True,
            warn_on_risky=True,
        )

        assert middleware._block_dangerous is True
        assert middleware._warn_on_risky is True

    @pytest.mark.asyncio
    async def test_backward_compatibility_behavior(self):
        """Coding middleware should behave the same as before."""
        middleware = CodingGitSafetyMiddleware(
            block_dangerous=True,
            warn_on_risky=True,
        )

        # Test blocking dangerous operation (framework requires "git " prefix)
        result = await middleware.before_tool_call(
            "git", {"command": "git push --force origin main"}
        )
        assert result.proceed is False

        # Test warning for risky operation
        result = await middleware.before_tool_call(
            "git", {"command": "git reset --hard"}
        )
        # Should proceed but with warning
        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_non_git_tools_pass_through(self):
        """Non-git tools should pass through unchanged."""
        middleware = CodingGitSafetyMiddleware(block_dangerous=True)

        result = await middleware.before_tool_call(
            "read_file", {"path": "/tmp/test.txt"}
        )
        assert result.proceed is True

    def test_priority_is_critical(self):
        """Git safety should have CRITICAL priority."""
        from victor.core.verticals.protocols import MiddlewarePriority

        middleware = CodingGitSafetyMiddleware()
        assert middleware.get_priority() == MiddlewarePriority.CRITICAL

    def test_applicable_tools(self):
        """Should be applicable to git-related tools."""
        middleware = CodingGitSafetyMiddleware()
        tools = middleware.get_applicable_tools()

        assert "git" in tools
        assert "execute_bash" in tools or "bash" in tools


class TestFrameworkFeatures:
    """Test that framework features are available in coding middleware."""

    def test_protected_branches_available(self):
        """Framework protected branches should be available."""
        middleware = CodingGitSafetyMiddleware()

        # Protected branches from framework
        assert "main" in middleware._protected_branches
        assert "master" in middleware._protected_branches

    def test_allowed_force_branches_configurable(self):
        """Should be able to configure allowed force branches."""
        middleware = CodingGitSafetyMiddleware(
            allowed_force_branches={"feature/*"},
        )

        assert "feature/*" in middleware._allowed_force_branches

    def test_custom_blocked_operations(self):
        """Should be able to add custom blocked operations."""
        middleware = CodingGitSafetyMiddleware(
            custom_blocked={"my-dangerous-command"},
        )

        assert "my-dangerous-command" in middleware._blocked
