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

"""Unit tests for RecursionContext and RecursionDepthError."""

import pytest

from victor.core.errors import RecursionDepthError
from victor.workflows.recursion import RecursionContext, RecursionGuard


class TestRecursionContext:
    """Test RecursionContext functionality."""

    def test_initialization(self):
        """Test default initialization."""
        ctx = RecursionContext()
        assert ctx.current_depth == 0
        assert ctx.max_depth == 3
        assert ctx.execution_stack == []

    def test_custom_max_depth(self):
        """Test initialization with custom max_depth."""
        ctx = RecursionContext(max_depth=5)
        assert ctx.max_depth == 5

    def test_enter_exit(self):
        """Test enter and exit operations."""
        ctx = RecursionContext(max_depth=3)

        # First level
        ctx.enter("workflow", "main")
        assert ctx.current_depth == 1
        assert len(ctx.execution_stack) == 1
        assert ctx.execution_stack == ["workflow:main"]

        # Second level
        ctx.enter("team", "research")
        assert ctx.current_depth == 2
        assert ctx.execution_stack == ["workflow:main", "team:research"]

        # Exit first
        ctx.exit()
        assert ctx.current_depth == 1
        assert ctx.execution_stack == ["workflow:main"]

        # Exit second
        ctx.exit()
        assert ctx.current_depth == 0
        assert ctx.execution_stack == []

    def test_recursion_limit_enforced(self):
        """Test that recursion limit is enforced."""
        ctx = RecursionContext(max_depth=2)

        ctx.enter("workflow", "level1")
        ctx.enter("team", "level2")

        # Third level should raise error
        with pytest.raises(RecursionDepthError) as exc_info:
            ctx.enter("workflow", "level3")

        error = exc_info.value
        assert error.current_depth == 2
        assert error.max_depth == 2
        assert "workflow:level1" in error.execution_stack
        assert "team:level2" in error.execution_stack

    def test_can_nest(self):
        """Test can_nest method."""
        ctx = RecursionContext(max_depth=3)

        assert ctx.can_nest(1) is True  # 0 -> 1
        assert ctx.can_nest(2) is True  # 0 -> 2
        assert ctx.can_nest(3) is True  # 0 -> 3
        assert ctx.can_nest(4) is False  # would exceed

        ctx.enter("workflow", "test")
        assert ctx.can_nest(2) is True  # 1 -> 3
        assert ctx.can_nest(3) is False  # would exceed

    def test_get_depth_info(self):
        """Test get_depth_info method."""
        ctx = RecursionContext(max_depth=5)
        ctx.enter("workflow", "main")
        ctx.enter("team", "research")

        info = ctx.get_depth_info()
        assert info["current_depth"] == 2
        assert info["max_depth"] == 5
        assert info["remaining_depth"] == 3
        assert len(info["execution_stack"]) == 2

    def test_context_manager(self):
        """Test context manager behavior."""
        ctx = RecursionContext(max_depth=3)

        with ctx:
            ctx.enter("workflow", "test")
            assert ctx.current_depth == 1

        # Should be reset after exiting context
        assert ctx.current_depth == 0
        assert ctx.execution_stack == []

    def test_exit_when_empty(self):
        """Test exit() when already at depth 0 is safe."""
        ctx = RecursionContext()
        ctx.exit()  # Should not raise
        ctx.exit()  # Should not raise
        assert ctx.current_depth == 0

    def test_repr(self):
        """Test __repr__ method."""
        ctx = RecursionContext(max_depth=3)
        ctx.enter("workflow", "test")

        repr_str = repr(ctx)
        assert "RecursionContext" in repr_str
        assert "depth=1/3" in repr_str
        assert "workflow:test" in repr_str


class TestRecursionGuard:
    """Test RecursionGuard context manager."""

    def test_automatic_cleanup(self):
        """Test that RecursionGuard automatically cleans up."""
        ctx = RecursionContext(max_depth=3)

        with RecursionGuard(ctx, "workflow", "test"):
            assert ctx.current_depth == 1
            assert ctx.execution_stack == ["workflow:test"]

        # Should auto-exit
        assert ctx.current_depth == 0
        assert ctx.execution_stack == []

    def test_guard_with_exception(self):
        """Test that RecursionGuard exits even on exception."""
        ctx = RecursionContext(max_depth=3)

        try:
            with RecursionGuard(ctx, "workflow", "test"):
                assert ctx.current_depth == 1
                raise ValueError("test error")
        except ValueError:
            pass

        # Should still exit
        assert ctx.current_depth == 0
        assert ctx.execution_stack == []

    def test_nested_guards(self):
        """Test nested RecursionGuard contexts."""
        ctx = RecursionContext(max_depth=3)

        with RecursionGuard(ctx, "workflow", "outer"):
            assert ctx.current_depth == 1
            with RecursionGuard(ctx, "team", "inner"):
                assert ctx.current_depth == 2
            assert ctx.current_depth == 1
        assert ctx.current_depth == 0

    def test_guard_enforces_limit(self):
        """Test that RecursionGuard enforces recursion limit."""
        ctx = RecursionContext(max_depth=2)

        with RecursionGuard(ctx, "workflow", "level1"):
            with RecursionGuard(ctx, "team", "level2"):
                # Third level should raise error
                with pytest.raises(RecursionDepthError):
                    with RecursionGuard(ctx, "workflow", "level3"):
                        pass


class TestRecursionDepthError:
    """Test RecursionDepthError exception."""

    def test_error_creation(self):
        """Test creating RecursionDepthError."""
        error = RecursionDepthError(
            message="Too deep",
            current_depth=5,
            max_depth=3,
            execution_stack=[
                "workflow:main",
                "team:outer",
                "team:middle",
                "workflow:nested",
                "team:inner",
            ],
        )

        assert error.current_depth == 5
        assert error.max_depth == 3
        assert len(error.execution_stack) == 5
        assert "Too deep" in str(error)
        assert "5/3" in str(error)

    def test_error_has_recovery_hint(self):
        """Test that error includes recovery hint."""
        error = RecursionDepthError(
            message="Limit exceeded",
            current_depth=4,
            max_depth=3,
            execution_stack=["a", "b", "c", "d"],
        )

        # Should have auto-generated recovery hint
        assert error.recovery_hint is not None
        assert "Reduce nesting depth" in error.recovery_hint
        assert "4/3" in error.recovery_hint

    def test_error_str_representation(self):
        """Test string representation of error."""
        error = RecursionDepthError(
            message="Maximum recursion depth exceeded",
            current_depth=3,
            max_depth=2,
            execution_stack=["workflow:main", "team:research", "workflow:nested"],
        )

        error_str = str(error)
        assert "Recursion depth limit exceeded" in error_str
        assert "3/2" in error_str
        assert "workflow:main" in error_str
        assert "team:research" in error_str
        assert "workflow:nested" in error_str

    def test_error_details_dict(self):
        """Test that error details are populated."""
        error = RecursionDepthError(
            message="Test error",
            current_depth=2,
            max_depth=3,
            execution_stack=["workflow:test"],
        )

        assert error.details["current_depth"] == 2
        assert error.details["max_depth"] == 3
        assert error.details["execution_stack"] == ["workflow:test"]
        assert "stack_trace" in error.details

    def test_custom_recovery_hint(self):
        """Test that custom recovery hint is preserved."""
        custom_hint = "Custom recovery instructions"

        error = RecursionDepthError(
            message="Test",
            current_depth=5,
            max_depth=3,
            execution_stack=["a", "b", "c", "d", "e"],
            recovery_hint=custom_hint,
        )

        assert error.recovery_hint == custom_hint
        assert error.recovery_hint == custom_hint

    def test_empty_execution_stack(self):
        """Test error with empty execution stack."""
        error = RecursionDepthError(
            message="No stack",
            current_depth=0,
            max_depth=3,
            execution_stack=[],
        )

        assert error.execution_stack == []
        assert error.details["stack_trace"] == "empty"
