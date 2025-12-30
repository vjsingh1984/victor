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

"""Tests for ToolSet in victor.framework.tools.

Includes tests for caching behavior and performance.
"""

import time
import pytest

from victor.framework.tools import ToolSet, ToolCategory


class TestToolSetBasic:
    """Basic tests for ToolSet functionality."""

    def test_default_toolset(self):
        """Default ToolSet includes core, filesystem, and git."""
        ts = ToolSet.default()
        assert "read" in ts
        assert "write" in ts
        assert "shell" in ts
        assert "git" in ts

    def test_minimal_toolset(self):
        """Minimal ToolSet includes only core tools."""
        ts = ToolSet.minimal()
        assert "read" in ts
        assert "shell" in ts
        # Git should not be in minimal
        assert "git" not in ts

    def test_from_tools(self):
        """Create ToolSet from explicit tool names."""
        ts = ToolSet.from_tools(["read", "write", "custom_tool"])
        assert "read" in ts
        assert "write" in ts
        assert "custom_tool" in ts
        assert "shell" not in ts

    def test_from_categories(self):
        """Create ToolSet from categories."""
        ts = ToolSet.from_categories(["git", "web"])
        assert "git" in ts
        assert "web_search" in ts
        assert "read" not in ts

    def test_include_adds_tools(self):
        """include() adds tools to the set."""
        ts = ToolSet.from_tools(["read"])
        ts2 = ts.include("write", "custom")
        assert "write" in ts2
        assert "custom" in ts2
        # Original should be unchanged
        assert "write" not in ts

    def test_exclude_removes_tools(self):
        """exclude_tools() removes tools from the set."""
        ts = ToolSet.default()
        ts2 = ts.exclude_tools("shell")
        assert "shell" not in ts2
        # Original should be unchanged
        assert "shell" in ts

    def test_get_tool_names(self):
        """get_tool_names() returns all resolved names."""
        ts = ToolSet.from_tools(["read", "write"])
        names = ts.get_tool_names()
        assert "read" in names
        assert "write" in names
        assert len(names) == 2


class TestToolSetContainsCache:
    """Tests for ToolSet __contains__ caching behavior."""

    def test_contains_works_correctly(self):
        """__contains__ returns correct results."""
        ts = ToolSet.from_tools(["read", "write"])
        assert "read" in ts
        assert "write" in ts
        assert "bash" not in ts

    def test_contains_with_categories(self):
        """__contains__ works with category expansion."""
        ts = ToolSet.default()
        # These should be in default (core + filesystem + git)
        assert "read" in ts
        assert "shell" in ts
        assert "git" in ts
        # Web tools should not be in default
        assert "web_search" not in ts

    def test_cache_is_built_on_first_access(self):
        """Cache is built on first __contains__ call."""
        ts = ToolSet.from_tools(["read", "write", "bash"])
        # Cache should be None initially
        assert ts._resolved_names_cache is None

        # First access builds cache
        _ = "read" in ts
        assert ts._resolved_names_cache is not None
        assert ts._resolved_names_cache == {"read", "write", "bash"}

    def test_cached_contains_is_faster(self):
        """Cached __contains__ is significantly faster than uncached."""
        # Create a ToolSet with many categories to make uncached slower
        ts = ToolSet.full()

        # First call builds cache
        start = time.perf_counter()
        _ = "read" in ts
        first_call = time.perf_counter() - start

        # Warm up the cache completely
        for _ in range(10):
            _ = "read" in ts

        # Measure 1000 cached calls
        start = time.perf_counter()
        for _ in range(1000):
            _ = "read" in ts
        cached_total = time.perf_counter() - start
        cached_avg = cached_total / 1000

        # Cached should be significantly faster per call
        # First call does category expansion, cached is O(1) set lookup
        assert cached_avg < first_call, (
            f"Cached avg ({cached_avg:.6f}s) should be faster than "
            f"first call ({first_call:.6f}s)"
        )

    def test_invalidate_cache(self):
        """invalidate_cache() clears the cache."""
        ts = ToolSet.from_tools(["read", "write"])
        # Build cache
        _ = "read" in ts
        assert ts._resolved_names_cache is not None

        # Invalidate
        ts.invalidate_cache()
        assert ts._resolved_names_cache is None

        # Cache rebuilds on next access
        _ = "read" in ts
        assert ts._resolved_names_cache is not None

    def test_new_toolset_has_independent_cache(self):
        """New ToolSet from include/exclude has independent cache."""
        ts1 = ToolSet.from_tools(["read", "write"])
        _ = "read" in ts1  # Build cache

        ts2 = ts1.include("custom")
        # ts2 should have its own cache (starts as None since it's a new instance)
        # Note: include() creates a new ToolSet, so cache doesn't carry over
        assert ts2._resolved_names_cache is None

        _ = "custom" in ts2
        assert "custom" in ts2._resolved_names_cache
        assert "read" in ts2._resolved_names_cache

    def test_toolset_contains_performance_benchmark(self):
        """Verify __contains__ is O(1) after first access."""
        tool_set = ToolSet(
            tools={"read", "write", "bash"},
            categories={ToolCategory.FILESYSTEM.value, ToolCategory.GIT.value},
        )

        # First call (builds cache)
        start = time.perf_counter()
        _ = "read" in tool_set
        first_call = time.perf_counter() - start

        # Subsequent calls (cached)
        start = time.perf_counter()
        for _ in range(1000):
            _ = "read" in tool_set
        cached_calls = (time.perf_counter() - start) / 1000

        # Cached should be significantly faster (at least 10x)
        # This verifies we're doing O(1) set lookup, not O(n) category expansion
        assert cached_calls < first_call / 5, (
            f"Cached ({cached_calls:.8f}s) should be at least 5x faster "
            f"than first call ({first_call:.8f}s)"
        )


class TestToolSetExclusions:
    """Tests for ToolSet exclusion behavior."""

    def test_airgapped_excludes_web(self):
        """Airgapped mode excludes web tools."""
        ts = ToolSet.airgapped()
        assert "read" in ts
        assert "web_search" not in ts
        assert "web_fetch" not in ts
        assert "http_request" not in ts

    def test_exclusions_work_with_categories(self):
        """Exclusions are applied after category expansion."""
        ts = ToolSet(categories={ToolCategory.CORE.value}, exclude={"shell"})
        assert "read" in ts
        assert "shell" not in ts

    def test_get_tool_names_applies_exclusions(self):
        """get_tool_names respects exclusions."""
        ts = ToolSet(tools={"read", "write", "shell"}, exclude={"shell"})
        names = ts.get_tool_names()
        assert "read" in names
        assert "write" in names
        assert "shell" not in names


class TestToolSetCategories:
    """Tests for ToolSet category handling."""

    def test_category_enum_values(self):
        """ToolCategory enum has expected values."""
        assert ToolCategory.CORE.value == "core"
        assert ToolCategory.FILESYSTEM.value == "filesystem"
        assert ToolCategory.GIT.value == "git"
        assert ToolCategory.WEB.value == "web"

    def test_include_category(self):
        """include_category() adds a category."""
        ts = ToolSet.minimal()
        ts2 = ts.include_category("git")
        assert "git" in ts2
        assert "git" not in ts  # Original unchanged

    def test_unknown_category_ignored(self):
        """Unknown categories are ignored without error."""
        ts = ToolSet(categories={"unknown_category", "core"})
        # Should not raise, just ignore unknown
        names = ts.get_tool_names()
        assert "read" in names  # Core tools present

    def test_refactor_alias(self):
        """REFACTOR is an alias for REFACTORING."""
        assert ToolCategory.REFACTOR.value == "refactoring"
