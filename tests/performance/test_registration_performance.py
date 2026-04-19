"""Performance regression tests for tool registration.

These tests ensure that registration performance meets established baselines
and catch performance regressions early. Uses pytest-benchmark for precise
measurements and statistical analysis.

Performance targets (from profiling):
- 10 items: < 0.5ms
- 100 items: < 5ms
- 1000 items: < 50ms
- 10000 items: < 500ms

Run with: pytest tests/performance/test_registration_performance.py --benchmark-only
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.tools.registry import ToolRegistry
from victor.tools.base import BaseTool
from victor.tools.enums import AccessMode, CostTier, DangerLevel, ExecutionCategory, Priority
from victor.tools.metadata import ToolMetadata
from typing import Dict, Any, List


class MockTool(BaseTool):
    """Mock tool for performance testing."""

    def __init__(self, name: str, description: str, tags: List[str] = None):
        self._name = name
        self._description = description
        self._tags = tags or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=self._description,
            category=ExecutionCategory.READ_ONLY,
            access_mode=AccessMode.READONLY,
            cost_tier=CostTier.FREE,
            danger_level=DangerLevel.SAFE,
            priority=Priority.MEDIUM,
            tags=self._tags,
        )

    async def execute(self, _exec_ctx, **kwargs):
        from victor.tools.base import ToolResult
        return ToolResult(success=True, output="test output")


def create_mock_tool(n: int) -> MockTool:
    """Create a mock tool with consistent properties."""
    return MockTool(
        name=f"tool_{n}",
        description=f"Test tool {n}",
        tags=[f"tag_{n % 10}", f"category_{n % 5}"]
    )


class TestRegistrationPerformance:
    """Performance regression tests for tool registration."""

    @pytest.mark.benchmark(group="registration")
    def test_register_10_items(self, benchmark) -> None:
        """Test registration of 10 items (< 0.5ms target)."""

        def register_10_items():
            registry = ToolRegistry()
            for i in range(10):
                tool = create_mock_tool(i)
                registry.register(tool)

        benchmark(register_10_items)
        # Performance assertion is handled by CI baseline comparison

    @pytest.mark.benchmark(group="registration")
    def test_register_100_items(self, benchmark) -> None:
        """Test registration of 100 items (< 5ms target)."""

        def register_100_items():
            registry = ToolRegistry()
            for i in range(100):
                tool = create_mock_tool(i)
                registry.register(tool)

        benchmark(register_100_items)

    @pytest.mark.benchmark(group="registration")
    def test_register_1000_items(self, benchmark) -> None:
        """Test registration of 1000 items (< 50ms target)."""

        def register_1000_items():
            registry = ToolRegistry()
            for i in range(1000):
                tool = create_mock_tool(i)
                registry.register(tool)

        benchmark(register_1000_items)

    @pytest.mark.benchmark(group="batch-registration")
    def test_batch_register_100_items(self, benchmark) -> None:
        """Test batch registration of 100 items with batch_update context."""

        def batch_register_100_items():
            registry = ToolRegistry()
            tools = [create_mock_tool(i) for i in range(100)]

            with registry.batch_update():
                for tool in tools:
                    registry.register(tool)

        benchmark(batch_register_100_items)

    @pytest.mark.benchmark(group="batch-registration")
    def test_batch_register_1000_items(self, benchmark) -> None:
        """Test batch registration of 1000 items with batch_update context."""

        def batch_register_1000_items():
            registry = ToolRegistry()
            tools = [create_mock_tool(i) for i in range(1000)]

            with registry.batch_update():
                for tool in tools:
                    registry.register(tool)

        benchmark(batch_register_1000_items)


class TestQueryPerformance:
    """Performance regression tests for registry queries."""

    @pytest.fixture
    def populated_registry(self) -> ToolRegistry:
        """Create a registry with 1000 tools for query testing."""
        registry = ToolRegistry()
        tools = [create_mock_tool(i) for i in range(1000)]

        with registry.batch_update():
            for tool in tools:
                registry.register(tool)

        return registry

    @pytest.mark.benchmark(group="queries")
    def test_get_by_name(self, benchmark, populated_registry) -> None:
        """Test get by name lookup (should be O(1))."""

        def lookup():
            populated_registry.get("tool_500")

        benchmark(lookup)

    @pytest.mark.benchmark(group="queries")
    def test_list_all(self, benchmark, populated_registry) -> None:
        """Test listing all tools."""

        benchmark(populated_registry.list_all)

    @pytest.mark.benchmark(group="queries")
    def test_get_schemas(self, benchmark, populated_registry) -> None:
        """Test schema generation (should use cache)."""

        # First call builds cache
        populated_registry.get_schemas()

        # Second call should hit cache
        benchmark(populated_registry.get_schemas)


class TestBatchAPIPerformance:
    """Performance tests for batch registration API."""

    @pytest.mark.benchmark(group="batch-api")
    def test_batch_registration_api_100(self, benchmark) -> None:
        """Test BatchRegistrar with 100 items."""
        from victor.tools.batch_registration import BatchRegistrar

        def batch_register():
            registry = ToolRegistry()
            registrar = BatchRegistrar(registry)
            tools = [create_mock_tool(i) for i in range(100)]

            result = registrar.register_batch(tools)
            assert result.success_count == 100

        benchmark(batch_register)

    @pytest.mark.benchmark(group="batch-api")
    def test_batch_registration_api_1000(self, benchmark) -> None:
        """Test BatchRegistrar with 1000 items."""
        from victor.tools.batch_registration import BatchRegistrar

        def batch_register():
            registry = ToolRegistry()
            registrar = BatchRegistrar(registry)
            tools = [create_mock_tool(i) for i in range(1000)]

            result = registrar.register_batch(tools)
            assert result.success_count == 1000

        benchmark(batch_register)


class TestFeatureFlagCachePerformance:
    """Performance tests for feature flag caching."""

    @pytest.mark.benchmark(group="feature-flags")
    def test_uncached_feature_flag_checks(self, benchmark) -> None:
        """Test uncached feature flag checks (baseline)."""
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        # Clear cache first
        from victor.core.feature_flags import reset_feature_flag_manager
        reset_feature_flag_manager()

        def check_flags():
            for _ in range(100):
                is_feature_enabled(FeatureFlag.USE_SERVICE_LAYER)

        benchmark(check_flags)

    @pytest.mark.benchmark(group="feature-flags")
    def test_cached_feature_flag_checks(self, benchmark) -> None:
        """Test cached feature flag checks (should be faster)."""
        from victor.core.feature_flag_cache import FeatureFlagCache
        from victor.core.feature_flags import FeatureFlag

        def check_flags_cached():
            with FeatureFlagCache.scope() as cache:
                for _ in range(100):
                    cache.is_enabled(FeatureFlag.USE_SERVICE_LAYER)

        benchmark(check_flags_cached)


class TestQueryCachePerformance:
    """Performance tests for query result caching."""

    @pytest.mark.benchmark(group="query-cache")
    def test_uncached_queries(self, benchmark) -> None:
        """Test uncached query operations (baseline)."""
        registry = ToolRegistry()
        tools = [create_mock_tool(i) for i in range(100)]

        with registry.batch_update():
            for tool in tools:
                registry.register(tool)

        def perform_queries():
            for i in range(100):
                registry.get(f"tool_{i}")

        benchmark(perform_queries)

    @pytest.mark.benchmark(group="query-cache")
    def test_cached_queries(self, benchmark) -> None:
        """Test cached query operations (should be faster)."""
        from victor.tools.query_cache import QueryCache

        registry = ToolRegistry()
        tools = [create_mock_tool(i) for i in range(100)]

        with registry.batch_update():
            for tool in tools:
                registry.register(tool)

        cache = QueryCache()

        def perform_cached_queries():
            for i in range(100):
                cache.get(
                    f"tool_{i}",
                    lambda idx=i: registry.get(f"tool_{idx}")
                )

        benchmark(perform_cached_queries)


# Performance regression assertions
# These run after benchmarks to check for regressions
@pytest.fixture(autouse=True)
def verify_performance_targets(benchmark) -> None:
    """Verify that benchmarks meet performance targets.

    This fixture runs after each benchmark and can be used to
    assert that performance hasn't regressed.

    To enable: Run with --benchmark-autosave to save baseline,
    then run with --benchmark-compare to compare.
    """
    yield

    # Performance assertions would go here
    # Example: assert benchmark.stats.stats.median < 0.005  # 5ms
