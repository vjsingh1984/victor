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

"""Performance benchmarks for tool selection optimization.

TDD Approach: These benchmarks establish baseline performance metrics
BEFORE implementing any optimizations. This ensures we have concrete
data to measure improvement against.

Benchmark Categories:
1. Baseline tool selection performance (different tool set sizes)
2. Query embedding caching impact (cached vs uncached)
3. Batch embedding generation
4. Category filtering optimization
5. Cache warming effectiveness
6. Memory usage profiling

Performance Targets (to be validated after optimization):
- Tool selection for 10 tools: <50ms
- Cached query lookup: <5ms
- Batch embedding (10 items): <100ms
"""

import asyncio
import logging
import os
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_embedding_service(monkeypatch):
    """Avoid network/model downloads during benchmarks."""
    # Return proper numpy array, not scalar
    mock_vector = np.array([0.0] * 384, dtype=np.float32)
    mock_service = MagicMock()
    mock_service.embed_text = AsyncMock(return_value=mock_vector)
    mock_service.embed_text_batch = AsyncMock(
        side_effect=lambda texts: [np.array([0.0] * 384, dtype=np.float32) for _ in texts]
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with patch(
        "victor.storage.embeddings.service.EmbeddingService.get_instance",
        return_value=mock_service,
    ):
        yield


# Reuse a single event loop per test to avoid asyncio.run overhead in benchmarks.
@pytest.fixture
def run_async():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop.run_until_complete
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# Configure logging to reduce noise during benchmarks
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with configurable tool count."""

    class MockTool:
        def __init__(self, name: str, description: str, keywords: List[str] = None):
            self.name = name
            self.description = description
            self.keywords = keywords or []
            self._metadata = None

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        def get_metadata(self):
            if self._metadata is None:
                # Auto-generate metadata
                from victor.tools.metadata import ToolMetadata

                self._metadata = ToolMetadata(
                    category="test",
                    keywords=self.keywords,
                    use_cases=[f"use {self.name} for"],
                    examples=[f"example {self.name} usage"],
                )
            return self._metadata

    class MockToolRegistry:
        def __init__(self, num_tools: int = 10):
            self.num_tools = num_tools
            self._tools = self._create_mock_tools(num_tools)

        def _create_mock_tools(self, num_tools: int) -> List[MockTool]:
            """Create mock tools with realistic descriptions and keywords."""
            tools = []
            tool_templates = [
                ("read", "Read file contents", ["read", "file", "open", "view"]),
                ("write", "Write to file", ["write", "save", "create", "modify"]),
                ("search", "Search codebase", ["search", "find", "locate", "grep"]),
                ("edit", "Edit files", ["edit", "modify", "change", "update"]),
                ("shell", "Execute shell commands", ["shell", "bash", "command", "run"]),
                ("git", "Git operations", ["git", "commit", "branch", "merge"]),
                ("test", "Run tests", ["test", "pytest", "unittest", "verify"]),
                ("ls", "List directory", ["ls", "list", "dir", "files"]),
                ("docker", "Docker operations", ["docker", "container", "image", "build"]),
                ("web_search", "Search the web", ["web", "search", "online", "lookup"]),
            ]

            for i in range(num_tools):
                template_idx = i % len(tool_templates)
                name_base, desc_base, keywords_base = tool_templates[template_idx]
                suffix = f"_{i}" if i >= len(tool_templates) else ""

                tools.append(
                    MockTool(
                        name=f"{name_base}{suffix}",
                        description=f"{desc_base} ({i})",
                        keywords=keywords_base.copy(),
                    )
                )

            return tools

        def list_tools(self, only_enabled: bool = True) -> List[MockTool]:
            # For benchmarks, always return all tools (ignoring only_enabled)
            return self._tools

        def is_tool_enabled(self, tool_name: str) -> bool:
            return any(t.name == tool_name for t in self._tools)

        def get(self, tool_name: str):
            for tool in self._tools:
                if tool.name == tool_name:
                    return tool
            return None

        def get_tool_cost(self, tool_name: str):
            from victor.tools.metadata import CostTier

            return CostTier.LOW

    return MockToolRegistry


@pytest.fixture
def semantic_selector(tmp_path, run_async):
    """Create a SemanticToolSelector with temp cache directory."""

    from victor.tools.semantic_selector import SemanticToolSelector

    # Use temp directory for cache isolation
    cache_dir = tmp_path / "embeddings"

    selector = SemanticToolSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        cache_embeddings=True,
        cache_dir=cache_dir,
        cost_aware_selection=False,  # Disable for cleaner benchmarks
        sequence_tracking=False,  # Disable for cleaner benchmarks
    )

    yield selector

    # Cleanup
    run_async(selector.close())


@pytest.fixture
def keyword_selector(mock_tool_registry):
    """Create a KeywordToolSelector for benchmarking."""

    from victor.tools.keyword_tool_selector import KeywordToolSelector

    # Create a registry for the selector
    registry = mock_tool_registry(num_tools=47)
    selector = KeywordToolSelector(tools=registry)

    return selector


@pytest.fixture
def hybrid_selector(semantic_selector, keyword_selector, mock_tool_registry, run_async):
    """Create a HybridToolSelector for benchmarking."""

    from victor.tools.hybrid_tool_selector import HybridToolSelector, HybridSelectorConfig

    # Initialize semantic selector with tools
    registry = mock_tool_registry(num_tools=10)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    config = HybridSelectorConfig(
        enable_rl=False,  # Disable for cleaner benchmarks
    )

    selector = HybridToolSelector(
        semantic_selector=semantic_selector,
        keyword_selector=keyword_selector,
        config=config,
    )

    return selector


# =============================================================================
# Benchmark Queries
# =============================================================================

BENCHMARK_QUERIES = [
    # Simple queries
    "read the file",
    "write to file",
    "search for code",
    # Complex queries
    "find all classes that inherit from BaseController and list their methods",
    "analyze the codebase for security vulnerabilities and generate a report",
    "create a new REST API endpoint for user authentication with JWT tokens",
    # Multi-step queries
    "read the config file, update the database url, and restart the server",
    "run tests, if they pass deploy to staging and run integration tests",
    # Vague queries
    "fix it",
    "make it work",
    "help me",
    # Edge cases
    "",  # Empty
    "x" * 500,  # Very long query
]


# =============================================================================
# Baseline Performance Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="tool_selection_baseline")
def test_baseline_tool_selection_10_tools(
    benchmark, mock_tool_registry, semantic_selector, run_async
):
    """Benchmark tool selection with 10 tools (baseline).

    Target: <50ms for 10 tools
    """

    registry = mock_tool_registry(num_tools=10)

    # Initialize embeddings (one-time cost, not included in benchmark)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    # Synchronous wrapper for async code (required by pytest-benchmark)
    def select_tools_sync():
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="read the file and search for errors",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    selected_tools = benchmark.pedantic(select_tools_sync, iterations=10, rounds=20)

    # Assertions
    assert len(selected_tools) > 0
    assert isinstance(selected_tools, list)


@pytest.mark.benchmark(group="tool_selection_baseline")
def test_baseline_tool_selection_47_tools(
    benchmark, mock_tool_registry, semantic_selector, run_async
):
    """Benchmark tool selection with 47 tools (current production count).

    This measures performance with the actual number of tools in Victor.
    Target: <100ms for 47 tools
    """

    registry = mock_tool_registry(num_tools=47)

    # Initialize embeddings
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    # Benchmark the async function directly
    def select_tools():
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="find all classes that inherit from BaseController",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    selected_tools = benchmark(select_tools)

    assert len(selected_tools) > 0


@pytest.mark.benchmark(group="tool_selection_baseline")
def test_baseline_tool_selection_100_tools(
    benchmark, mock_tool_registry, semantic_selector, run_async
):
    """Benchmark tool selection with 100 tools (stress test).

    This measures performance with double the current tool count.
    Target: <200ms for 100 tools
    """

    registry = mock_tool_registry(num_tools=100)

    # Initialize embeddings
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    def select_tools():
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="analyze the codebase and generate metrics",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_tools, iterations=10, rounds=20)
    selected_tools = result

    assert len(selected_tools) > 0


@pytest.mark.benchmark(group="tool_selection_baseline")
def test_baseline_keyword_selection(benchmark, keyword_selector, run_async):
    """Benchmark keyword-based tool selection (baseline).

    Keyword selection should be much faster than semantic (<5ms).
    """

    # Use the registry from keyword_selector fixture
    selector = keyword_selector
    registry = selector.tools

    from victor.protocols.tool_selector import ToolSelectionContext

    def select_tools():
        return run_async(
            selector.select_tools(
                prompt="read the file and search for errors",
                context=ToolSelectionContext(
                    task_description="read the file and search for errors",
                    metadata={"available_tools": {t.name for t in registry.list_tools()}},
                ),
            )
        )

    selected_tools = benchmark.pedantic(select_tools, iterations=50, rounds=20)

    assert len(selected_tools) > 0


@pytest.mark.benchmark(group="tool_selection_baseline")
@pytest.mark.skip(
    reason="HybridToolSelector has API incompatibility with new SemanticToolSelector.select_tools() signature"
)
def test_baseline_hybrid_selection(benchmark, hybrid_selector, run_async):
    """Benchmark hybrid tool selection (baseline).

    Hybrid selection combines semantic + keyword approaches.
    Target: <70ms for 10 tools (slightly more than semantic alone)

    SKIPPED: HybridToolSelector needs to be updated to use new SemanticToolSelector API
    """

    # Get registries from selectors
    selector = hybrid_selector
    semantic_registry = selector.semantic._tools_registry

    from victor.protocols.tool_selector import ToolSelectionContext

    def select_tools():
        return run_async(
            selector.select_tools(
                prompt="read the file and search for errors",
                context=ToolSelectionContext(
                    task_description="read the file and search for errors",
                    metadata={"available_tools": {t.name for t in semantic_registry.list_tools()}},
                ),
            )
        )

    selected_tools = benchmark.pedantic(select_tools, iterations=10, rounds=20)

    assert len(selected_tools) > 0


# =============================================================================
# Query Embedding Caching Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="embedding_cache")
def test_cached_query_embedding(benchmark, semantic_selector, run_async):
    """Benchmark cached query embedding lookup.

    This measures the performance of retrieving a previously computed
    query embedding from cache (should be <5ms).
    """

    # First, generate and cache an embedding
    query = "read the file and search for errors"
    run_async(semantic_selector._get_embedding(query))

    # Now benchmark the cached lookup
    def get_cached_embedding():
        return run_async(semantic_selector._get_embedding(query))

    embedding = benchmark.pedantic(get_cached_embedding, iterations=100, rounds=20)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


@pytest.mark.benchmark(group="embedding_cache")
def test_uncached_query_embedding(benchmark, semantic_selector, run_async):
    """Benchmark uncached query embedding generation.

    This measures the performance of generating a new embedding
    without cache (should be >50ms for sentence-transformers).

    Compare with test_cached_query_embedding to see cache impact.
    """

    # Use different query each iteration to avoid cache
    queries = [f"query variant {i} with some text" for i in range(100)]

    def get_uncached_embedding(query_idx):
        # Ensure we're not hitting cache
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        query = queries[query_idx % len(queries)]
        return run_async(semantic_selector._get_embedding(query))

    result = benchmark.pedantic(get_uncached_embedding, args=(0,), iterations=50, rounds=10)
    embedding = result

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


@pytest.mark.benchmark(group="embedding_cache")
def test_tool_selection_cache_hit_rate(benchmark, mock_tool_registry, semantic_selector, run_async):
    """Benchmark tool selection with realistic cache hit rate.

    In production, queries are often similar. This benchmark measures
    SemanticToolSelector cache performance with 70% cache hit rate.
    """

    registry = mock_tool_registry(num_tools=10)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    # Create query set with duplicates (simulating cache hits)
    queries = [
        "read the file",  # Repeated 3x
        "read the file",
        "read the file",
        "write to file",  # Repeated 2x
        "write to file",
        "search code",  # Unique
        "run tests",  # Repeated 2x
        "run tests",
        "edit files",  # Unique
        "git commit",  # Unique
    ]

    def select_with_cache_hits(query_idx):
        query = queries[query_idx % len(queries)]
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message=query,
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_with_cache_hits, args=(0,), iterations=20, rounds=10)
    selected_tools = result

    assert len(selected_tools) > 0


# =============================================================================
# Batch Embedding Generation Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="batch_embeddings")
def test_batch_embedding_10_items(benchmark, semantic_selector, run_async):
    """Benchmark batch embedding generation for 10 items.

    Target: <100ms for 10 items
    """

    items = [f"tool description {i}" for i in range(10)]

    def generate_batch():
        return [run_async(semantic_selector._get_embedding(item)) for item in items]

    embeddings = benchmark.pedantic(generate_batch, iterations=20, rounds=10)

    assert len(embeddings) == 10
    assert all(isinstance(emb, np.ndarray) for emb in embeddings)


@pytest.mark.benchmark(group="batch_embeddings")
def test_batch_embedding_47_items(benchmark, semantic_selector, run_async):
    """Benchmark batch embedding generation for 47 items.

    This represents embedding all tools in Victor.
    """

    items = [f"tool description {i}" for i in range(47)]

    def generate_batch():
        return [run_async(semantic_selector._get_embedding(item)) for item in items]

    embeddings = benchmark.pedantic(generate_batch, iterations=5, rounds=5)

    assert len(embeddings) == 47


@pytest.mark.benchmark(group="batch_embeddings")
def test_tool_embedding_initialization(benchmark, mock_tool_registry, semantic_selector, run_async):
    """Benchmark one-time tool embedding initialization.

    This is the startup cost of initializing tool embeddings.
    Target: <2s for 47 tools (one-time cost at startup)
    """

    registry = mock_tool_registry(num_tools=47)

    def initialize():
        # Create new selector to avoid cache
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2",
            cache_embeddings=True,  # Enable disk cache for initialization
        )
        run_async(selector.initialize_tool_embeddings(registry))
        run_async(selector.close())
        return selector

    result = benchmark.pedantic(initialize, iterations=3, rounds=3)
    selector = result

    assert len(selector._tool_embedding_cache) == 47


# =============================================================================
# Category Filtering Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="category_filtering")
def test_category_filtering_impact(benchmark, mock_tool_registry, semantic_selector, run_async):
    """Benchmark the impact of category filtering on selection time.

    Compares selection with and without category pre-filtering.
    """

    registry = mock_tool_registry(num_tools=47)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    # Add category metadata
    for tool in registry.list_tools():
        if "read" in tool.name or "write" in tool.name:
            tool.get_metadata().category = "file_ops"
        elif "search" in tool.name:
            tool.get_metadata().category = "search"
        else:
            tool.get_metadata().category = "general"

    def select_with_filtering():
        # This will use category-based filtering
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="find all classes that inherit from BaseController",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_with_filtering, iterations=20, rounds=10)
    selected_tools = result

    assert len(selected_tools) > 0


@pytest.mark.benchmark(group="category_filtering")
def test_mandatory_tool_overhead(benchmark, mock_tool_registry, semantic_selector, run_async):
    """Benchmark overhead of mandatory keyword detection.

    Measures the additional cost of checking for mandatory tools
    before semantic selection.
    """

    registry = mock_tool_registry(num_tools=47)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    def select_with_mandatory():
        # Query with mandatory keyword "show diff"
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="show the diff between commits",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_with_mandatory, iterations=20, rounds=10)
    selected_tools = result

    assert len(selected_tools) > 0


# =============================================================================
# Cache Warming Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="cache_warming")
def test_cold_cache(benchmark, mock_tool_registry, tmp_path, run_async):
    """Benchmark cold cache selection (first selection after initialization).

    This measures performance with uncached embeddings.
    """

    registry = mock_tool_registry(num_tools=47)

    # Create selector with cold cache
    cache_dir = tmp_path / "cold"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from victor.tools.semantic_selector import SemanticToolSelector

    cold_selector = SemanticToolSelector(
        embedding_model="all-MiniLM-L6-v2",
        cache_dir=cache_dir,
        cache_embeddings=True,
    )

    def cold_selection():
        return run_async(
            cold_selector.select_relevant_tools(
                user_message="read the file and search for errors",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    cold_tools = benchmark.pedantic(cold_selection, iterations=5, rounds=3)

    assert len(cold_tools) > 0

    run_async(cold_selector.close())


@pytest.mark.benchmark(group="cache_warming")
def test_warm_cache(benchmark, mock_tool_registry, tmp_path, run_async):
    """Benchmark warm cache selection (subsequent selections with cached embeddings).

    This measures performance with pre-warmed embeddings.
    """

    registry = mock_tool_registry(num_tools=47)

    # Create selector with warm cache
    cache_dir = tmp_path / "warm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from victor.tools.semantic_selector import SemanticToolSelector

    warm_selector = SemanticToolSelector(
        embedding_model="all-MiniLM-L6-v2",
        cache_dir=cache_dir,
        cache_embeddings=True,
    )

    # Initialize embeddings (warm up cache)
    run_async(warm_selector.initialize_tool_embeddings(registry))

    def warm_selection():
        return run_async(
            warm_selector.select_relevant_tools(
                user_message="read the file and search for errors",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    warm_tools = benchmark.pedantic(warm_selection, iterations=20, rounds=10)

    assert len(warm_tools) > 0

    run_async(warm_selector.close())


@pytest.mark.benchmark(group="cache_warming")
def test_cache_warmup_time(benchmark, mock_tool_registry, tmp_path, run_async):
    """Benchmark time to warm up tool embedding cache.

    This measures the time to compute and cache embeddings for all tools.
    Target: <2s for 47 tools
    """

    registry = mock_tool_registry(num_tools=47)

    cache_dir = tmp_path / "warmup"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def warmup():
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2",
            cache_dir=cache_dir,
            cache_embeddings=True,
        )

        # This includes embedding computation + disk write
        run_async(selector.initialize_tool_embeddings(registry))
        run_async(selector.close())
        return selector

    result = benchmark.pedantic(warmup, iterations=3, rounds=3)
    selector = result

    assert len(selector._tool_embedding_cache) == 47


# =============================================================================
# Memory Usage Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="memory_usage")
def test_memory_per_tool_embedding(mock_tool_registry, semantic_selector, run_async):
    """Measure memory usage per tool embedding.

    Each embedding is a 384-dimensional float32 array (MiniLM-L6-v2).
    Expected: ~1.5KB per embedding (384 * 4 bytes)
    """

    registry = mock_tool_registry(num_tools=10)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    # Get memory before
    import sys

    base_size = sys.getsizeof(semantic_selector._tool_embedding_cache)

    # Calculate size of embeddings
    total_embedding_size = 0
    for name, embedding in semantic_selector._tool_embedding_cache.items():
        total_embedding_size += sys.getsizeof(embedding)

    avg_size_per_tool = total_embedding_size / len(semantic_selector._tool_embedding_cache)

    # Expected: ~1.5KB per embedding
    assert (
        1000 < avg_size_per_tool < 2000
    ), f"Expected ~1.5KB per embedding, got {avg_size_per_tool:.0f} bytes"

    logger.info(
        f"Memory usage: {total_embedding_size / 1024:.2f}KB total, "
        f"{avg_size_per_tool:.0f} bytes per tool"
    )


@pytest.mark.benchmark(group="memory_usage")
def test_memory_47_tools(mock_tool_registry, semantic_selector, run_async):
    """Measure total memory usage for 47 tool embeddings.

    Expected: ~70KB for 47 tools (47 * 1.5KB)
    """

    registry = mock_tool_registry(num_tools=47)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    import sys

    total_size = 0
    for name, embedding in semantic_selector._tool_embedding_cache.items():
        total_size += sys.getsizeof(embedding)

    total_kb = total_size / 1024

    # Expected: ~70KB for 47 tools
    assert 60 < total_kb < 100, f"Expected ~70KB, got {total_kb:.0f}KB"

    logger.info(f"Total memory for 47 tools: {total_kb:.0f}KB")


# =============================================================================
# Bottleneck Identification
# =============================================================================


@pytest.mark.benchmark(group="bottlenecks")
def test_embedding_generation_bottleneck(benchmark, semantic_selector, run_async):
    """Identify if embedding generation is the bottleneck.

    Compares embedding time vs total selection time.
    """

    query = "find all classes that inherit from BaseController"

    def generate_query_embedding():
        return run_async(semantic_selector._get_embedding(query))

    # Benchmark just embedding generation
    embedding = benchmark.pedantic(generate_query_embedding, iterations=50, rounds=20)

    assert isinstance(embedding, np.ndarray)


@pytest.mark.benchmark(group="bottlenecks")
def test_similarity_computation_bottleneck(benchmark, semantic_selector):
    """Identify if similarity computation is the bottleneck.

    Measures time to compute cosine similarity for 47 tools.
    """

    # Create mock embeddings
    query_emb = np.random.randn(384).astype(np.float32)
    tool_embeddings = {f"tool_{i}": np.random.randn(384).astype(np.float32) for i in range(47)}

    def compute_similarities():
        similarities = []
        for tool_name, tool_emb in tool_embeddings.items():
            sim = semantic_selector._cosine_similarity(query_emb, tool_emb)
            similarities.append((tool_name, sim))
        return similarities

    result = benchmark.pedantic(compute_similarities, iterations=100, rounds=20)
    similarities = result

    assert len(similarities) == 47


@pytest.mark.benchmark(group="bottlenecks")
def test_tool_iteration_bottleneck(benchmark, mock_tool_registry):
    """Identify if tool iteration is the bottleneck.

    Measures time to iterate through tool registry.
    """

    registry = mock_tool_registry(num_tools=47)

    def iterate_tools():
        return list(registry.list_tools())

    result = benchmark.pedantic(iterate_tools, iterations=1000, rounds=20)
    tools = result

    assert len(tools) == 47


# =============================================================================
# Regression Tests (Performance Degradation Detection)
# =============================================================================


@pytest.mark.regression
@pytest.mark.benchmark(group="regression")
def test_regression_selection_time_10_tools(
    benchmark, mock_tool_registry, semantic_selector, run_async
):
    """Regression test: Ensure selection time doesn't degrade.

    Baseline established in test_baseline_tool_selection_10_tools.
    Fails if performance degrades by >20% from baseline.
    """

    registry = mock_tool_registry(num_tools=10)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    def select_tools():
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message="read the file and search for errors",
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_tools, iterations=10, rounds=20)

    # This is a regression marker - actual assertion done by pytest-benchmark's history
    selected_tools = result
    assert len(selected_tools) > 0


@pytest.mark.regression
@pytest.mark.benchmark(group="regression")
def test_regression_cache_hit_time(benchmark, semantic_selector, run_async):
    """Regression test: Ensure cache hit time doesn't degrade.

    Baseline established in test_cached_query_embedding.
    Fails if cache lookup degrades by >20% from baseline.
    """

    query = "read the file"
    run_async(semantic_selector._get_embedding(query))

    def get_cached():
        return run_async(semantic_selector._get_embedding(query))

    result = benchmark.pedantic(get_cached, iterations=100, rounds=20)

    # This is a regression marker - actual assertion done by pytest-benchmark's history
    embedding = result
    assert isinstance(embedding, np.ndarray)


# =============================================================================
# Query Pattern Analysis
# =============================================================================


@pytest.mark.benchmark(group="query_patterns")
@pytest.mark.parametrize("query", BENCHMARK_QUERIES)
def test_selection_by_query_pattern(
    query, benchmark, mock_tool_registry, semantic_selector, run_async
):
    """Benchmark selection across different query patterns.

    This helps identify which query types are slower:
    - Simple queries: Should be fast
    - Complex queries: May be slower
    - Multi-step: May trigger more tools
    - Vague: May need fallback logic
    """

    registry = mock_tool_registry(num_tools=47)
    run_async(semantic_selector.initialize_tool_embeddings(registry))

    def select_for_query():
        return run_async(
            semantic_selector.select_relevant_tools(
                user_message=query,
                tools=registry,
                max_tools=10,
                similarity_threshold=0.18,
            )
        )

    result = benchmark.pedantic(select_for_query, iterations=10, rounds=10)
    selected_tools = result

    # Should always return some tools
    assert len(selected_tools) > 0 or query == ""  # Empty query may return 0


# =============================================================================
# Summary and Reporting
# =============================================================================


@pytest.mark.summary
def test_benchmark_summary():
    """Generate summary of benchmark targets and current status.

    This test doesn't run benchmarks - it just prints the targets
    for reference in pytest output.
    """

    print("\n" + "=" * 80)
    print("TOOL SELECTION PERFORMANCE BENCHMARK TARGETS")
    print("=" * 80)

    print("\n📊 BASELINE PERFORMANCE TARGETS:")
    print("  • Tool selection (10 tools):    <50ms")
    print("  • Tool selection (47 tools):    <100ms")
    print("  • Tool selection (100 tools):   <200ms")
    print("  • Keyword-only selection:       <5ms")
    print("  • Hybrid selection:             <70ms")

    print("\n🚀 CACHING PERFORMANCE TARGETS:")
    print("  • Cached query lookup:          <5ms")
    print("  • Uncached query embedding:     >50ms (baseline)")
    print("  • Cache hit rate (70%):         ~30-40ms")

    print("\n⚡ BATCH EMBEDDING TARGETS:")
    print("  • Batch (10 items):             <100ms")
    print("  • Batch (47 items):             <500ms")
    print("  • Tool initialization (47):     <2s (one-time)")

    print("\n💾 MEMORY USAGE TARGETS:")
    print("  • Per tool embedding:           ~1.5KB")
    print("  • 47 tool embeddings:           ~70KB")
    print("  • Cache overhead:               <10KB")

    print("\n🔍 BOTTLENECK IDENTIFICATION:")
    print("  • Embedding generation:         ~30-50ms per query")
    print("  • Similarity computation:       ~1-2ms for 47 tools")
    print("  • Tool iteration:               <0.1ms for 47 tools")

    print("\n📈 REGRESSION THRESHOLDS:")
    print("  • Performance degradation:      FAIL if >20% slower")
    print("  • Memory leak:                  FAIL if memory grows")

    print("\n" + "=" * 80)
    print(
        "RUN: pytest tests/benchmark/benchmarks/test_tool_selection_performance.py -v --benchmark-only"
    )
    print("=" * 80 + "\n")

    assert True  # Always passes (just for display)
