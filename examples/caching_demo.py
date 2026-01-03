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

"""Demo of Victor's Caching System.

Demonstrates the tiered caching architecture:
- L1: Memory cache (fast, short-lived)
- L2: Disk cache (persistent, longer-lived)

Usage:
    python examples/caching_demo.py
"""

import time
from victor.storage.cache import CacheConfig, CacheManager
from victor.storage.cache.tiered_cache import EmbeddingCache, ResponseCache


def demo_basic_caching():
    """Demo basic cache operations."""
    print("\n💾 Basic Caching Demo")
    print("=" * 70)

    # Create cache with custom config
    config = CacheConfig(
        memory_max_size=100,
        memory_ttl=60,  # 1 minute
        disk_ttl=3600,  # 1 hour
    )
    cache = CacheManager(config)

    print("\n1️⃣ Store values in cache...")
    cache.set("user_123", {"name": "Alice", "email": "alice@example.com"})
    cache.set("user_456", {"name": "Bob", "email": "bob@example.com"})
    cache.set("config", {"theme": "dark", "language": "en"})
    print("✓ Stored 3 values")

    print("\n2️⃣ Retrieve from cache...")
    user = cache.get("user_123")
    print(f"Retrieved user: {user}")

    print("\n3️⃣ Cache statistics...")
    stats = cache.get_stats()
    print(f"Memory hits: {stats['memory_hits']}")
    print(f"Memory misses: {stats['memory_misses']}")
    print(f"Hit rate: {stats['memory_hit_rate']:.2%}")

    print("\n4️⃣ Cache with namespaces...")
    cache.set("api_key", "secret123", namespace="credentials")
    cache.set("db_url", "postgresql://...", namespace="credentials")
    cache.set("token", "abc123", namespace="session")

    cred = cache.get("api_key", namespace="credentials")
    print(f"Retrieved credential: {cred}")

    print("\n5️⃣ Delete from cache...")
    cache.delete("user_123")
    deleted_user = cache.get("user_123")
    print(f"After delete: {deleted_user}")  # Should be None

    print("\n6️⃣ Clear namespace...")
    cache.clear(namespace="credentials")
    print("✓ Cleared credentials namespace")

    cache.close()


def demo_tiered_caching():
    """Demo L1 (memory) and L2 (disk) tiering."""
    print("\n\n🏗️  Tiered Caching Demo")
    print("=" * 70)

    cache = CacheManager()

    print("\n1️⃣ Store in both caches...")
    cache.set("expensive_data", {"result": "computed value", "cost": 100})
    print("✓ Stored in memory (L1) and disk (L2)")

    print("\n2️⃣ Retrieve from memory (fast)...")
    start = time.time()
    data = cache.get("expensive_data")
    elapsed_ms = (time.time() - start) * 1000
    print(f"Retrieved in {elapsed_ms:.3f}ms from memory")
    print(f"Data: {data}")

    print("\n3️⃣ Simulate memory cache clear...")
    # Clear only memory cache to test disk fallback
    if cache._memory_cache:
        cache._memory_cache.clear()
    print("✓ Cleared memory cache")

    print("\n4️⃣ Retrieve from disk (slower but persistent)...")
    start = time.time()
    data = cache.get("expensive_data")  # Will hit disk cache
    elapsed_ms = (time.time() - start) * 1000
    print(f"Retrieved in {elapsed_ms:.3f}ms from disk")
    print(f"Data: {data}")
    print("✓ Automatically promoted back to memory cache")

    print("\n5️⃣ Final statistics...")
    stats = cache.get_stats()
    print(f"Memory hits: {stats['memory_hits']}")
    print(f"Disk hits: {stats['disk_hits']}")
    print(f"Total sets: {stats['sets']}")

    cache.close()


def demo_response_cache():
    """Demo LLM response caching."""
    print("\n\n🤖 Response Caching Demo")
    print("=" * 70)

    response_cache = ResponseCache()

    print("\n1️⃣ Simulate expensive LLM call...")
    prompt = "Write a function to calculate Fibonacci"
    model = "claude-sonnet-4-5"
    temperature = 0.7

    # Simulate LLM response
    simulated_response = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

    # Cache the response
    response_cache.cache_response(prompt, model, temperature, simulated_response)
    print("✓ Cached LLM response")

    print("\n2️⃣ Retrieve cached response (instant)...")
    start = time.time()
    cached = response_cache.get_response(prompt, model, temperature)
    elapsed_ms = (time.time() - start) * 1000
    print(f"Retrieved in {elapsed_ms:.3f}ms (no API call!)")
    print(f"Response:\n{cached}")

    print("\n3️⃣ Different temperature = cache miss...")
    different_temp = response_cache.get_response(prompt, model, 1.0)
    print(f"Different temperature result: {different_temp}")  # Should be None

    print("\n💰 Cost Savings:")
    print("  - First call: ~$0.015 (API cost)")
    print("  - Cached calls: $0.000 (FREE!)")
    print("  - 100 cached calls = $1.50 saved")

    response_cache.cache.close()


def demo_embedding_cache():
    """Demo embedding caching."""
    print("\n\n🔍 Embedding Caching Demo")
    print("=" * 70)

    embedding_cache = EmbeddingCache()

    print("\n1️⃣ Simulate embedding computation...")
    text = "Victor is an AI coding assistant"
    model = "all-MiniLM-L12-v2"  # Local sentence-transformers model

    # Simulate embedding vector (normally from API)
    simulated_embedding = [0.123, -0.456, 0.789] * 100  # 300-dim vector

    embedding_cache.cache_embedding(text, model, simulated_embedding)
    print(f"✓ Cached embedding ({len(simulated_embedding)} dimensions)")

    print("\n2️⃣ Retrieve cached embedding...")
    start = time.time()
    cached_emb = embedding_cache.get_embedding(text, model)
    elapsed_ms = (time.time() - start) * 1000
    print(f"Retrieved in {elapsed_ms:.3f}ms")
    print(f"Vector length: {len(cached_emb) if cached_emb else 0}")

    print("\n3️⃣ Different text = cache miss...")
    different_text = "Different text here"
    result = embedding_cache.get_embedding(different_text, model)
    print(f"Different text result: {result}")

    print("\n💰 Performance Impact:")
    print("  - Embedding API call: ~100ms")
    print("  - Cached retrieval: <1ms")
    print("  - 100x faster for repeated queries!")

    embedding_cache.cache.close()


def demo_cache_warmup():
    """Demo cache warmup."""
    print("\n\n🔥 Cache Warmup Demo")
    print("=" * 70)

    cache = CacheManager()

    print("\n1️⃣ Prepare warmup data...")
    warmup_data = {
        "config_theme": "dark",
        "config_language": "en",
        "config_auto_save": True,
        "user_preferences": {"notifications": True},
        "recent_files": ["/path/to/file1.py", "/path/to/file2.py"],
    }

    print("\n2️⃣ Warm up cache...")
    count = cache.warmup(warmup_data, namespace="app")
    print(f"✓ Warmed up cache with {count} entries")

    print("\n3️⃣ Verify warmup...")
    theme = cache.get("config_theme", namespace="app")
    prefs = cache.get("user_preferences", namespace="app")
    print(f"Theme: {theme}")
    print(f"Preferences: {prefs}")

    print("\n✅ Benefits of Cache Warmup:")
    print("  - Instant access to common data")
    print("  - Reduced cold start latency")
    print("  - Better user experience")

    cache.close()


def demo_cache_stats():
    """Demo comprehensive cache statistics."""
    print("\n\n📊 Cache Statistics Demo")
    print("=" * 70)

    cache = CacheManager()

    # Generate some activity
    print("\n1️⃣ Generate cache activity...")
    for i in range(50):
        cache.set(f"key_{i}", f"value_{i}")

    # Mix of hits and misses
    for i in range(100):
        cache.get(f"key_{i % 50}")  # Will hit for first 50, miss for rest

    print("\n2️⃣ Cache statistics...")
    stats = cache.get_stats()

    print("\nMemory Cache:")
    print(f"  Hits: {stats['memory_hits']}")
    print(f"  Misses: {stats['memory_misses']}")
    print(f"  Hit Rate: {stats['memory_hit_rate']:.2%}")
    print(f"  Current Size: {stats.get('memory_size', 0)}/{stats.get('memory_max_size', 0)}")

    print("\nDisk Cache:")
    print(f"  Hits: {stats['disk_hits']}")
    print(f"  Misses: {stats['disk_misses']}")
    print(f"  Hit Rate: {stats['disk_hit_rate']:.2%}")
    print(f"  Entries: {stats.get('disk_size', 0)}")

    print("\nOverall:")
    print(f"  Total Sets: {stats['sets']}")
    print(f"  Total Requests: {stats['memory_hits'] + stats['memory_misses']}")

    cache.close()


def main():
    """Run all caching demos."""
    print("🎯 Victor Caching System Demo")
    print("=" * 70)
    print("\nDemonstrating tiered caching architecture\n")

    # Run demos
    demo_basic_caching()
    demo_tiered_caching()
    demo_response_cache()
    demo_embedding_cache()
    demo_cache_warmup()
    demo_cache_stats()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Caching System provides:")
    print("  • Tiered architecture (memory + disk)")
    print("  • Zero external dependencies")
    print("  • Automatic persistence")
    print("  • Thread-safe operations")
    print("  • Specialized caches (responses, embeddings)")
    print("  • Cost savings through response caching")
    print("  • Performance gains (100x+ for embeddings)")
    print("\nReady for production use!")


if __name__ == "__main__":
    main()
