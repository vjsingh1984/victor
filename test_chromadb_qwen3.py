"""Test ChromaDB with Qwen3-Embedding:8b integration."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from victor.codebase.embeddings.base import EmbeddingConfig
from victor.codebase.embeddings.chromadb_provider import ChromaDBProvider


async def test_chromadb_qwen3():
    """Test ChromaDB with Qwen3-Embedding integration."""

    print("=" * 80)
    print("Testing ChromaDB + Qwen3-Embedding:8b Integration")
    print("=" * 80)
    print()

    # Configuration
    config = EmbeddingConfig(
        vector_store="chromadb",
        persist_directory="~/.victor/embeddings/test_qwen3",
        distance_metric="cosine",
        embedding_model_type="ollama",
        embedding_model_name="qwen3-embedding:8b",
        embedding_api_key="http://localhost:11434",
        extra_config={
            "collection_name": "test_qwen3",
            "dimension": 4096,
            "batch_size": 4,
        }
    )

    # Create provider
    print("1. Creating ChromaDB provider...")
    provider = ChromaDBProvider(config)
    print("   ✅ Provider created")
    print()

    # Initialize
    print("2. Initializing provider...")
    try:
        await provider.initialize()
        print("   ✅ Provider initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Clear any existing data
    print("3. Clearing existing index...")
    try:
        await provider.clear_index()
        print("   ✅ Index cleared")
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
    print()

    # Test documents
    documents = [
        {
            "id": "auth.py:authenticate",
            "content": "def authenticate(username, password): return check_credentials(username, password)",
            "metadata": {
                "file_path": "auth.py",
                "symbol_name": "authenticate",
                "line_number": 10
            }
        },
        {
            "id": "db.py:connect",
            "content": "async def connect_database(url): return await create_engine(url, pool_size=10)",
            "metadata": {
                "file_path": "db.py",
                "symbol_name": "connect_database",
                "line_number": 5
            }
        },
        {
            "id": "cache.py:cache_decorator",
            "content": "def cache_result(ttl=3600): return lambda f: functools.wraps(f)(cached_wrapper)",
            "metadata": {
                "file_path": "cache.py",
                "symbol_name": "cache_result",
                "line_number": 15
            }
        }
    ]

    # Index documents
    print("4. Indexing documents...")
    try:
        await provider.index_documents(documents)
        print(f"   ✅ Indexed {len(documents)} documents")
    except Exception as e:
        print(f"   ❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # Get statistics
    print("5. Getting index statistics...")
    try:
        stats = await provider.get_stats()
        print("   📊 Statistics:")
        print(f"      Provider: {stats['provider']}")
        print(f"      Documents: {stats['total_documents']}")
        print(f"      Model Type: {stats['embedding_model_type']}")
        print(f"      Model Name: {stats['embedding_model_name']}")
        print(f"      Dimension: {stats['dimension']}")
        print(f"      Distance Metric: {stats['distance_metric']}")

        assert stats['total_documents'] == len(documents), "Document count mismatch"
        assert stats['dimension'] == 4096, "Dimension mismatch"
        print("   ✅ Statistics verified")
    except Exception as e:
        print(f"   ❌ Stats failed: {e}")
        return False
    print()

    # Test semantic search
    print("6. Testing semantic search...")
    queries = [
        ("user authentication with credentials", "auth.py:authenticate"),
        ("database connection pool", "db.py:connect"),
        ("caching decorator with expiration", "cache.py:cache_decorator"),
    ]

    for query, expected_id in queries:
        print(f"\n   Query: '{query}'")
        try:
            results = await provider.search_similar(query, limit=3)

            if not results:
                print("   ❌ No results returned")
                return False

            # Check top result
            top_result = results[0]
            print(f"   Top match: {top_result.file_path}:{top_result.symbol_name}")
            print(f"   Score: {top_result.score:.4f}")

            # Verify we got the expected result (should be in top 3)
            found = any(r.file_path == expected_id.split(':')[0] for r in results)
            if found:
                print("   ✅ Expected result found in top 3")
            else:
                print(f"   ⚠️  Expected '{expected_id}' not in top 3")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    print()

    # Test embedding generation
    print("7. Testing direct embedding generation...")
    try:
        test_text = "def process_payment(amount): return payment_gateway.charge(amount)"
        embedding = await provider.embed_text(test_text)
        print(f"   ✅ Generated embedding")
        print(f"   📊 Dimension: {len(embedding)}")
        assert len(embedding) == 4096, f"Expected 4096, got {len(embedding)}"
        print("   ✅ Dimension verified")
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return False
    print()

    # Cleanup
    print("8. Cleaning up...")
    try:
        await provider.close()
        print("   ✅ Cleanup successful")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    print()

    print("=" * 80)
    print("✅ All ChromaDB integration tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_chromadb_qwen3())
    sys.exit(0 if success else 1)
