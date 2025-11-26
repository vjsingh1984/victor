"""Quick test of OllamaEmbeddingModel class."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from victor.codebase.embeddings.models import (
    EmbeddingModelConfig,
    OllamaEmbeddingModel,
)


async def test_ollama_embedding():
    """Test OllamaEmbeddingModel class."""

    print("=" * 80)
    print("Testing OllamaEmbeddingModel Class")
    print("=" * 80)
    print()

    # Create configuration
    config = EmbeddingModelConfig(
        model_type="ollama",
        model_name="qwen3-embedding:8b",
        dimension=4096,
        api_key="http://localhost:11434",  # Base URL
        batch_size=8
    )

    # Create model instance
    print("1. Creating OllamaEmbeddingModel instance...")
    model = OllamaEmbeddingModel(config)
    print("   ✅ Instance created")
    print()

    # Initialize model
    print("2. Initializing model...")
    try:
        await model.initialize()
        print("   ✅ Model initialized successfully")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    print()

    # Test single embedding
    print("3. Testing single text embedding...")
    test_text = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    try:
        embedding = await model.embed_text(test_text)
        print(f"   ✅ Generated embedding")
        print(f"   📊 Dimension: {len(embedding)}")
        print(f"   📈 First 5 values: {embedding[:5]}")
        print(f"   📉 Last 5 values: {embedding[-5:]}")

        # Verify dimension
        assert len(embedding) == 4096, f"Expected 4096 dimensions, got {len(embedding)}"
        print("   ✅ Dimension verification passed")
    except Exception as e:
        print(f"   ❌ Single embedding failed: {e}")
        return False
    print()

    # Test batch embedding
    print("4. Testing batch embedding...")
    test_texts = [
        "def add(a, b): return a + b",
        "def multiply(x, y): return x * y",
        "class Calculator: pass",
        "async def fetch_data(): return await api.get('/data')",
    ]
    try:
        embeddings = await model.embed_batch(test_texts)
        print(f"   ✅ Generated {len(embeddings)} embeddings")

        for i, emb in enumerate(embeddings):
            print(f"   📊 Text {i+1}: {len(emb)} dimensions")
            assert len(emb) == 4096, f"Expected 4096, got {len(emb)}"

        print("   ✅ All batch embeddings verified")
    except Exception as e:
        print(f"   ❌ Batch embedding failed: {e}")
        return False
    print()

    # Test dimension getter
    print("5. Testing get_dimension() method...")
    dim = model.get_dimension()
    print(f"   📊 Reported dimension: {dim}")
    assert dim == 4096, f"Expected 4096, got {dim}"
    print("   ✅ Dimension getter works")
    print()

    # Test cleanup
    print("6. Testing cleanup...")
    await model.close()
    print("   ✅ Cleanup successful")
    print()

    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_ollama_embedding())
    sys.exit(0 if success else 1)
