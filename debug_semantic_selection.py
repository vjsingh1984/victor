#!/usr/bin/env python3
"""Debug semantic tool selection to understand why 0 tools are selected."""

import asyncio
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

from victor.config.settings import Settings
from victor.tools.base import ToolRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.agent.orchestrator import AgentOrchestrator


async def debug_semantic_selection():
    """Debug why semantic selection is selecting 0 tools."""

    settings = Settings()

    print("=" * 70)
    print("DEBUG: Semantic Tool Selection")
    print("=" * 70)
    print()

    # Check cache file
    cache_file = Path.home() / ".victor" / "embeddings" / "tool_embeddings_all-MiniLM-L12-v2.pkl"
    print(f"Cache file: {cache_file}")
    print(f"Exists: {cache_file.exists()}")

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        print(f"Cache embedding_model: {cache_data.get('embedding_model')}")
        print(f"Cache tools_hash: {cache_data.get('tools_hash')[:16]}...")
        print(f"Cache embeddings count: {len(cache_data.get('embeddings', {}))}")

        # Check embedding dimensions
        first_tool = list(cache_data.get('embeddings', {}).keys())[0]
        first_embedding = cache_data.get('embeddings', {})[first_tool]
        print(f"Sample tool: {first_tool}")
        print(f"Sample embedding shape: {first_embedding.shape if hasattr(first_embedding, 'shape') else len(first_embedding)}")
        print(f"Sample embedding type: {type(first_embedding)}")
        print()

    # Create selector
    selector = SemanticToolSelector(
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        cache_embeddings=True
    )

    print(f"Selector model: {selector.embedding_model}")
    print(f"Selector provider: {selector.embedding_provider}")
    print()

    # Create minimal tool registry
    tools = ToolRegistry()
    from victor.tools.filesystem import read_file, write_file, list_directory
    from victor.tools.bash import execute_bash

    tools.register(read_file)
    tools.register(write_file)
    tools.register(list_directory)
    tools.register(execute_bash)

    print(f"Registered {len(tools.list_tools())} tools")
    print()

    # Initialize embeddings
    print("Initializing tool embeddings...")
    await selector.initialize_tool_embeddings(tools)
    print(f"Tool embedding cache size: {len(selector._tool_embedding_cache)}")
    print()

    # Test query
    test_query = "Write a Python function to validate email addresses"
    print(f"Test query: '{test_query}'")
    print()

    # Get query embedding
    print("Computing query embedding...")
    query_embedding = await selector._get_embedding(test_query)
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding type: {type(query_embedding)}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.4f}")
    print()

    # Compute similarities manually
    print("Computing similarities for each tool:")
    print()

    for tool_name, tool_embedding in selector._tool_embedding_cache.items():
        print(f"Tool: {tool_name}")
        print(f"  Embedding shape: {tool_embedding.shape}")
        print(f"  Embedding type: {type(tool_embedding)}")
        print(f"  Embedding norm: {np.linalg.norm(tool_embedding):.4f}")

        # Check if shapes match
        if query_embedding.shape != tool_embedding.shape:
            print(f"  ❌ SHAPE MISMATCH! Query: {query_embedding.shape}, Tool: {tool_embedding.shape}")
            continue

        # Compute cosine similarity
        similarity = selector._cosine_similarity(query_embedding, tool_embedding)
        print(f"  Similarity: {similarity:.4f}")

        if similarity >= 0.3:
            print(f"  ✅ Above threshold (0.3)")
        else:
            print(f"  ❌ Below threshold (0.3)")
        print()

    # Try actual tool selection
    print("=" * 70)
    print("Actual tool selection:")
    print("=" * 70)

    selected = await selector.select_relevant_tools(
        test_query,
        tools,
        max_tools=5,
        similarity_threshold=0.3
    )

    print(f"Selected {len(selected)} tools:")
    for tool in selected:
        print(f"  - {tool.name}")


if __name__ == "__main__":
    asyncio.run(debug_semantic_selection())
