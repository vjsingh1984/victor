#!/usr/bin/env python3
"""Debug what text is being embedded for each tool."""

import asyncio
from victor.tools.base import ToolRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.config.settings import Settings
from victor.tools.filesystem import read_file, write_file, list_directory
from victor.tools.bash import execute_bash


async def debug_tool_text():
    """Check what text is being embedded for tools."""

    print("=" * 70)
    print("DEBUG: Tool Text for Embedding")
    print("=" * 70)
    print()

    # Create selector
    settings = Settings()
    selector = SemanticToolSelector(
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
    )

    # Create tools
    tools = ToolRegistry()
    tools.register(read_file)
    tools.register(write_file)
    tools.register(list_directory)
    tools.register(execute_bash)

    print(f"Registered {len(tools.list_tools())} tools")
    print()

    # Check what text is created for each tool
    for tool in tools.list_tools():
        tool_text = selector._create_tool_text(tool)

        print(f"Tool: {tool.name}")
        print(f"  Description: {tool.description[:100]}..." if len(tool.description) > 100 else f"  Description: {tool.description}")
        print(f"  Generated text for embedding:")
        print(f"  ---")
        print(f"  {tool_text}")
        print(f"  ---")
        print(f"  Text length: {len(tool_text)} characters")
        print()

    # Also check test query
    test_query = "Write a Python function to validate email addresses"
    print("=" * 70)
    print(f"Test query: '{test_query}'")
    print(f"Query length: {len(test_query)} characters")
    print()

    # Compute embeddings and check if model is loaded
    print("Computing embedding for test query...")
    query_embedding = await selector._get_embedding(test_query)
    print(f"Query embedding computed: shape={query_embedding.shape}, norm={query_embedding.sum():.4f}")
    print()

    # Compute embedding for a simple tool
    print("Computing embedding for write_file tool...")
    tool = [t for t in tools.list_tools() if t.name == "write_file"][0]
    tool_text = selector._create_tool_text(tool)
    tool_embedding = await selector._get_embedding(tool_text)
    print(f"Tool embedding computed: shape={tool_embedding.shape}, norm={tool_embedding.sum():.4f}")
    print()

    # Compute similarity
    similarity = selector._cosine_similarity(query_embedding, tool_embedding)
    print(f"Cosine similarity: {similarity:.4f}")

    if similarity < 0.3:
        print(f"❌ Similarity below threshold (0.3)")
        print()
        print("This suggests the embeddings are not semantically meaningful.")
        print("Possible causes:")
        print("  1. Model not loaded correctly")
        print("  2. Tool text is too generic")
        print("  3. Embedding normalization issue")
    else:
        print(f"✅ Similarity above threshold (0.3)")


if __name__ == "__main__":
    asyncio.run(debug_tool_text())
