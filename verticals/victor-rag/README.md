# victor-rag

**RAG (Retrieval-Augmented Generation) Vertical for Victor AI**

A complete RAG implementation showcasing document ingestion, vector search, and knowledge management with Victor AI.

## Features

- 📄 **Multi-format Document Ingestion**
  - PDF, Markdown, Text, Code files
  - Semantic chunking with configurable overlap
  - Automatic document type detection

- 🔍 **Hybrid Search**
  - Vector search (semantic similarity)
  - Full-text search (keyword matching)
  - Combined reranking for best results

- 💾 **Embedded Vector Storage**
  - LanceDB (no server required)
  - Persistent local storage
  - Fast similarity search

- 🎯 **Query Enhancement**
  - Context retrieval from relevant documents
  - Source attribution and citations
  - Confidence scoring

## Installation

```bash
# Install with Victor core
pip install victor-ai

# Install RAG vertical
pip install victor-rag
```

## Quick Start

```python
from victor.framework import Agent

# Create agent with RAG vertical
agent = await Agent.create(
    provider="ollama",
    model="qwen2.5-coder:7b",
    vertical="rag"
)

# Ingest a document
await agent.chat("Ingest README.md into the knowledge base")

# Query the knowledge base
result = await agent.chat("What does this project do?")
```

## Available Tools

Once installed, the RAG vertical provides these tools:

- **rag_ingest** - Ingest documents into the knowledge base
- **rag_search** - Search for relevant document chunks
- **rag_query** - Query with context retrieval
- **rag_list** - List all indexed documents
- **rag_delete** - Delete documents from knowledge base
- **rag_stats** - Get knowledge base statistics

## System Prompt

The RAG vertical includes specialized prompt contributions:

- **Query Strategy**: When to use semantic vs keyword search
- **Citation Format**: How to reference sources
- **Document Handling**: Chunking strategies by document type
- **Quality Rules**: Grounding rules to prevent hallucination

## Configuration

The RAG vertical can be configured via environment variables:

```bash
# Vector store location
export VICTOR_RAG_DB_PATH=./rag_db

# Chunking configuration
export VICTOR_RAG_CHUNK_SIZE=512
export VICTOR_RAG_CHUNK_OVERLAP=50

# Search configuration
export VICTOR_RAG_TOP_K=5
export VICTOR_RAG_MIN_CONFIDENCE=0.3
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black victor_rag/

# Type check
mypy victor_rag/
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Links

- **Victor AI**: https://github.com/vjsingh1984/victor
- **Documentation**: https://docs.victor.dev/verticals/rag
- **Victor Registry**: https://github.com/vjsingh1984/victor-registry
