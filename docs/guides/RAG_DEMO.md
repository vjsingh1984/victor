# RAG Vertical Demo Guide

This guide demonstrates the RAG (Retrieval-Augmented Generation) vertical for document ingestion, vector search, and Q&A workflows.

## Overview

The RAG vertical provides:
- **Document Ingestion**: Ingest files, URLs, directories into a vector store
- **Vector Search**: LanceDB-based hybrid search (vector + full-text)
- **Query Interface**: Context-aware Q&A with source citations
- **Management Tools**: List, delete, and get statistics on indexed documents

## Quick Start

### 1. Ingest Documents

```bash
# Ingest a single file
victor rag ingest ./README.md

# Ingest a directory recursively
victor rag ingest ./docs --recursive --pattern "*.md"

# Ingest from URL
victor rag ingest https://example.com/api-docs
```

### 2. Search Documents

```bash
# Basic search
victor rag search "authentication"

# Search with more results
victor rag search "error handling" --top-k 10
```

### 3. Query with Context

```bash
# Get relevant context (no LLM synthesis)
victor rag query "What are the main features of this project?"

# Get synthesized answer using Ollama
victor rag query "What is the authentication flow?" --synthesize

# Use specific provider and model
victor rag query "Explain the API endpoints" --synthesize --provider anthropic --model claude-sonnet-4-20250514
```

## Demo Scripts

The RAG vertical includes two demo scripts showcasing real-world use cases.

### SEC 10-K/10-Q Filing Demo

Ingest and query SEC filings for **100 S&P 500 companies** including FAANG, Magnificent 7, and top companies by sector.

```bash
# Quick start: Ingest FAANG company filings
victor rag demo-sec --preset faang

# List available company presets
victor rag demo-sec --list
```

#### Available Presets

| Preset | Companies | Description |
|--------|-----------|-------------|
| `faang` | 5 | Meta, Amazon, Apple, Netflix, Google |
| `mag7` | 7 | Magnificent 7 (Apple, Microsoft, Google, Amazon, NVIDIA, Meta, Tesla) |
| `top10` | 10 | Top 10 S&P 500 by market cap |
| `top25` | 25 | Top 25 S&P 500 by market cap |
| `top50` | 50 | Top 50 S&P 500 by market cap |
| `top100` | 100 | All 100 tracked companies |
| `tech` | ~25 | Technology sector companies |
| `healthcare` | ~15 | Healthcare sector companies |
| `financials` | ~15 | Financial services companies |
| `energy` | 5 | Energy sector companies |

#### Ingest Filings

```bash
# Ingest by preset
victor rag demo-sec --preset faang          # 5 companies
victor rag demo-sec --preset mag7           # Magnificent 7
victor rag demo-sec --preset top50          # Top 50 S&P 500
victor rag demo-sec --preset tech           # Tech sector only

# Ingest specific companies
victor rag demo-sec --company AAPL
victor rag demo-sec --company AAPL --company MSFT --company NVDA

# Ingest 10-Q filings instead of 10-K
victor rag demo-sec --preset faang --filing-type 10-Q

# Ingest multiple filings per company
victor rag demo-sec --preset mag7 --count 3

# Control concurrent downloads
victor rag demo-sec --preset top50 --max-concurrent 10
```

#### Query Filings

```bash
# Query using victor rag CLI (after ingesting)
victor rag query "What is Apple's total revenue?"
victor rag query "What are Amazon's main risk factors?" --synthesize
victor rag query "Compare NVIDIA and AMD revenue growth" --synthesize -p anthropic

# Search without synthesis
victor rag search "technology risk factors"

# Use the demo command with sector filtering
victor rag demo-sec --query "Revenue trends" --sector Technology --synthesize
```

#### Manage SEC Filings

```bash
# Show statistics on ingested SEC filings
victor rag demo-sec --stats

# Clear all SEC filings from the store
victor rag demo-sec --clear

# List all documents (shows SEC filings too)
victor rag list
```

#### Example Output

```
================================================================================
Found 5 relevant chunks:

[1] Apple Inc - 10-K (2025-10-31)
    Score: 0.3526
    Content preview:
    Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS (In millions)
    Net sales: Products $ 307,003...

[2] Apple Inc - 10-K (2025-10-31)
    Score: 0.3416
    Content preview:
    Products and Services Performance...iPhone $ 209,586...

================================================================================
```

### Project Documentation Demo

Ingest and query project documentation.

```bash
# Run the built-in docs demo
victor rag demo docs

# Or ingest custom project docs using CLI
victor rag ingest ./docs --recursive --pattern "*.md"

# Query the documentation
victor rag query "How do I add a new provider?"
victor rag query "What tools are available?" --synthesize

# Show store statistics
victor rag stats

# List all indexed documents
victor rag list
```

## RAG Tools Reference

### rag_ingest

Ingest documents into the RAG knowledge base.

```python
# Ingest a file
await tool.execute(path="/path/to/document.md")

# Ingest from URL
await tool.execute(url="https://example.com/docs")

# Ingest directory recursively
await tool.execute(path="/path/to/docs", recursive=True, pattern="*.md")

# Ingest with custom metadata
await tool.execute(
    path="/path/to/file.txt",
    metadata={"category": "api", "version": "2.0"}
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Path to file or directory |
| `url` | string | URL to fetch and ingest |
| `content` | string | Direct content to ingest |
| `doc_type` | string | Type: text, markdown, code, pdf, html |
| `doc_id` | string | Custom document ID |
| `recursive` | boolean | Recursively ingest directory |
| `pattern` | string | Glob pattern for directory (e.g., "*.md") |
| `metadata` | object | Custom metadata to attach |

### rag_search

Search for relevant document chunks.

```python
# Basic search
results = await tool.execute(query="authentication methods")

# Search with filters
results = await tool.execute(
    query="error handling",
    top_k=10,
    doc_type="code"
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `top_k` | integer | Number of results (default: 5) |
| `doc_type` | string | Filter by document type |

### rag_query

Query with automatic context retrieval and answer generation.

```python
# Ask a question
result = await tool.execute(query="What is the authentication flow?")
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Question to answer |
| `top_k` | integer | Context chunks to retrieve |

### rag_list

List all indexed documents.

```python
docs = await tool.execute()
# Returns: List of document IDs with metadata
```

### rag_delete

Delete a document from the store.

```python
await tool.execute(doc_id="doc_abc123")
```

### rag_stats

Get store statistics.

```python
stats = await tool.execute()
# Returns: {total_documents, total_chunks, store_path}
```

## Architecture

### Document Store

The RAG vertical uses LanceDB for vector storage:

```
┌─────────────────────────────────────────────────────────────┐
│                    DocumentStore                             │
├─────────────────────────────────────────────────────────────┤
│  LanceDB (Embedded)                                          │
│  ├── Vector Index (HNSW)                                    │
│  ├── Full-Text Index                                        │
│  └── Metadata Store                                         │
├─────────────────────────────────────────────────────────────┤
│  Embedding: sentence-transformers/all-MiniLM-L6-v2 (384d)   │
│  Chunk Size: 512 tokens with 50 token overlap               │
│  Storage: .victor/rag/                                      │
└─────────────────────────────────────────────────────────────┘
```

### Chunking Strategies

Documents are chunked based on type:

| Type | Strategy | Notes |
|------|----------|-------|
| Text | Sentence boundary | Preserves complete sentences |
| Markdown | Header-aware | Respects ## sections |
| Code | Function/class boundary | Preserves complete definitions |
| PDF | Page-aware | Maintains page references |

### Search Pipeline

```
Query → Embedding → Vector Search → Reranking → Results
                         ↓
                   Full-Text Search
                         ↓
                   Hybrid Merge (RRF)
```

## Configuration

### Document Store Config

```python
from victor.rag.document_store import DocumentStoreConfig

config = DocumentStoreConfig(
    path=Path(".victor/rag"),      # Storage location
    table_name="documents",         # LanceDB table
    embedding_dim=384,              # Embedding dimension
    use_hybrid_search=True,         # Vector + FTS
    rerank_results=True,            # Rerank top results
    max_results=20,                 # Maximum results
)
```

### Chunking Config

```python
from victor.rag.chunker import ChunkingConfig

config = ChunkingConfig(
    chunk_size=512,        # Tokens per chunk
    chunk_overlap=50,      # Overlap between chunks
    min_chunk_size=100,    # Minimum chunk size
)
```

## Dependencies

Required packages:
```bash
pip install lancedb sentence-transformers pypdf beautifulsoup4
```

Optional for better HTML parsing:
```bash
pip install lxml
```

## Troubleshooting

### LanceDB Not Available

If you see "LanceDB not available - using in-memory fallback":
```bash
pip install lancedb pyarrow
```

### Embedding Model Download

On first use, the embedding model downloads automatically:
```
INFO:sentence_transformers:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
```

For air-gapped environments, pre-download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("/path/to/models/all-MiniLM-L6-v2")
```

### SSL Certificate Errors

If fetching URLs fails with SSL errors:
- The demo scripts include SSL workarounds for development
- For production, configure proper SSL certificates

## Example Use Cases

### 1. Code Documentation Q&A

```bash
# Ingest codebase documentation
victor rag ingest ./my-project/docs --recursive --pattern "*.md"

# Ask questions
victor rag query "How do I authenticate users?" --synthesize
victor rag query "What database does this project use?" --synthesize
```

### 2. Legal Document Analysis

```bash
# Ingest contracts
victor rag ingest ./contracts --recursive --pattern "*.pdf"

# Query for specific clauses
victor rag query "What are the termination conditions?" --synthesize
```

### 3. Research Paper Analysis

```bash
# Ingest papers
victor rag ingest ./papers --pattern "*.pdf"

# Query findings
victor rag query "What methods achieve state-of-the-art results?" --synthesize
```

### 4. Financial Analysis (SEC Filings)

```bash
# Ingest top 50 S&P 500 10-K filings
victor rag demo-sec --preset top50

# Query using CLI
victor rag query "What is Apple's revenue?" --synthesize
victor rag query "Compare risk factors across tech companies" --synthesize --provider anthropic

# Filter queries by sector
victor rag demo-sec --query "Revenue growth" --sector Healthcare --synthesize
```

## Related Documentation

- [VERTICALS.md](../VERTICALS.md) - Overview of all verticals
- [TOOL_CATALOG.md](../TOOL_CATALOG.md) - Complete tool reference
- [embeddings/README.md](../embeddings/README.md) - Embedding system details
