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
# Using Victor CLI with RAG vertical
victor chat --vertical rag "Ingest the file docs/README.md"

# Using RAG tools directly
victor tools call rag_ingest --path ./docs/README.md
```

### 2. Search Documents

```bash
victor chat --vertical rag "Search for information about authentication"
```

### 3. Query with Context

```bash
victor chat --vertical rag "What are the main features of this project?"
```

## Demo Scripts

The RAG vertical includes two demo scripts showcasing real-world use cases.

### SEC 10-K/10-Q Filing Demo

Ingest and query SEC filings for FAANG companies (Meta, Amazon, Apple, Netflix, Google).

```bash
# List available companies
python -m victor.verticals.rag.demo_sec_filings --list-companies

# Output:
# Available FAANG Companies:
# --------------------------------------------------
#   META: Meta Platforms (Facebook) (CIK: 0001326801)
#   AMZN: Amazon.com Inc (CIK: 0001018724)
#   AAPL: Apple Inc (CIK: 0000320193)
#   NFLX: Netflix Inc (CIK: 0001065280)
#   GOOGL: Alphabet Inc (Google) (CIK: 0001652044)
```

#### Ingest Filings

```bash
# Ingest Apple's latest 10-K filing
python -m victor.verticals.rag.demo_sec_filings --company AAPL

# Ingest all FAANG 10-K filings
python -m victor.verticals.rag.demo_sec_filings

# Ingest 10-Q filings instead
python -m victor.verticals.rag.demo_sec_filings --filing-type 10-Q

# Ingest multiple filings per company
python -m victor.verticals.rag.demo_sec_filings --count 3
```

#### Query Filings

```bash
# Query for financial information
python -m victor.verticals.rag.demo_sec_filings --query "What is Apple's total revenue?"

# Query for risk factors
python -m victor.verticals.rag.demo_sec_filings --query "What are Amazon's main risk factors?"

# Query for specific metrics
python -m victor.verticals.rag.demo_sec_filings --query "What is Netflix's subscriber count?"
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
# Ingest Victor's own documentation
python -m victor.verticals.rag.demo_docs

# Ingest custom project docs
python -m victor.verticals.rag.demo_docs --path /path/to/project --pattern "*.md"

# Query the documentation
python -m victor.verticals.rag.demo_docs --query "How do I add a new provider?"
python -m victor.verticals.rag.demo_docs --query "What tools are available?"

# Show store statistics
python -m victor.verticals.rag.demo_docs --stats
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
from victor.verticals.rag.document_store import DocumentStoreConfig

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
from victor.verticals.rag.chunker import ChunkingConfig

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
python -m victor.verticals.rag.demo_docs --path ./my-project

# Ask questions
victor chat -V rag "How do I authenticate users?"
victor chat -V rag "What database does this project use?"
```

### 2. Legal Document Analysis

```bash
# Ingest contracts
victor tools call rag_ingest --path ./contracts --recursive --pattern "*.pdf"

# Query for specific clauses
victor chat -V rag "What are the termination conditions?"
```

### 3. Research Paper Analysis

```bash
# Ingest papers
victor tools call rag_ingest --path ./papers --pattern "*.pdf"

# Query findings
victor chat -V rag "What methods achieve state-of-the-art results?"
```

### 4. Financial Analysis (SEC Filings)

```bash
# Ingest all FAANG 10-K filings
python -m victor.verticals.rag.demo_sec_filings

# Compare companies
python -m victor.verticals.rag.demo_sec_filings --query "Compare revenue growth across companies"
```

## Related Documentation

- [VERTICALS.md](../VERTICALS.md) - Overview of all verticals
- [TOOL_CATALOG.md](../TOOL_CATALOG.md) - Complete tool reference
- [embeddings/README.md](../embeddings/README.md) - Embedding system details
