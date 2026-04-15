# Project Overview

**victor-rag** is a Python-based RAG (Retrieval-Augmented Generation) system implementing document ingestion, vector search, and knowledge management. It supports structured data workflows and integrates with tools like ChromaDB, LanceDB, and Sentence Transformers.

# Package Layout

| Path               | Description                                      |
|--------------------|--------------------------------------------------|
| `tests/`           | Unit and integration tests                       |
| `victor_rag/`      | Core source code                                 |

# Key Entry Points

| Component                     | Path                                | Description                                               |
|------------------------------|-------------------------------------|-----------------------------------------------------------|
| RAGEnrichmentConfig          | `victor_rag/enrichment.py:68`       | Configuration for RAG enrichment.                         |
| RAGEnrichmentStrategy        | `victor_rag/enrichment.py:98`       | Enrichment strategy for RAG vertical.                     |
| ChunkingConfig               | `victor_rag/chunker.py:96`          | Configuration for document chunking.                      |
| DocumentChunker              | `victor_rag/chunker.py:194`         | Class for splitting documents into chunks.                |
| RAGToolProvider              | `victor_rag/protocols.py:44`        | Tool provider for RAG vertical.                           |
| RAGToolSelectionStrategy     | `victor_rag/protocols.py:86`        | Stage-aware tool selection for RAG tasks.                 |
| RAGSafetyProvider            | `victor_rag/protocols.py:107`       | Safety provider for RAG vertical.                         |
| RAGPromptProvider            | `victor_rag/protocols.py:146`       | Prompt provider for RAG vertical.                         |
| RAGWorkflowProvider          | `victor_rag/protocols.py:188`       | Workflow provider for RAG vertical.                       |
| RAGAssistant                 | `victor_rag/assistant.py:47`        | Core assistant class for RAG operations.                  |
| EntityResolver               | `victor_rag/entity_resolver.py:194` | Resolves entities from queries using document store metadata. |
| SECFilingFetcher             | `victor_rag/demo_sec_filings.py:242`| Class for fetching SEC filings.                           |

# Development Commands

```bash
pip install -e ".[dev]"
pytest
```

# Dependencies

Core dependencies: 5 packages — victor-ai, chromadb, sentence-transformers, lancedb, pyarrow

# Configuration

Settings are loaded from `.env` → `~/.victor/profiles.yaml` → CLI flags (override order). Project context is defined in `.victor/init.md`.

# Architecture Notes

- Centralized protocol interfaces (`RAGToolProvider`, `RAGPromptProvider`, etc.) define vertical behavior.
- `RAGAssistant` is the main workflow orchestrator.
- `DocumentChunker` and `EntityResolver` are key data processors with high coupling to ingestion/query flows.
- Inheritance is minimal; most abstractions are through interfaces in `protocols.py`.
- Configuration flows through layered settings, supporting both runtime and static definitions.

# Codebase Scale

14,286 lines of code across 50 files (11,785 LOC source, 2,501 LOC config).

Run `/init --update` to refresh after code changes.