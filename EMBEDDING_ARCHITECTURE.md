# Embedding Plugin Architecture

## Overview

Victor uses a **plugin-based architecture** for vector embeddings and semantic search. This allows seamless switching between different embedding providers based on your needs:

- **Development/Testing**: ChromaDB (lightweight, in-memory, easy setup)
- **Production/Scale**: ProximaDB (custom-built, optimized for scale)
- **Other Options**: FAISS, Pinecone, Weaviate, etc.

## Design Goals

1. **Provider Independence**: Code should work with any embedding provider
2. **Hot-Swappable**: Change providers via configuration without code changes
3. **Consistent Interface**: Unified API regardless of backend
4. **Performance**: Async/await for non-blocking operations
5. **Extensibility**: Easy to add new providers

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CodebaseIndex                          │
│  (Orchestrator - uses embedding provider for search)     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ uses
                       │
┌──────────────────────▼──────────────────────────────────┐
│           BaseEmbeddingProvider (Abstract)               │
│  • embed_text(text) -> Vector                            │
│  • embed_batch(texts) -> List[Vector]                    │
│  • search_similar(query, limit) -> List[Result]          │
│  • index_documents(docs)                                 │
│  • get_stats()                                           │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┬──────────────┐
          │            │            │              │
┌─────────▼────┐  ┌───▼────────┐  ┌▼──────────┐  ┌▼─────────────┐
│  ChromaDB    │  │ ProximaDB  │  │   FAISS   │  │  Pinecone    │
│   Plugin     │  │   Plugin   │  │  Plugin   │  │    Plugin    │
└──────────────┘  └────────────┘  └───────────┘  └──────────────┘
```

## Plugin Interface

### BaseEmbeddingProvider

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    provider: str  # chromadb, proximadb, faiss, etc.
    model: str = "all-MiniLM-L6-v2"  # Default embedding model
    dimension: int = 384
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    persist_directory: Optional[str] = None

    # Provider-specific config
    extra_config: Dict[str, Any] = {}

class SearchResult(BaseModel):
    """Result from semantic search."""
    file_path: str
    symbol_name: Optional[str] = None
    content: str
    score: float
    metadata: Dict[str, Any] = {}

class BaseEmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load models, connect to DB)."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (optimized)."""
        pass

    @abstractmethod
    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Index a single document."""
        pass

    @abstractmethod
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """Batch index multiple documents."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from index."""
        pass

    @abstractmethod
    async def clear_index(self) -> None:
        """Clear entire index."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass
```

## Provider Registry

Similar to LLM providers, embeddings use a registry:

```python
class EmbeddingRegistry:
    """Registry for embedding providers."""

    _providers: Dict[str, Type[BaseEmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseEmbeddingProvider]):
        """Register a provider."""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingProvider:
        """Create provider from config."""
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown embedding provider: {config.provider}")
        return provider_class(config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())
```

## Configuration

### profiles.yaml

```yaml
codebase:
  # Development: Use ChromaDB (fast setup, no external dependencies)
  embedding_provider: chromadb
  embedding_model: "all-MiniLM-L6-v2"
  persist_directory: ~/.victor/embeddings/chroma

  # Production: Use ProximaDB (optimized for scale)
  # embedding_provider: proximadb
  # embedding_model: "text-embedding-ada-002"
  # proximadb_path: ~/code/proximaDB
  # proximadb_collection: victor-codebase
```

### Environment Variables

```bash
# Development (ChromaDB)
VICTOR_EMBEDDING_PROVIDER=chromadb

# Production (ProximaDB)
VICTOR_EMBEDDING_PROVIDER=proximadb
VICTOR_PROXIMADB_PATH=/path/to/proximaDB
VICTOR_PROXIMADB_COLLECTION=victor-prod
```

## Implementation Phases

### Phase 1: Core Interface (NOW)
- ✅ Define BaseEmbeddingProvider
- ✅ Define EmbeddingConfig, SearchResult models
- ✅ Create EmbeddingRegistry
- ✅ Update CodebaseIndex to use providers

### Phase 2: ChromaDB Plugin (Development)
- ⏳ Implement ChromaDBProvider
- ⏳ Use sentence-transformers for embeddings
- ⏳ In-memory and persistent modes
- ⏳ Test with codebase indexing

### Phase 3: ProximaDB Plugin (Production)
- ⏳ Implement ProximaDBProvider
- ⏳ Integrate with ~/code/proximaDB
- ⏳ Handle custom protocols
- ⏳ Performance benchmarks

### Phase 4: Additional Providers (Optional)
- ⏳ FAISSProvider (fast, CPU-only)
- ⏳ PineconeProvider (cloud, managed)
- ⏳ WeaviateProvider (graph-based)

## Integration with CodebaseIndex

Updated `CodebaseIndex` class:

```python
class CodebaseIndex:
    def __init__(
        self,
        root_path: str,
        ignore_patterns: Optional[List[str]] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        self.root = Path(root_path).resolve()
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE

        # Initialize embedding provider
        if embedding_config:
            self.embedding_provider = EmbeddingRegistry.create(embedding_config)
        else:
            # Default: ChromaDB for development
            default_config = EmbeddingConfig(
                provider="chromadb",
                persist_directory=str(Path.home() / ".victor/embeddings")
            )
            self.embedding_provider = EmbeddingRegistry.create(default_config)

    async def index_codebase(self) -> None:
        """Index with semantic search."""
        # ... existing indexing ...

        # Build documents for embedding
        documents = []
        for file_path, metadata in self.files.items():
            for symbol in metadata.symbols:
                doc = {
                    "id": f"{file_path}:{symbol.name}",
                    "content": self._build_symbol_context(symbol),
                    "metadata": {
                        "file_path": file_path,
                        "symbol_name": symbol.name,
                        "symbol_type": symbol.type,
                        "line_number": symbol.line_number,
                    }
                }
                documents.append(doc)

        # Index with embedding provider
        await self.embedding_provider.index_documents(documents)

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Semantic search using embeddings."""
        return await self.embedding_provider.search_similar(
            query=query,
            limit=max_results
        )
```

## Usage Examples

### Example 1: Development with ChromaDB

```python
from victor.codebase.indexer import CodebaseIndex
from victor.codebase.embeddings import EmbeddingConfig

# Create index with ChromaDB (default)
config = EmbeddingConfig(
    provider="chromadb",
    model="all-MiniLM-L6-v2",
    persist_directory="~/.victor/embeddings/my-project"
)

indexer = CodebaseIndex(root_path=".", embedding_config=config)
await indexer.index_codebase()

# Semantic search
results = await indexer.semantic_search(
    query="authentication middleware implementation",
    max_results=5
)

for result in results:
    print(f"{result.file_path}:{result.symbol_name} (score: {result.score})")
```

### Example 2: Production with ProximaDB

```python
from victor.codebase.embeddings import EmbeddingConfig

# Switch to ProximaDB for production
config = EmbeddingConfig(
    provider="proximadb",
    model="text-embedding-ada-002",
    extra_config={
        "proximadb_path": "~/code/proximaDB",
        "collection_name": "victor-enterprise",
        "batch_size": 1000,
        "cache_embeddings": True
    }
)

indexer = CodebaseIndex(root_path=".", embedding_config=config)
await indexer.index_codebase()

# Same API, different backend!
results = await indexer.semantic_search(
    query="database connection pooling",
    max_results=10
)
```

### Example 3: CLI Usage

```bash
# Initialize with ChromaDB (default)
victor init

# Index current codebase
victor index --provider chromadb

# Search semantically
victor search "how does error handling work?"

# Switch to ProximaDB
victor config set embedding_provider proximadb
victor config set proximadb_path ~/code/proximaDB

# Re-index with new provider
victor index --provider proximadb --force
```

## Performance Considerations

### ChromaDB (Development)
- **Pros**: Easy setup, no external dependencies, good for < 100k documents
- **Cons**: Slower at scale, memory-intensive
- **Best For**: Local development, small-medium codebases

### ProximaDB (Production)
- **Pros**: Custom-built for your use case, optimized for scale
- **Cons**: Requires setup/maintenance
- **Best For**: Large codebases, production deployments, specific requirements

### FAISS (Alternative)
- **Pros**: Very fast, CPU-only, good for large datasets
- **Cons**: No built-in persistence, manual index management
- **Best For**: High-performance search, research/experimentation

## Testing Strategy

```python
# tests/unit/test_embedding_providers.py
import pytest
from victor.codebase.embeddings import EmbeddingConfig, EmbeddingRegistry

@pytest.mark.parametrize("provider", ["chromadb", "proximadb", "faiss"])
async def test_provider_interface(provider):
    """Test all providers implement same interface."""
    config = EmbeddingConfig(provider=provider)
    provider_instance = EmbeddingRegistry.create(config)

    # Test basic operations
    await provider_instance.initialize()

    # Embed text
    embedding = await provider_instance.embed_text("test code snippet")
    assert len(embedding) == config.dimension

    # Index document
    await provider_instance.index_document(
        doc_id="test_file.py:function_name",
        content="def test(): pass",
        metadata={"file": "test_file.py"}
    )

    # Search
    results = await provider_instance.search_similar("test function", limit=5)
    assert len(results) <= 5

    # Clean up
    await provider_instance.close()
```

## Migration Path

### From ChromaDB to ProximaDB

```python
# 1. Export embeddings from ChromaDB
from victor.codebase.embeddings.migration import export_embeddings

await export_embeddings(
    from_provider="chromadb",
    from_config=chromadb_config,
    output_file="embeddings_export.json"
)

# 2. Import into ProximaDB
await import_embeddings(
    to_provider="proximadb",
    to_config=proximadb_config,
    input_file="embeddings_export.json"
)
```

## Future Enhancements

1. **Hybrid Search**: Combine keyword + semantic search
2. **Embedding Caching**: Cache embeddings to avoid re-computation
3. **Multi-Model Support**: Different models for different file types
4. **Incremental Updates**: Only re-embed changed files
5. **Distributed Indexing**: Parallel embedding generation
6. **Query Expansion**: Automatically expand queries for better results

## Conclusion

This plugin-based architecture provides:
- ✅ **Flexibility**: Easy to switch providers
- ✅ **Consistency**: Same interface across providers
- ✅ **Scalability**: Start simple, scale with ProximaDB
- ✅ **Extensibility**: Add new providers easily
- ✅ **Performance**: Async operations, optimized for speed

Start with ChromaDB for development, scale to ProximaDB for production!
