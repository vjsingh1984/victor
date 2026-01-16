# ProximaDB Integration (Experimental)

## Status: Experimental

**Last Updated**: 2025-01-16

ProximaDB is an **experimental** vector store integration in Victor. It is under active development and should not be used in production environments.

## Overview

ProximaDB is an optional vector store that provides an alternative to LanceDB for semantic code search and conversation embeddings.

### Key Characteristics

- **Status**: Experimental (under active development)
- **Stability**: May have bugs, API changes, or performance issues
- **Default**: No - LanceDB is the default and recommended vector store
- **Installation**: Optional via `pip install victor-ai[vector-experimental]`

## Installation

### Step 1: Install ProximaDB dependency

```bash
pip install victor-ai[vector-experimental]
```

This installs ProximaDB from the Git repository:
```
proximadb @ git+ssh://git@github.com/vjsingh1984/proximaDB.git@f2df18818ae808524920c99d6bd617990786acef#egg=proximadb
```

### Step 2: Configure Victor to use ProximaDB

Victor will automatically fall back to LanceDB if ProximaDB is configured but not installed. To use ProximaDB:

```python
from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.registry import EmbeddingRegistry

config = EmbeddingConfig(
    vector_store="proximadb",
    persist_directory="~/.victor/embeddings/proximadb",
    embedding_model_type="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
)

provider = EmbeddingRegistry.create(config)
```

Or via environment variables:

```bash
export VICTOR_VECTOR_STORE=proximadb
export VICTOR_EMBEDDINGS_DIR=~/.victor/embeddings/proximadb
```

## Graceful Fallback

Victor implements automatic fallback to LanceDB if ProximaDB is configured but not available:

```python
# If ProximaDB is not installed, Victor will:
# 1. Log a warning message
# 2. Automatically use LanceDB instead
# 3. Continue normal operation

# Example warning:
# "ProximaDB is configured but not installed.
#  Install with: pip install victor-ai[vector-experimental].
#  Using LanceDB as fallback."
```

## Recommendations

### Use LanceDB (Default) For:

- Production environments
- Stable, reliable operation
- Well-tested and documented workflows
- Team collaboration

### Use ProximaDB (Experimental) For:

- Testing and development only
- Contributing to ProximaDB development
- Evaluating new vector store technologies
- Providing feedback on bugs and issues

## Troubleshooting

### ProximaDB Not Available

If you see this warning:
```
ProximaDB is configured but not installed.
Install with: pip install victor-ai[vector-experimental].
Using LanceDB as fallback.
```

**Solution**: Either install ProximaDB or remove the ProximaDB configuration to use LanceDB (default).

### Git Dependency Issues

If installation fails with Git-related errors:
```
ERROR: Command errored out with exit status 128: git clone ...
```

**Solutions**:
1. Ensure Git is installed and accessible
2. Check SSH key configuration for GitHub access
3. Use LanceDB (default) instead - no installation required

### Import Errors

If you see:
```
ImportError: No module named 'proximadb'
```

**Solution**: Install the experimental dependencies:
```bash
pip install victor-ai[vector-experimental]
```

## Architecture

### Vector Store Registry

Victor uses a registry pattern to manage multiple vector store implementations:

**File**: `/Users/vijaysingh/code/codingagent/victor/storage/vector_stores/registry.py`

Key features:
- Auto-discovery of available vector stores
- Graceful handling of missing dependencies
- Automatic fallback to LanceDB
- Singleton pattern for provider instances

**File**: `/Users/vijaysingh/code/codingagent/victor/coding/codebase/embeddings/registry.py`

Separate registry for codebase-specific embeddings with same graceful fallback behavior.

### Provider Implementation

**File**: `/Users/vijaysingh/code/codingagent/victor/storage/vector_stores/proximadb_provider.py`

Implements the `BaseEmbeddingProvider` protocol for ProximaDB integration.

## Migration from ProximaDB to LanceDB

If you need to migrate from ProximaDB to LanceDB:

1. **Export embeddings** (if export functionality is available)
2. **Reconfigure** Victor to use LanceDB:
   ```bash
   export VICTOR_VECTOR_STORE=lancedb
   ```
3. **Rebuild embeddings**:
   ```bash
   victor index  # Rebuild codebase index with LanceDB
   ```
4. **Verify** functionality with LanceDB

## Future Roadmap

- [ ] Stabilize ProximaDB integration
- [ ] Add comprehensive tests
- [ ] Performance benchmarks vs LanceDB
- [ ] Migration tools for data transfer
- [ ] Production-ready status

## Contributing

If you want to improve ProximaDB integration:

1. Test thoroughly before committing
2. Add tests for new functionality
3. Document breaking changes
4. Submit issues for bugs discovered
5. Follow the experimental feature guidelines

## Related Documentation

- [Vector Stores Overview](../architecture/vector_stores.md)
- [Embedding System](../architecture/embeddings.md)
- [Installation Guide](../getting-started/installation.md)
- [Configuration Reference](../reference/configuration.md)

## See Also

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [ProximaDB Repository](https://github.com/vjsingh1984/proximaDB)
- [Vector Store Comparison](../performance/vector_stores.md)
