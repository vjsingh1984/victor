# ProximaDB Quick Start Guide

## For Most Users (Recommended)

**You don't need to do anything!** Victor works out of the box with LanceDB, the default vector store.

```bash
# Just install and use Victor
pip install victor-ai
victor chat "Help me understand this codebase"
```text

LanceDB provides excellent performance for:
- Semantic code search
- Conversation embeddings
- Codebase understanding
- All standard features

## For Experimental Users

If you want to experiment with ProximaDB (an experimental vector store under active development):

### Step 1: Install Experimental Dependencies

```bash
pip install victor-ai[vector-experimental]
```

**Note**: This installs ProximaDB from a Git repository. Ensure you have:
- Git installed
- SSH access to GitHub (or use HTTPS URL)
- Stable internet connection

### Step 2: Configure Victor

```bash
# Set environment variable
export VICTOR_VECTOR_STORE=proximadb

# Or use in code
from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.registry import EmbeddingRegistry

config = EmbeddingConfig(vector_store='proximadb')
provider = EmbeddingRegistry.create(config)
```text

### Step 3: Use Victor Normally

```bash
victor chat "Help me understand this codebase"
```

## What If Installation Fails?

If ProximaDB installation fails, **don't worry!** Victor automatically falls back to LanceDB.

You'll see a warning:
```bash
ProximaDB is configured but not installed.
Install with: pip install victor-ai[vector-experimental].
Using LanceDB as fallback.
```text

Victor will continue working normally with LanceDB.

## Troubleshooting

### Error: "Command errored out with exit status 128: git clone..."

**Cause**: Git not installed or SSH keys not configured

**Solutions**:
1. Install Git:
   - macOS: `xcode-select --install`
   - Ubuntu: `sudo apt-get install git`
   - Windows: Download from git-scm.com

2. Configure SSH keys for GitHub (or use HTTPS instead)

3. Or just use LanceDB (default) - no installation needed!

### Error: "No module named 'proximadb'"

**Cause**: ProximaDB not installed

**Solution**:
```bash
pip install victor-ai[vector-experimental]
```bash

Or just remove the ProximaDB configuration and use LanceDB (default).

### Warning: "ProximaDB is configured but not installed"

**This is normal!** Victor is telling you it's falling back to LanceDB.

**You can**:
- Install ProximaDB: `pip install victor-ai[vector-experimental]`
- Or ignore the warning and use LanceDB (recommended)

## Performance Comparison

| Vector Store | Status | Recommended For | Stability |
|--------------|--------|-----------------|-----------|
| **LanceDB** | Default (Always installed) | Production, daily use | ✅ Stable |
| **ProximaDB** | Experimental (Optional) | Testing, development | ⚠️ Experimental |

## Recommendation

**Use LanceDB** (default) for:
- ✅ Production environments
- ✅ Stable operation
- ✅ Team projects
- ✅ Daily development

**Use ProximaDB** only for:
- ⚠️ Experimental testing
- ⚠️ Contributing to ProximaDB development
- ⚠️ Evaluating new technologies

## Need Help?

- [ProximaDB Documentation](../features/proximadb_experimental.md)
- [Vector Store Architecture](../verticals/rag.md)
- [Installation Guide](../getting-started/installation.md)
- [GitHub Issues](https://github.com/vjsingh1984/victor/issues)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 2 min
