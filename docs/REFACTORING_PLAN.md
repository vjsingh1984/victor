# God File Refactoring Plan: codebase/indexer.py

## Overview

**File**: `victor_coding/codebase/indexer.py`
**Lines**: 3,516 total
**God Class**: `CodebaseIndex` (lines 1081-3230, ~2,149 lines)

## Current Structure

### Main Classes
1. **CodebaseIndex** (lines 1081-3230) - The god class
2. **CodebaseFileHandler** (lines 934-1006) - File system watcher
3. **IndexedSymbol** (lines 1032-1060) - Data model
4. **FileMetadata** (lines 1061-1080) - Data model
5. **SymbolVisitor** (lines 3236-3357) - AST visitor
6. **BackgroundIndexerService** (lines 3358-3500) - Background indexing

### CodebaseIndex Methods by Category

| Category | Methods | Lines | Purpose |
|----------|---------|-------|---------|
| Core/Setup | `__init__`, `_normalize_graph_writer_mode`, `make_symbol_id`, `make_file_id`, `_reset_graph_buffers`, `_enriched_to_symbol`, `_should_ignore`, `_merge_parallel_result`, `_detect_language`, `_is_config_language` | ~11 | Initialization and utility |
| Indexing Operations | `index_codebase`, `_index_files_parallel`, `ensure_indexed`, `incremental_reindex`, `_index_single_file`, `_remove_file_from_index`, `_index_tree_sitter_file` | ~7 | Main indexing workflow |
| Symbol Extraction | `_extract_references`, `_extract_config_keys`, `_extract_symbols_with_tree_sitter`, `_extract_inheritance`, `_extract_implements`, `_extract_composition`, `_find_enclosing_symbol_name`, `_extract_calls_with_tree_sitter`, `_extract_imports_with_tree_sitter` | ~9 | Extract symbols from code |
| Graph Building | `_record_symbols`, `_resolve_cross_file_calls`, `_build_dependency_graph`, `_find_dependents` | ~4 | Build call graph |
| Embedding & Search | `get_stats`, `_resolve_project_storage_root`, `_default_embedding_persist_dir`, `_initialize_embeddings`, `_build_symbol_context`, `semantic_search` | ~6 | Vector embeddings |
| Query Operations | `find_symbol`, `get_file_context` | ~2 | Public query API |

## Proposed Refactoring

### Phase 1: Extract Symbol Extraction (Low Risk)

**New File**: `codebase/symbol_extraction.py`

**Extract**:
- `_extract_symbols_with_tree_sitter`
- `_extract_inheritance`
- `_extract_implements`
- `_extract_composition`
- `_extract_calls_with_tree_sitter`
- `_extract_imports_with_tree_sitter`
- `_extract_references`
- `_extract_config_keys`
- `_find_enclosing_symbol_name`

**Dependencies**: 
- Tree-sitter
- Plugin queries
- Language detection (keep in CodebaseIndex)

**Interface**:
```python
class SymbolExtractor:
    def __init__(self, plugin_getter_func, language_detector):
        self._get_plugin_query = plugin_getter_func
        self._detect_language = language_detector
    
    def extract_symbols(self, file_path: Path, language: str) -> List[Symbol]:
        """Extract all symbols from a file."""
    
    def extract_relationships(self, tree: Any, language: str) -> List[Relationship]:
        """Extract inheritance, composition, calls."""
```

### Phase 2: Extract Embedding Management (Low Risk)

**New File**: `codebase/embeddings/index_embeddings.py`

**Extract**:
- `_initialize_embeddings`
- `_build_symbol_context`
- `_get_symbol_embedding_text`
- `_resolve_project_storage_root`
- `_default_embedding_persist_dir`
- `semantic_search`

**Dependencies**:
- Embedding providers
- Vector stores

**Interface**:
```python
class EmbeddingManager:
    def __init__(self, config: Optional[Dict]):
        self.config = config or {}
    
    def initialize(self, storage_root: Path) -> None:
        """Initialize embedding model and vector store."""
    
    def embed_symbols(self, symbols: List[Symbol]) -> None:
        """Generate and store embeddings for symbols."""
    
    def semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Search for semantically similar symbols."""
```

### Phase 3: Extract Graph Building (Medium Risk)

**New File**: `codebase/graph/call_graph_builder.py`

**Extract**:
- `_record_symbols`
- `_resolve_cross_file_calls`
- `_build_dependency_graph`
- `_find_dependents`

**Dependencies**:
- Graph store
- Symbol store
- Extracted symbols

**Interface**:
```python
class CallGraphBuilder:
    def __init__(self, graph_store, symbol_store):
        self.graph_store = graph_store
        self.symbol_store = symbol_store
    
    def record_symbols(self, symbols: List[Symbol]) -> None:
        """Record symbols and their relationships in graph."""
    
    def resolve_calls(self) -> None:
        """Resolve cross-file call relationships."""
    
    def build_graph(self) -> None:
        """Build complete dependency graph."""
```

### Phase 4: Simplify CodebaseIndex (Medium Risk)

After extractions, `CodebaseIndex` becomes an orchestrator:

```python
class CodebaseIndex:
    def __init__(self, ...):
        self.symbol_extractor = SymbolExtractor(...)
        self.embedding_manager = EmbeddingManager(...)
        self.call_graph_builder = CallGraphBuilder(...)
    
    # Main operations
    async def index_codebase(self) -> None:
        """Orchestrate indexing workflow."""
    
    async def ensure_indexed(self) -> None:
        """Ensure index is up to date."""
    
    # Query operations (delegate)
    def find_symbol(self, name: str) -> Optional[Symbol]:
        return self.symbol_store.find(name)
    
    def semantic_search(self, query: str, top_k: int):
        return self.embedding_manager.semantic_search(query, top_k)
```

### Phase 5: Separate Data Models (Low Risk)

**New File**: `codebase/models.py`

**Extract**:
- `IndexedSymbol`
- `FileMetadata`
- Related enums and constants

## Migration Strategy

### Step 1: Create New Modules (Additive)
1. Create `codebase/symbol_extraction.py`
2. Create `codebase/embeddings/index_embeddings.py`
3. Create `codebase/graph/call_graph_builder.py`
4. Create `codebase/models.py`

### Step 2: Add Delegation (Non-Breaking)
1. Add instances of new classes to `CodebaseIndex.__init__`
2. Delegate methods to new classes
3. Keep old methods as forwarding wrappers

### Step 3: Update Tests
1. Update imports in tests
2. Test new classes independently
3. Integration tests for `CodebaseIndex`

### Step 4: Remove Old Code (Breaking)
1. Remove forwarding wrappers
2. Clean up unused methods
3. Update documentation

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Symbol Extraction | Low | Well-defined interface, minimal dependencies |
| Embedding Management | Low | Self-contained functionality |
| Graph Building | Medium | Shared state, careful coordination needed |
| Simplify CodebaseIndex | Medium | Core changes, extensive testing required |

## Estimated Effort

| Phase | Time | Complexity |
|-------|------|------------|
| Phase 1: Symbol Extraction | 4 hours | Medium |
| Phase 2: Embedding Management | 3 hours | Low |
| Phase 3: Graph Building | 5 hours | Medium |
| Phase 4: Simplify CodebaseIndex | 3 hours | Medium |
| Phase 5: Data Models | 2 hours | Low |
| Testing | 4 hours | - |
| **Total** | **21 hours** | - |

## Success Criteria

1. **Reduced Complexity**: Each module under 500 lines
2. **Testability**: Each class can be tested independently
3. **Maintainability**: Changes to one area don't affect others
4. **Performance**: No regression in indexing speed
5. **Compatibility**: Existing tests pass without modification

## Deferred Items

- `BackgroundIndexerService` - Could be separated but used differently
- Helper functions at module level - Can be extracted later
- `CodebaseFileHandler` - Already well-separated

## Related Work

- Similar refactoring done in `victor_coding/languages/base.py`
- Reference: `docs/ANALYSIS_SUMMARY.md`
