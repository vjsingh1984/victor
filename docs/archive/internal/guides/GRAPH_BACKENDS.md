# Graph Backends

Victor stores per-project code graphs under `.victor/graph/`. This doc explains how to configure the backend and how to add a new one.

## Defaults
- Backend: `sqlite`
- Path: `{project_root}/.victor/graph/graph.db`

## Configuration
- Settings (prefer): `codebase_graph_store`, `codebase_graph_path`
- Environment: `VICTOR_GRAPH_STORE` (overrides backend)
- CLI / code search: picked up by `CodebaseIndex` and `code_search_tool`

Examples:
```bash
export VICTOR_GRAPH_STORE=sqlite
victor chat ...
```

```python
from victor.codebase.indexer import CodebaseIndex
index = CodebaseIndex(root_path=".", graph_store_name="memory")
```

## Built-in backends

### Production-Ready
- `sqlite` (persistent, default) - Recommended for most use cases
- `duckdb` (persistent, optional `duckdb` dependency) - Better for large graphs
- `memory` (ephemeral, testing) - Fast but not persistent

### Experimental (Not Yet Implemented)
> **Warning**: The following backends are stubs and will raise `NotImplementedError`.
> They are planned for future releases. Contributions welcome!

- `lancedb` - Columnar storage for graph data (requires `lancedb`)
- `neo4j` - Native graph database integration (requires `neo4j`)

## Adding a backend
1. Implement `GraphStoreProtocol` (see `victor/codebase/graph/protocol.py`).
2. Add the implementation (e.g., `neo4j_store.py`).
3. Register the backend name in `victor/codebase/graph/registry.py`.
4. Add unit tests similar to `tests/unit/test_graph_registry.py`.

## Where the graph is used
- `CodebaseIndex` writes symbols, CONTAINS, CALLS, IMPORTS, REFERENCES edges.
- Tree-sitter is used (when available) to improve identifier capture.
