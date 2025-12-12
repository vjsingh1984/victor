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
- `sqlite` (persistent, default)
- `duckdb` (persistent, optional `duckdb` dependency)
- `memory` (ephemeral, testing)
- `lancedb` (placeholder, optional `lancedb` dependency)
- `neo4j` (placeholder, optional `neo4j` dependency)

## Adding a backend
1. Implement `GraphStoreProtocol` (see `victor/codebase/graph/protocol.py`).
2. Add the implementation (e.g., `neo4j_store.py`).
3. Register the backend name in `victor/codebase/graph/registry.py`.
4. Add unit tests similar to `tests/unit/test_graph_registry.py`.

## Where the graph is used
- `CodebaseIndex` writes symbols, CONTAINS, CALLS, IMPORTS, REFERENCES edges.
- Tree-sitter is used (when available) to improve identifier capture.
