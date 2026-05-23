# Tree-Sitter Analysis Service Plan

Last updated: 2026-05-23
Owner: Architecture/Foundation + Coding Vertical
Status: Active
Decision status: Design baseline accepted for phased implementation

## Goal

Make Victor's tree-sitter usage service-owned, plugin-backed, and reusable across
code graph indexing, code intelligence tools, structural embeddings, and
language-specific analysis.

This file is the persistent execution tracker for the tree-sitter analysis work.
Future sessions should update this file first, continue from the highest-priority
open task, and append to the session log before stopping.

## How To Use This File

- Treat this document as the design record, implementation backlog, and handoff
  note for this workstream.
- Update task checkboxes and status rows before ending a session.
- Keep root `victor-ai` and sibling `victor-coding` responsibilities separate.
- Do not add new language-specific grammar knowledge to root `victor-ai`; put it
  in `victor-coding` language plugins.
- Preserve fallback behavior when tree-sitter packages, vector dependencies, or
  `victor-coding` are unavailable.

## Verified Context Snapshot

These facts were verified on 2026-05-23 against the current local checkouts.

### VS Code Baseline

- VS Code checkout: `/Users/vijaysingh/code/vscode`
- Commit: `5bb22231` (`2026-05-23`)
- Relevant patterns:
  - `TreeSitterLibraryService` lazily initializes tree-sitter, caches languages,
    and caches queries by language/query kind.
  - Editor tokenization uses tree edits, old-tree incremental parse, included
    ranges, bounded parse yielding, changed-range computation, and explicit tree
    disposal.
  - Terminal/agent command parsing uses tree-sitter queries for shell structure,
    then layers command-specific parsers for behavior grammar alone cannot model.
  - Copilot parser code uses a bounded parse-tree LRU with reference-counted
    disposal and a query cache.

### Root Repo Baseline: `codingagent`

- Root repo currently has uncommitted changes in:
  - `victor/core/capability_registry.py`
  - `victor/core/indexing/ccg_builder.py`
  - `tests/unit/core/indexing/test_ccg_builder.py`
  - `tests/unit/core/test_capability_registry.py`
- Root tree-sitter usage currently exists in:
  - `victor/core/graph_rag/indexing.py`
  - `victor/core/graph_rag/language_handlers.py`
  - `victor/core/indexing/ccg_builder.py`
  - `victor/tools/code_intelligence_tool.py`
  - `victor/framework/search/codebase_embedding_bridge.py`
  - `victor/storage/vector_stores/proximadb_multi.py`
  - `victor/storage/vector_stores/code_chunking.py`
- Root still contains hardcoded tree-sitter grammar module maps and definition
  node-type maps in `victor/core/graph_rag/indexing.py`.
- Root has good degradation behavior and thread-local parser caching in graph
  indexing, but parser/query/parse-context work is duplicated across consumers.

### Sibling Repo Baseline: `victor-coding`

- Sibling repo checkout: `/Users/vijaysingh/code/victor-coding`
- Commit: `ca5c29c` (`2026-05-22`)
- `victor-coding` already owns:
  - `victor_coding/codebase/tree_sitter_manager.py`
  - `victor_coding/codebase/tree_sitter_extractor.py`
  - `victor_coding/languages/base.py`
  - `victor_coding/languages/registry.py`
  - `victor_coding/languages/plugins/*`
  - `victor_coding/codebase/ccg_builder.py`
- It already has a language plugin registry and plugin-owned
  `TreeSitterQueries`.
- Current parser cache in `tree_sitter_manager.py` is process-global by
  language. Future code should avoid sharing parser instances across worker
  threads unless protected by a lock or replaced with thread-local parsers.

## Target Ownership Model

`victor-ai` root owns orchestration, capability discovery, graph persistence,
project database writes, fallback behavior, and public framework protocols.

`victor-coding` owns tree-sitter language knowledge, grammar loading, query
compilation, language plugins, symbol extraction, edge extraction, import
extraction, and AST-aware code chunk inputs.

```text
victor-ai root
  GraphIndexingPipeline
  CodebaseEmbeddingBridge
  CodeIntelligence tools
  ProximaDB vector integration
        |
        v
  TreeSitterAnalysisProtocol
        |
        v
victor-coding
  TreeSitterService
  LanguageRegistry
  LanguagePlugin queries
  Symbol / edge / import / chunk extraction
```

## High-Level Design

### Root Framework Capability

Root should add an additive capability protocol that exposes analysis-level
operations instead of raw parser access only.

Candidate protocol location:

- Short term: `victor/framework/vertical_protocols.py`
- Later, if promoted to external contract: `victor-contracts`

Sketch:

```python
@runtime_checkable
class TreeSitterAnalysisProtocol(Protocol):
    def supports_language(self, language: str) -> bool: ...

    def parse(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str | None = None,
    ) -> Any | None: ...

    def extract_symbols(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> list[dict[str, Any]]: ...

    def extract_edges(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> list[dict[str, Any]]: ...

    def extract_imports(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str | None = None,
    ) -> list[str]: ...

    def build_chunk_context(
        self,
        content: str,
        language: str,
        *,
        file_path: str | None = None,
    ) -> Any | None: ...
```

The return type should start as `Any`/`dict` in root to avoid forcing root to
import `victor-coding` implementation types. If this graduates to
`victor-contracts`, introduce stable dataclasses there.

### `victor-coding` Service

`victor-coding` should introduce one canonical tree-sitter service.

Candidate file:

- `victor_coding/codebase/tree_sitter_service.py`

Sketch:

```python
@dataclass(frozen=True)
class ParsedSource:
    language: str
    content: bytes
    tree: Any
    root_node: Any
    file_path: str | None = None


class TreeSitterService:
    def get_language(self, language: str) -> Any: ...
    def get_parser(self, language: str) -> Any: ...
    def parse(self, content: bytes, language: str, *, file_path: str | None = None) -> ParsedSource | None: ...
    def get_query(self, language: str, query_source: str) -> Any: ...
    def run_query(self, parsed: ParsedSource, query_source: str) -> dict[str, list[Any]]: ...
```

Service responsibilities:

- Normalize language aliases (`typescriptreact` -> `tsx`, `csharp` -> `c_sharp`,
  etc.) using the language registry where possible.
- Cache `Language` objects process-wide.
- Cache `Parser` objects per thread and language.
- Cache compiled queries by `(language, query_source)`.
- Provide one parse result that can be reused by symbol extraction, edge
  extraction, import extraction, and chunk-context construction.
- Expose failure reasons through debug logs and lightweight metrics.

Compatibility wrappers in `tree_sitter_manager.py` should remain:

```python
def get_parser(language: str) -> Parser:
    return get_tree_sitter_service().get_parser(language)

def run_query(tree: Tree, query_src: str, language: str) -> dict[str, list[Node]]:
    ...
```

## Low-Level Design

### Data Contracts

Root graph code needs stable, JSON-like extraction output:

```python
{
    "name": "foo",
    "symbol_type": "function",
    "file_path": "pkg/mod.py",
    "line_start": 10,
    "line_end": 18,
    "parent_symbol": "Bar",
    "signature": "def foo(x):",
    "docstring": "optional",
    "visibility": "public",
    "ast_kind": "function_definition",
}
```

Edges should include enough metadata for resolver policy:

```python
{
    "source": "caller",
    "target": "callee",
    "edge_type": "CALLS",
    "file_path": "pkg/mod.py",
    "line_number": 42,
    "receiver_type": "Foo",
    "is_method_call": True,
}
```

### Root Graph Integration

`victor/core/graph_rag/indexing.py` should use the analysis capability in this
order:

1. Detect language.
2. Read file once as bytes.
3. Ask `TreeSitterAnalysisProtocol` for symbols and edges when the provider is
   enhanced and supports the language.
4. Convert provider dicts into root `GraphNode`/`GraphEdge`.
5. Fall back to the existing direct tree-sitter or regex path only when no
   enhanced provider is available.
6. Record fallback reason in `GraphIndexStats.errors` or debug stats.

### Structural Embedding Integration

`victor/framework/search/codebase_embedding_bridge.py` should stop loading
`load_tree_sitter_get_parser()` directly. It should request chunk context from
the analysis capability, then feed that to `TreeSitterParseContext` or a
contract-compatible equivalent.

### Vector Store Integration

`victor/storage/vector_stores/proximadb_multi.py` should build one analysis per
file and reuse it for:

- symbols
- call/reference/import extraction
- structural chunk context

This removes repeated parse work and keeps query failures coherent.

### Code Intelligence Tool Integration

`victor/tools/code_intelligence_tool.py` should preserve the current tool API,
but delegate symbol/reference lookup to the analysis capability. Python-only
queries can remain as fallback.

### Query Validation

`victor-coding` should add tests that:

- discover all language plugins,
- skip languages whose grammar wheel is missing,
- compile every non-empty query string,
- parse a small fixture for supported languages,
- fail clearly when query capture names drift from extractor expectations.

### Parser And Tree Lifetime

Python tree-sitter does not require explicit `Tree.delete()` like the WASM API,
but the implementation should still bound memory:

- Avoid process-global parser objects shared across threads.
- Prefer thread-local parser caches.
- Add a small optional per-run parse-result LRU only where repeated analysis of
  identical content is measured.
- Do not store parse trees in project/global databases.

## Migration Backlog

### Phase TSA-0: Design And Tracker

Status: In Progress

- [x] Clone and inspect VS Code tree-sitter usage.
- [x] Inspect root `codingagent` tree-sitter call sites.
- [x] Inspect sibling `victor-coding` tree-sitter/plugin call sites.
- [x] Create persistent design/progress tracker.
- [ ] Confirm whether current dirty root files are user work or available for
  this task.

Exit criteria:

- This file documents HLD, LLD, repo split, phases, and verification gates.

### Phase TSA-1: `victor-coding` TreeSitterService

Status: Not Started

- [ ] Add `victor_coding/codebase/tree_sitter_service.py`.
- [ ] Implement language normalization.
- [ ] Implement process-wide language cache.
- [ ] Implement thread-local parser cache.
- [ ] Implement compiled query cache.
- [ ] Update `tree_sitter_manager.py` to delegate to the service.
- [ ] Add service unit tests.

Exit criteria:

- Existing `get_parser()` and `run_query()` behavior remains compatible.
- Parser cache is not shared across worker threads.
- Query compilation is cached and test-covered.

### Phase TSA-2: `victor-coding` Analysis Provider

Status: Not Started

- [ ] Add provider implementing root `TreeSitterAnalysisProtocol`.
- [ ] Refactor `TreeSitterExtractor` to use `TreeSitterService`.
- [ ] Add symbol extraction from already-parsed source.
- [ ] Add edge extraction from already-parsed source.
- [ ] Add import extraction from already-parsed source.
- [ ] Add chunk-context builder from already-parsed source.
- [ ] Register provider through existing capability entry point.

Exit criteria:

- Root can request language analysis without importing `victor_coding` statically.
- Existing plugin query definitions remain source of truth.

### Phase TSA-3: Root Protocol And Stubs

Status: In Progress

- [x] Add `TreeSitterAnalysisProtocol`.
- [x] Add null/stub provider.
- [x] Register stub during bootstrap.
- [x] Add focused protocol/bootstrap tests.
- [ ] Add broader capability registry tests if the existing dirty test file is
  available for this task.
- [ ] Add boundary tests preventing direct `victor_coding` imports.

Exit criteria:

- Root imports and tests pass without `victor-coding`.
- Capability is available as a stub when no enhanced provider exists.

### Phase TSA-4: Root Graph Indexing Migration

Status: Not Started

- [ ] Convert provider symbol dicts into `GraphNode`.
- [ ] Convert provider edge dicts into `GraphEdge`.
- [ ] Prefer provider-backed extraction in `_parse_file_sync`.
- [ ] Keep direct tree-sitter and regex fallback.
- [ ] Add tests for enhanced provider, missing provider, and provider failure.

Exit criteria:

- Root graph indexing no longer needs hardcoded language definition maps for the
  enhanced path.
- Fallback behavior remains intact.

### Phase TSA-5: Shared Parse Reuse Across Embeddings And Vector Stores

Status: Not Started

- [ ] Update `codebase_embedding_bridge.py`.
- [ ] Update `proximadb_multi.py`.
- [ ] Ensure structural chunking can consume capability-built context.
- [ ] Add regression tests for one-parse-per-file behavior where practical.

Exit criteria:

- Symbol, edge, import, and chunk extraction no longer reparse the same file in
  normal enhanced-provider flows.

### Phase TSA-6: Tool And Docs Cleanup

Status: Not Started

- [ ] Update `code_intelligence_tool.py` to delegate to analysis provider.
- [ ] Document tree-sitter provider ownership in development docs.
- [ ] Update release/migration notes if public behavior changes.
- [ ] Run repo hygiene check if docs navigation or links change.

Exit criteria:

- Code intelligence behavior is provider-backed and can become multi-language.
- Documentation reflects root/plugin ownership boundaries.

## Verification Gates

### Root `codingagent`

Focused commands:

```bash
pytest tests/unit/core/test_tree_sitter_capability_pattern.py
pytest tests/unit/core/test_capability_registry.py
pytest tests/unit/core/graph_rag/test_indexing.py
pytest tests/unit/storage/vector_stores/test_code_chunking.py
pytest tests/unit/commands/test_codebase_analyzer_loader.py
```

Broader commands:

```bash
make test-definition-boundaries
python scripts/ci/repo_hygiene_check.py
```

### Sibling `victor-coding`

Focused commands:

```bash
pytest tests/indexers/test_tree_sitter_manager.py
pytest tests/indexers/test_tree_sitter_extractor.py
pytest tests/indexers/test_indexer_multilang_tree_sitter.py
pytest tests/indexers/test_indexer_edge_extraction.py
pytest tests/codebase/test_ccg_builder.py
```

New tests to add:

```bash
pytest tests/indexers/test_tree_sitter_service.py
pytest tests/languages/test_tree_sitter_query_compilation.py
```

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Parser instances shared across threads | Use thread-local parser cache in `victor-coding` service |
| Query drift across grammar package versions | Add query compilation tests for every plugin query |
| Root starts depending on `victor_coding` internals | Keep entry-point/capability discovery and boundary tests |
| Performance regression from provider indirection | Reuse one parsed source per file and cache compiled queries |
| Missing optional grammar wheels break indexing | Preserve graceful fallback and debug logging |
| Contract churn before API stabilizes | Keep first protocol in root; promote to `victor-contracts` only after implementation settles |

## Open Questions

1. Should `TreeSitterAnalysisProtocol` remain root-only for one release before
   moving to `victor-contracts`?
2. Should root keep direct tree-sitter fallback long term, or eventually rely on
   `victor-coding` for all enhanced parsing?
3. Should parse-result reuse be scoped to one file-processing call or use a
   bounded per-run LRU?
4. Should `victor-coding` expose dataclasses from its provider, or should all
   capability outputs stay JSON-like dictionaries until contracts stabilize?

## Session Log

### 2026-05-23

- Cloned VS Code into `/Users/vijaysingh/code/vscode`.
- Reviewed VS Code tree-sitter architecture:
  - editor tree service,
  - tokenization implementation,
  - command parser,
  - Copilot parser/query caches.
- Reviewed root `codingagent` tree-sitter consumers.
- Reviewed sibling `victor-coding` tree-sitter manager, extractor, language
  registry, language plugins, and CCG builder.
- Created this persistent HLD/LLD and implementation tracker.
- Started Phase TSA-3 in root `codingagent`:
  - added `TreeSitterAnalysisProtocol`,
  - added `NullTreeSitterAnalysis`,
  - registered the stub in `bootstrap_capabilities()`,
  - added focused tests in `tests/unit/core/test_tree_sitter_analysis_protocol.py`.
- Validation passed:
  - `pytest tests/unit/core/test_tree_sitter_analysis_protocol.py`
  - `pytest tests/unit/core/test_tree_sitter_capability_pattern.py`
  - `pytest tests/unit/core/test_capability_registry.py`
  - `python scripts/ci/repo_hygiene_check.py`
  - `python -m ruff check victor/contrib/parsing/analysis.py victor/contrib/parsing/__init__.py victor/core/bootstrap.py victor/framework/vertical_protocols.py tests/unit/core/test_tree_sitter_analysis_protocol.py`
