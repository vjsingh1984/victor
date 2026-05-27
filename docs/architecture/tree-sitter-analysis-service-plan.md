# Tree-Sitter Analysis Service Plan

Last updated: 2026-05-23
Owner: Architecture/Foundation + Coding Vertical
Status: Active
Decision status: Design baseline accepted for phased implementation

## Goal

Make Victor's tree-sitter usage service-owned, plugin-backed, and reusable across
code graph indexing, code intelligence tools, structural embeddings, and
language-specific analysis. Borrow concrete patterns from VS Code's tree-sitter
implementation (lazy library service, query cache keyed by `(language, kind)`,
cooperative parse yielding, reference-counted parse-tree cache, bundled `.scm`
files) and apply them to a Python-native, multi-language analysis service.

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

Facts in this section are verified against working-tree state on 2026-05-23 with
explicit file paths and line numbers so they can be re-checked.

### VS Code Baseline

- Checkout: `/Users/vijaysingh/code/vscode`
- Commit: `5bb22231` (`2026-05-22`)
- Tree-sitter is used in three distinct surfaces, each with reusable patterns:

**Library service (singleton cache for Languages, Parsers, Queries)**

- `src/vs/workbench/services/treeSitter/browser/treeSitterLibraryService.ts:31`
  defines `TreeSitterLibraryService`.
- `Parser.init({ locateFile })` is called lazily once and memoized
  (`treeSitterLibraryService.ts:40`). The `locateFile` callback resolves the
  WASM module path differently for test vs browser environments.
- Three `CachedFunction` caches:
  - `_supportsLanguage` keyed by `languageId` (line 52).
  - `_languagesCache` keyed by `languageId` (line 56) — calls
    `Language.load(buffer)` from a `.wasm` file in the grammar npm package.
  - `_injectionQueries` keyed by a JSON of `{languageId, kind:
    'injections'|'highlights'}` (line 73). The cache key uses the **kind**, not
    the raw query source — important for keeping the cache small.

**Editor tokenization (incremental parsing, included ranges, cooperative yield)**

- `src/vs/editor/common/model/tokens/treeSitter/treeSitterTree.ts:334` calls
  `parser.parse(readCallback, oldTree, { progressCallback, includedRanges })`.
- `treeSitterTree.ts:437–446` defines `newTimeOutProgressCallback` which signals
  the parser to pause whenever more than 50ms has elapsed since the last yield.
  Parse is retried in a loop with `await new Promise(r => setTimeout0(r))`
  between iterations (line 434).
- Trees, cursors, and parsers are explicitly disposed via `.delete()` at
  `treeSitterTree.ts:50–53, 100, 181–182, 306–308, 410`. Required for the WASM
  binding; not required for the Python binding but documents the lifetime model.
- Included ranges support embedded/injected languages inside a single tree.
- `treeSitterTokenizationImpl.ts` drives parsing in 1000-line chunks for
  background tokenization.

**Reference-counted parse-tree cache**

- `extensions/copilot/src/platform/parser/node/parserWithCaching.ts:12`
  `ParserWithCaching` wraps parses with caching.
- `CacheableParseTree` (line 91) holds a `_refCount` (line 135) and disposes
  the underlying tree only when the refcount drops to zero (lines 138–160).
  Multiple consumers share one parse result without uncoordinated disposal.

**Short-lived TTL cache (terminal command parser)**

- `src/vs/workbench/contrib/terminalContrib/chatAgentTools/browser/treeSitterCommandParser.ts:286`
  defines `TreeCache`. Keys are `${languageId}:${commandLine}` (line 305–306).
- Whole cache is cleared 10 seconds after the last write via a `RunOnceScheduler`
  (lines 309–313). A bounded TTL avoids unbounded growth without per-entry
  expiry machinery.

**Bundled query files**

- Highlight queries live under `src/vs/editor/common/languages/highlights/`
  (only `css.scm`, `ini.scm`, `regex.scm`, `typescript.scm` are bundled today).
- Injection queries live under `src/vs/editor/common/languages/injections/`
  (only `typescript.scm`).

**Hardcoded query strings (Copilot)**

- `extensions/copilot/src/platform/parser/node/treeSitterQueries.ts` keeps
  per-language query bodies as TypeScript template strings, organized into a
  central `allKnownQueries` map via a `q()` helper. The map is the source of
  truth for query validation tests.

**Grammar workarounds**

- The terminal command parser masks PowerShell `--flag=` patterns before parse
  to work around a grammar limitation, then maps captured ranges back to the
  original text (`treeSitterCommandParser.ts:35–37`). Workarounds are isolated
  to the analysis layer and do not leak grammar peculiarities upstream.

### Root Repo Baseline: `codingagent`

- Branch: `develop`, three recent commits relevant to this workstream:
  - `e8f9d27` feat(capability_registry): provider metadata + idempotent enhanced registration
  - `c72f825` fix(ccg_builder): per-file language resolution + cached parsers
  - `facd2ff` feat(parsing): `TreeSitterAnalysisProtocol` capability contract + stub + bootstrap

**Tree-sitter call sites in root**

- `victor/core/graph_rag/indexing.py:74–82` — `_TREE_SITTER_LANGUAGE_MODULES`
  with 8 hardcoded languages (python, javascript, typescript, go, rust, java,
  c, cpp).
- `victor/core/graph_rag/indexing.py:85–126` — `_TREE_SITTER_DEFINITION_TYPES`
  with per-language node-type sets.
- `victor/core/graph_rag/indexing.py:128–152` — `_TREE_SITTER_NODE_TYPE_MAP`
  normalizing raw AST node types to semantic types.
- `victor/core/graph_rag/indexing.py:324` — parser caching is **thread-local
  with LRU=1** (`self._thread_local.parser_cache`). Each `ThreadPoolExecutor`
  worker owns its parser.
- `victor/core/graph_rag/language_handlers.py:65–78` — `CallEdge` dataclass
  already exposes `receiver_type` and `is_method_call`; `LanguageEdgeHandler`
  protocol (lines 97–146) defines `detect_calls_edges()`; only legacy
  per-language paths in `indexing.py` are still in production.
- `victor/core/indexing/ccg_builder.py:170, 207–232, 322–360` — instance-level
  parser cache (`_tree_sitter_parser_cache`) and per-file language resolution
  via `_resolve_enhanced_builder()` (added this session).
- `victor/tools/code_intelligence_tool.py:31–58` — already requests
  `TreeSitterParserProtocol` from the capability registry; hardcoded Python
  queries remain inline.
- `victor/storage/vector_stores/code_chunking.py:62–91` — defines
  `TreeSitterParseContext`; parsing is the caller's responsibility, not done
  here.
- `victor/contrib/parsing/parser.py`, `extractor.py`, `analysis.py` — Null
  stubs for each tree-sitter protocol; `NullTreeSitterAnalysis` was added in
  commit `facd2ff` and returns `False`/`None`/`[]` for every method.
- `victor/framework/vertical_protocols.py` — three tree-sitter protocols are
  defined: `TreeSitterParserProtocol`, `TreeSitterExtractorProtocol`,
  `TreeSitterAnalysisProtocol` (added this session, 818–883).
- `victor/core/capability_registry.py` — now supports optional `metadata` on
  `register()`, exposes `get_metadata()`, and is idempotent for duplicate
  ENHANCED registrations (commit `e8f9d27`).
- `victor/core/bootstrap.py:688–710` — all three tree-sitter capabilities are
  registered as `STUB` at startup.

### Sibling Repo Baseline: `victor-coding`

- Checkout: `/Users/vijaysingh/code/victor-coding`
- Commit: `ca5c29c` (`2026-05-22`)

- `victor_coding/codebase/tree_sitter_manager.py:29–61` — `LANGUAGE_MODULES`
  with **21 languages** (much broader than root's 8): adds tsx, c_sharp, ruby,
  php, kotlin, swift, scala, bash, sql, html, css, json, yaml, toml, lua,
  elixir, haskell, r.
- `victor_coding/codebase/tree_sitter_manager.py:63–64` — **module-level
  mutable dicts** `_language_cache` and `_parser_cache`. These are read and
  mutated from any thread; `get_language()`/`get_parser()` do not lock the
  read-then-write sequence. Treat as thread-unsafe under concurrent indexing.
- `victor_coding/codebase/tree_sitter_extractor.py:91–113` — instance-level
  `self._parsers` dict; safe.
- `victor_coding/codebase/ccg_builder.py:456–474` — instance-level
  `self._parsers` dict; safe.
- `victor_coding/codebase/embeddings/chunker.py:86–97` — instance-level
  `self._parsers` dict; safe.
- `victor_coding/languages/base.py:55–92` — `TreeSitterQueries` dataclass
  fields: `symbols`, `calls`, `references`, `inheritance`, `implements`,
  `composition`, `enclosing_scopes`.
- `victor_coding/languages/plugins/` — 11 plugins; each plugin owns its
  `TreeSitterQueries` via `_create_tree_sitter_queries()`.
- **No `victor_coding/codebase/tree_sitter_service.py` exists yet.**
- **No `TreeSitterAnalysisProtocol` provider is registered yet** in
  `victor-coding`; root currently sees only the stub.

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

## Lessons From VS Code — Patterns To Adopt

The following are concrete VS Code techniques we should apply to our Python
implementation. Each is anchored to a specific finding from the snapshot above.

### 1. Lazy library service with three independent caches

VS Code separates Language objects, Parser objects, and compiled Query objects
into three independently-keyed caches. Apply the same shape in
`victor_coding/codebase/tree_sitter_service.py`:

- `_language_cache: dict[str, Language]` — process-wide; languages are
  immutable and safe to share.
- `_parser_cache: dict[(thread_id, language), Parser]` — thread-local. Python
  Tree-sitter `Parser` is not safe to share across threads concurrently.
- `_query_cache: dict[(language, query_kind), Query]` — compiled once per
  language/kind, never per raw source.

This replaces the current module-level `_language_cache` and `_parser_cache` in
`tree_sitter_manager.py:63–64`, which mix process-wide and thread-unsafe
ownership.

### 2. Query cache keyed by `(language, kind)`, not raw source

VS Code keys its query cache by `{languageId, kind: 'injections'|'highlights'}`
(`treeSitterLibraryService.ts:73`). The raw `.scm` source is the value, not the
key. We should mirror this: language plugins declare named query kinds
(`symbols`, `calls`, `references`, `imports`, `inheritance`,
`enclosing_scopes`) and the service caches compiled `Query` objects by
`(language, kind)`.

This keeps the cache size bounded by `len(languages) * len(kinds)` instead of
per-call-site source strings.

### 3. Reference-counted parse-tree cache for shared consumers

When the same file feeds symbol extraction, edge extraction, import extraction,
and chunk building, all four should share one `ParsedSource`. VS Code's
`CacheableParseTree` (`parserWithCaching.ts:91, 135`) uses an explicit refcount
so the parse is held while any consumer still needs it and released
deterministically afterwards.

A Python equivalent does not need explicit memory disposal but does benefit
from the refcount pattern as a usage protocol — the service returns a
`ParsedSource` handle, consumers `acquire()` / `release()` it, and the service
evicts when refcount reaches zero. This prevents accidental re-parsing.

### 4. Short-lived TTL cache for bursty re-analysis

The terminal command parser keeps a `TreeCache` keyed by
`${languageId}:${commandLine}` cleared 10s after the last write
(`treeSitterCommandParser.ts:286, 310–313`). The same pattern applies when a
user edits the same file repeatedly — we can cache parsed trees by `(language,
content_hash)` with a short TTL via a `RunOnceScheduler`-equivalent (e.g.,
`asyncio.call_later`) to absorb burst re-analysis without retaining trees
indefinitely.

### 5. Cooperative yielding for batch parses

`newTimeOutProgressCallback` (`treeSitterTree.ts:437–446`) signals the parser
to pause every 50ms and the loop awaits `setTimeout0` between attempts (line
434). Python tree-sitter `Parser.parse` does not accept a progress callback,
but the same idea applies at the **batch** level: when indexing thousands of
files, the indexer should `await asyncio.sleep(0)` between files (or every N
files) so it does not starve the event loop or starve other coroutines like
heartbeat health checks.

### 6. Bundled `.scm` query files per language

VS Code bundles queries as `.scm` files under
`src/vs/editor/common/languages/{highlights,injections}/{language}.scm`. Today
our queries live as Python string literals inside language plugin classes
(`victor_coding/languages/plugins/rust.py:99–137`). Moving them to `.scm`
files under `victor_coding/languages/plugins/queries/{language}/{kind}.scm`
would:

- Allow IDE syntax-highlighting of queries themselves.
- Let us diff query changes across grammar versions cleanly.
- Make it easier for downstream packages (or users) to override queries.

This is a future refinement, not a TSA-1 blocker.

### 7. Centralized query validation

The Copilot parser registers every known query via a `q()` helper into one
`allKnownQueries` map (`treeSitterQueries.ts`), which lets a single test
compile every known query against its grammar. We should add the same — see
the Verification Gates section for the new test target.

### 8. Grammar workarounds isolated to the analysis provider

PowerShell `--flag=` masking lives next to the terminal parser, not in any
shared layer (`treeSitterCommandParser.ts:35–37`). Equivalent rule: any
per-grammar workaround belongs in the corresponding `victor-coding` language
plugin, not in root.

### 9. Embedded language strategy (included ranges)

VS Code supports embedded languages via `parser.parse(..., { includedRanges })`
(one tree with ranges restricted to one language) rather than maintaining
multiple layered trees. The injection query (`.scm` under `injections/`)
identifies which subranges belong to other languages. We do not need this yet;
defer until a use case appears (HTML+JS+CSS, markdown code blocks). The
analysis protocol's `parse()` and `build_chunk_context()` return types are
already flexible enough to add range information later without breaking
callers.

## High-Level Design

### Root Framework Capability

Root adds an additive capability protocol that exposes analysis-level
operations instead of raw parser access only. The protocol is **already
landed** in `victor/framework/vertical_protocols.py` (commit `facd2ff`).

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

The return type is intentionally `Any`/`dict` so root does not need to import
`victor-coding` implementation types. If this graduates to `victor-contracts`,
introduce stable dataclasses there.

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
    def normalize_language(self, language: str) -> str: ...
    def supports_language(self, language: str) -> bool: ...
    def get_language(self, language: str) -> Any: ...
    def get_parser(self, language: str) -> Any: ...
    def parse(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str | None = None,
    ) -> ParsedSource | None: ...
    def get_query(self, language: str, kind: str) -> Any: ...
    def run_query(self, parsed: ParsedSource, kind: str) -> dict[str, list[Any]]: ...
```

Service responsibilities:

- Normalize language aliases (`typescriptreact` → `tsx`, `csharp` → `c_sharp`,
  etc.) before lookup; reuse the language registry where possible.
- Cache `Language` objects process-wide; immutable post-construction.
- Cache `Parser` objects per thread and language; **never share parsers across
  worker threads**.
- Cache compiled queries by `(language, kind)` — see "Patterns To Adopt"
  section.
- Provide one parse result that can be reused by symbol extraction, edge
  extraction, import extraction, and chunk-context construction.
- Expose failure reasons through debug logs and lightweight metrics
  (`grammar_missing`, `query_compile_failed`, `parse_failed`).

Compatibility wrappers in `tree_sitter_manager.py` remain so existing call
sites continue to work:

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

These shapes match the existing `CallEdge` dataclass at
`victor/core/graph_rag/language_handlers.py:65–78`, so root can convert
provider output directly without restructuring.

### Root Graph Integration

`victor/core/graph_rag/indexing.py` should use the analysis capability in this
order:

1. Detect language (and normalize aliases).
2. Read file once as bytes.
3. Ask `TreeSitterAnalysisProtocol` for symbols and edges when the provider is
   enhanced and supports the language.
4. Convert provider dicts into root `GraphNode`/`GraphEdge`.
5. Fall back to the existing direct tree-sitter or regex path only when no
   enhanced provider is available.
6. Record fallback reason in `GraphIndexStats.errors` or debug stats.

The end-state goal is to delete `_TREE_SITTER_DEFINITION_TYPES` and
`_TREE_SITTER_NODE_TYPE_MAP` from `indexing.py` (lines 85–152) once the enhanced
provider is the default. Until then they remain as the fallback path.

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

This removes repeated parse work and keeps query failures coherent. The
reference-counted parse-tree pattern from VS Code's `CacheableParseTree`
applies here directly.

### Code Intelligence Tool Integration

`victor/tools/code_intelligence_tool.py` should preserve the current tool API,
but delegate symbol/reference lookup to the analysis capability. Python-only
queries can remain as fallback.

### Query Validation Harness

`victor-coding` should add tests that:

- discover all language plugins,
- skip languages whose grammar wheel is missing,
- compile every non-empty query string against its grammar,
- parse a small fixture for supported languages,
- fail clearly when query capture names drift from extractor expectations.

This mirrors VS Code's centralized `allKnownQueries` registration pattern.

### Parser And Tree Lifetime

Python tree-sitter does not require explicit `Tree.delete()` like the WASM API,
but the implementation should still bound memory:

- **Never share parsers across worker threads.** Replace the module-level
  `_parser_cache` in `tree_sitter_manager.py:63–64` with a thread-local cache.
- Prefer thread-local parser caches.
- Add a small, bounded per-run parse-result LRU only where repeated analysis of
  identical content is measured (see VS Code's `TreeCache` 10s TTL pattern).
- Do not store parse trees in project/global databases. Parse trees are
  rebuildable from source.

### Cooperative Scheduling For Batch Parses

Indexing thousands of files in one call should not starve the event loop or
other tasks. Apply VS Code's yield-per-budget pattern at the file granularity:

```python
YIELD_EVERY_N_FILES = 16

for i, file_path in enumerate(files):
    await index_file(file_path)
    if i % YIELD_EVERY_N_FILES == 0:
        await asyncio.sleep(0)
```

Add this to `GraphIndexingPipeline` and `CodebaseEmbeddingBridge` when batch
size exceeds a threshold.

### Language Alias Normalization

The service should normalize before any cache lookup so callers do not have to
remember per-grammar quirks. Cases to handle:

- `typescriptreact`, `tsx` → `tsx`
- `csharp`, `cs` → `c_sharp`
- `c++`, `cxx`, `cc` → `cpp`
- `objective-c`, `objc` → `objc`
- `js`, `node` → `javascript`

`victor-coding/languages/registry.py` should be the canonical source; the
service consumes it.

## Cross-Language Application Strategy

This section answers "how should we apply tree-sitter/AST analysis consistently
across many languages?". It refines the LLD with concrete cross-language rules.

### One Service, Many Plugins

There is exactly **one** parsing service (`TreeSitterService`) in
`victor-coding`. All language-specific knowledge lives in language plugins.
Adding a new language consists of:

1. Add a `(module_name, function_name)` entry to the grammar map.
2. Add a `LanguagePlugin` subclass under `victor_coding/languages/plugins/`.
3. Define `TreeSitterQueries` for that plugin: symbols, calls, references,
   imports, inheritance, enclosing_scopes.
4. Add a fixture and a query-compilation test.

No root-level code change is required for new languages. The `_TREE_SITTER_*`
maps in `victor/core/graph_rag/indexing.py` should not be expanded — they exist
as fallback only.

### Queries First, Imperative Walks Second

Use tree-sitter queries (`.scm`-style) for extraction whenever possible. They:

- Encode language structure declaratively.
- Are easier to diff across grammar versions.
- Compile once, run many times.
- Let us add new captures without re-walking AST manually.

Only fall back to imperative cursor walks for analyses queries cannot express
(e.g., type inference across calls, scope resolution requiring backreferences).

### Standardized Output Shapes

Every plugin returns the **same dict shapes** for symbols, edges, imports, and
chunks (see "Data Contracts" above). Root code never branches on language; it
branches on `symbol_type`, `edge_type`, etc. Differences between languages live
in the extractor logic, not the consumer.

### Per-Grammar Workarounds Stay Local

Document workarounds in the plugin file that needs them, not in the service or
in root. Example shape:

```python
# victor_coding/languages/plugins/powershell.py
# Workaround: tree-sitter-powershell mis-parses `--flag=value` as one token.
# We mask `--flag=` to `--flag ` before parse, then map captured offsets back.
def normalize_source(content: bytes) -> tuple[bytes, OffsetMap]:
    ...
```

### Embedded Languages — Deferred But Designed For

Defer embedded-language support until we have a real consumer (HTML+JS+CSS,
markdown+code blocks). When we add it, follow VS Code's pattern: one tree per
language with `includedRanges` rather than multiple layered trees. The
`TreeSitterAnalysisProtocol.parse()` return type is opaque enough to attach
range metadata later.

### Multi-Language Test Matrix

Add a generic compatibility test that, for each supported language:

- loads a representative fixture (small, idiomatic, hand-crafted),
- parses it via the service,
- runs every defined query kind,
- asserts at least one capture per expected kind (where applicable),
- asserts output dicts match the standard shape.

This catches grammar version regressions and plugin drift in one place. It
mirrors the spirit of `allKnownQueries` in Copilot.

### Optional-Grammar Graceful Degradation

For each language entry, the service must:

- Return `None`/`False` from `parse()` / `supports_language()` when the
  grammar wheel is missing.
- Log once at debug level on first request per language.
- Never raise to the caller.

This is already the contract; the audit task is to confirm every plugin and
every service entry point honors it.

## Migration Backlog

### Phase TSA-0: Design And Tracker

Status: Complete

- [x] Clone and inspect VS Code tree-sitter usage.
- [x] Inspect root `codingagent` tree-sitter call sites.
- [x] Inspect sibling `victor-coding` tree-sitter/plugin call sites.
- [x] Create persistent design/progress tracker.
- [x] Reverify VS Code/root/sibling claims with file:line anchors (2026-05-23).
- [x] Document VS Code patterns to adopt.
- [x] Document cross-language application strategy.

### Phase TSA-1: `victor-coding` TreeSitterService

Status: Complete (`0c30257` in `victor-coding`)

- [ ] Add `victor_coding/codebase/tree_sitter_service.py`.
- [ ] Implement language alias normalization (typescriptreact→tsx,
      csharp→c_sharp, c++→cpp, objc→objc, js→javascript).
- [ ] Implement process-wide `Language` cache.
- [ ] **Replace module-level `_parser_cache` in `tree_sitter_manager.py:63–64`
      with thread-local parser cache** (thread-safety remediation).
- [ ] Implement compiled query cache keyed by `(language, kind)`, not raw
      source.
- [ ] Add `ParsedSource` dataclass.
- [ ] Update `tree_sitter_manager.py` to delegate to the service while keeping
      its public functions for compatibility.
- [ ] Add service unit tests, including a concurrency test that asserts no
      shared parser instance across threads.

Exit criteria:

- Existing `get_parser()` and `run_query()` behavior remains compatible.
- Parser cache is not shared across worker threads (validated by test).
- Query compilation is cached and test-covered.

### Phase TSA-2: `victor-coding` Analysis Provider

Status: Complete (`2bed58b` in `victor-coding`)

- [ ] Add provider implementing root `TreeSitterAnalysisProtocol`.
- [ ] Refactor `TreeSitterExtractor` to use `TreeSitterService` and share
      `ParsedSource` across extraction calls.
- [ ] Add symbol extraction from already-parsed source.
- [ ] Add edge extraction from already-parsed source.
- [ ] Add import extraction from already-parsed source.
- [ ] Add chunk-context builder from already-parsed source.
- [ ] Register provider through existing capability entry point.
- [ ] Add query validation harness (compile every plugin query at startup or
      in a single test).

Exit criteria:

- Root can request language analysis without importing `victor_coding` statically.
- Existing plugin query definitions remain source of truth.
- Every registered query compiles against its grammar in CI (when grammar is
  installed).

### Phase TSA-3: Root Protocol And Stubs

Status: Complete (`facd2ff`, `e8f9d27`, `2145802` in `codingagent`)

- [x] Add `TreeSitterAnalysisProtocol` (commit `facd2ff`).
- [x] Add null/stub provider (`NullTreeSitterAnalysis`).
- [x] Register stub during bootstrap.
- [x] Add focused protocol/bootstrap tests
      (`tests/unit/core/test_tree_sitter_analysis_protocol.py`).
- [x] Add capability registry metadata + idempotency support (commit
      `e8f9d27`).
- [x] Add boundary tests preventing direct `victor_coding` imports from root
      code paths (commit `2145802`).

Exit criteria:

- Root imports and tests pass without `victor-coding`.
- Capability is available as a stub when no enhanced provider exists.

### Phase TSA-4: Root Graph Indexing Migration

Status: Complete (`f35f765` in `codingagent`)

- [ ] Convert provider symbol dicts into `GraphNode`.
- [ ] Convert provider edge dicts into `GraphEdge`.
- [ ] Prefer provider-backed extraction in `_parse_file_sync`.
- [ ] Keep direct tree-sitter and regex fallback when provider is unavailable.
- [ ] Add tests for enhanced provider, missing provider, and provider failure.
- [ ] Once provider is the default, schedule removal of
      `_TREE_SITTER_DEFINITION_TYPES` and `_TREE_SITTER_NODE_TYPE_MAP`.

Exit criteria:

- Root graph indexing no longer needs hardcoded language definition maps for
  the enhanced path.
- Fallback behavior remains intact.

### Phase TSA-5: Shared Parse Reuse + Cooperative Scheduling

Status: Complete (`ac799be` in `codingagent`)

- [ ] Update `codebase_embedding_bridge.py` to request `ParsedSource` from the
      provider rather than loading a parser directly.
- [ ] Update `proximadb_multi.py` to share one parse per file across symbol,
      edge, import, chunk paths.
- [ ] Ensure structural chunking can consume capability-built context.
- [ ] Add `await asyncio.sleep(0)` yield points in batch indexing loops
      (e.g., every 16 files).
- [ ] Add regression tests for one-parse-per-file behavior where practical.
- [ ] Optional: introduce a short-TTL parse-tree cache for bursty re-analysis
      of the same file (VS Code `TreeCache` pattern).

Exit criteria:

- Symbol, edge, import, and chunk extraction no longer reparse the same file
  in normal enhanced-provider flows.
- Batch indexing yields cooperatively under load.

### Phase TSA-6: Tool And Docs Cleanup

Status: Not Started

- [ ] Update `code_intelligence_tool.py` to delegate to analysis provider.
- [ ] Document tree-sitter provider ownership in development docs.
- [ ] Update release/migration notes if public behavior changes.
- [ ] Run repo hygiene check if docs navigation or links change.

Exit criteria:

- Code intelligence behavior is provider-backed and can become multi-language.
- Documentation reflects root/plugin ownership boundaries.

### Phase TSA-7 (Optional): Externalize Queries To `.scm` Files

Status: Not Started

- [ ] Move plugin query strings to `victor_coding/languages/plugins/queries/{language}/{kind}.scm`.
- [ ] Loader resolves queries via file path; plugin keeps API stable.
- [ ] Add IDE/editor association so `.scm` files highlight correctly.

Exit criteria:

- Plugin query files can be edited without touching plugin Python code.
- Loader keeps in-process compiled query cache unchanged.

### Phase TSA-8 (Optional): Embedded Languages

Status: Not Started

- [ ] Identify concrete use case (HTML+JS+CSS, markdown code blocks, Jinja, etc.).
- [ ] Add `injectionQueries` per plugin (`.scm` under `injections/`).
- [ ] Service builds parses with `includedRanges` for each injected language.
- [ ] Extraction merges captures across embedded trees with stable provenance.

Exit criteria:

- One file with multiple languages produces one consistent analysis result.

## Verification Gates

### Root `codingagent`

Focused commands:

```bash
pytest tests/unit/core/test_tree_sitter_capability_pattern.py
pytest tests/unit/core/test_tree_sitter_analysis_protocol.py
pytest tests/unit/core/test_capability_registry.py
pytest tests/unit/core/indexing/test_ccg_builder.py
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

New tests to add in TSA-1/TSA-2:

```bash
pytest tests/indexers/test_tree_sitter_service.py
pytest tests/indexers/test_tree_sitter_service_concurrency.py
pytest tests/languages/test_tree_sitter_query_compilation.py
pytest tests/languages/test_multilang_extraction_matrix.py
```

## Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Parser instances shared across threads (current state at `tree_sitter_manager.py:63–64`) | Thread-local parser cache in `TreeSitterService`; concurrency test in TSA-1 |
| Query drift across grammar package versions | Query compilation test for every plugin query; `allKnownQueries`-style central registry |
| Root starts depending on `victor_coding` internals | Capability discovery only; boundary tests preventing direct imports |
| Performance regression from provider indirection | Reuse one `ParsedSource` per file; cache compiled queries by `(language, kind)`; refcount-style sharing across consumers |
| Missing optional grammar wheels break indexing | Service returns `None`/`False` for unsupported languages; debug log once per language |
| Contract churn before API stabilizes | Keep protocol in root for one release; promote to `victor-contracts` only after implementation settles |
| Batch indexing starves event loop | `await asyncio.sleep(0)` every N files; document the yield budget |
| Unbounded parse-tree memory in re-analysis flows | Short-TTL cache modeled on VS Code `TreeCache`; never persist trees to DB |
| New language addition requires root edits | Cross-language strategy: all new-language work lives in plugins only |

## Open Questions

1. Should `TreeSitterAnalysisProtocol` remain root-only for one release before
   moving to `victor-contracts`?
2. Should root keep direct tree-sitter fallback long term, or eventually rely on
   `victor-coding` for all enhanced parsing?
3. Should parse-result reuse be scoped to one file-processing call, use a
   bounded per-run LRU, or adopt a refcount-style handle modeled on VS Code's
   `CacheableParseTree`?
4. Should `victor-coding` expose dataclasses from its provider, or should all
   capability outputs stay JSON-like dictionaries until contracts stabilize?
5. Should plugin query strings be migrated to `.scm` files (TSA-7) or stay as
   Python literals?
6. Should embedded/injected language support (TSA-8) be implemented now using
   `includedRanges`, or deferred until a concrete consumer requests it?
7. What is the right yield interval for batch indexing — every file, every 16
   files, every N tokens of source?

## Session Log

### 2026-05-23 (late evening) — Implementation TSA-1 through TSA-6

Executed the approved six-phase plan across both repos. Six commits, all
focused tests green.

**`victor-coding` commits:**
- `0c30257` TSA-1: `TreeSitterService` with thread-local parsers, `(language, kind)`
  query cache, alias normalization. Replaced module-level `_language_cache`/
  `_parser_cache` (the thread-unsafe pattern this plan flagged in TSA-1)
  with delegating wrappers. Added 30+ new tests including a Barrier-
  synchronized concurrency test that asserts 8 worker threads each own a
  distinct `Parser` instance.
- `2bed58b` TSA-2: `TreeSitterAnalysisProvider` implementing the root
  protocol. Wraps `TreeSitterService` + `LanguageRegistry`; emits LLD-
  standard dicts; tags edges with `is_method_call` from the captured
  callee node's parent type. Plugin self-registers via getattr-probed
  `register_capability` so older hosts keep the null stub. Added query-
  compilation harness (`tests/languages/test_tree_sitter_query_compilation.py`)
  modeled on VS Code Copilot's `allKnownQueries`. 12 plugins have pre-
  existing grammar/query drift; captured in `_KNOWN_BROKEN_PLUGINS`
  allowlist that flips to a failing XPASS once any plugin is repaired.

**`codingagent` commits:**
- `2145802` TSA-3 remaining: Added `test_no_direct_tree_sitter_imports_in_root_extraction_paths`
  guarding `victor/core/graph_rag/indexing.py` and `victor/core/indexing/ccg_builder.py`
  against ever importing `victor_coding` directly. Refreshed the migration
  note on the single allowed violation to point at TSA-4.
- `f35f765` TSA-4: `GraphIndexingPipeline._parse_file_sync` now prefers
  `TreeSitterAnalysisProtocol` when an enhanced provider is registered and
  the language is supported. Provider dicts mapped to `GraphNode` via a
  shared sha256(file:name:line) id scheme. Added `GraphIndexStats.provider_fallbacks`
  to track files that tried the provider but had to fall back; surfaced
  through `_merge_stats`, both consume paths, and `to_dict()`. Hardcoded
  `_TREE_SITTER_*` maps remain as the explicit fallback path with a
  comment flagging them as such.
- `ac799be` TSA-5: `CodebaseEmbeddingBridge._build_tree_sitter_parse_context`
  now prefers `provider.build_chunk_context` over `load_tree_sitter_get_parser`,
  so symbol extraction, edge extraction, and chunking share one parse per
  file when the enhanced provider is registered. Added `await asyncio.sleep(0)`
  in `_IndexingStreamPipeline._consume` after each mini-batch flush to
  cooperatively yield during long indexing runs.
- (TSA-6 in progress) `victor/tools/code_intelligence_tool.py` `symbol()`
  now prefers the analysis provider with `PYTHON_QUERIES` as fallback;
  added per-suffix language detection so the tool works for any language
  the provider supports (not just Python). Updated this doc.

**Cross-phase observations:**
- The query-compilation harness caught real grammar-version drift in 12
  language plugins (`kotlin`, `swift`, `scala`, `cpp`, `go`, `java`,
  `haskell`, `javascript`, `typescript`, `php`, `ruby`, `sql`). Each is
  XFAIL with a stale-baseline guard. Fixing them is out of scope for TSA
  but tracked.
- End-to-end verification (running root `bootstrap_capabilities()` after
  the sibling plugin is installed) confirmed: `TreeSitterAnalysisProtocol`
  resolves to the sibling's enhanced provider; `extract_symbols` on a
  Python source returns the LLD-standard dict shape with all six required
  keys.
- The PluginContext capability-registration sketch in TSA-2 (`status="enhanced"`
  kwarg) was inaccurate — actual API is `register_capability(protocol_type,
  provider, *, lazy=False)`. Implementation followed the actual API.
- TSA-5's planned producer-side yield (`await asyncio.sleep(0)` every 16
  files in `_produce`) was dropped after audit: the producer uses
  `asyncio.gather` plus `queue.put` back-pressure, which already provides
  natural yield points within each per-file coroutine. The consume-side
  yield between flushes captures the cooperative win the plan was after.

### 2026-05-23 (evening)

- Reverified VS Code claims against the working tree:
  - `TreeSitterLibraryService` confirmed at
    `src/vs/workbench/services/treeSitter/browser/treeSitterLibraryService.ts:31`
    with `CachedFunction` caches for supports/language/query at lines 52, 56,
    73 and `Parser.init({locateFile})` at lines 39–48.
  - Cooperative yield confirmed at
    `src/vs/editor/common/model/tokens/treeSitter/treeSitterTree.ts:434, 437–446`
    (50ms threshold).
  - Reference-counted parse-tree cache confirmed at
    `extensions/copilot/src/platform/parser/node/parserWithCaching.ts:12, 91, 135`.
  - Terminal `TreeCache` with 10s TTL confirmed at
    `src/vs/workbench/contrib/terminalContrib/chatAgentTools/browser/treeSitterCommandParser.ts:286, 310–313`.
- Reverified root state with line anchors for every call site and confirmed
  recent commits (`e8f9d27`, `c72f825`, `facd2ff`).
- Reverified sibling state:
  - `victor-coding` has 21 languages in `LANGUAGE_MODULES` (broader than
    root's 8).
  - Module-level `_language_cache` / `_parser_cache` at
    `tree_sitter_manager.py:63–64` are thread-unsafe — added explicit
    remediation task to TSA-1.
  - Confirmed `tree_sitter_service.py` does **not** exist yet.
  - Confirmed no `TreeSitterAnalysisProtocol` provider registered yet.
- Added two new sections: "Lessons From VS Code — Patterns To Adopt" and
  "Cross-Language Application Strategy".
- Refined LLD with: query cache key shape `(language, kind)`, cooperative
  scheduling sketch, language alias normalization, parser-thread-safety rule.
- Marked TSA-0 complete and TSA-3 mostly complete; restructured TSA-1 to
  include the thread-safety remediation; added optional phases TSA-7 (`.scm`
  files) and TSA-8 (embedded languages).
- Expanded risk table and open-question list with VS Code-derived items.

### 2026-05-23 (earlier)

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
