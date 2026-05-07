# LanceDB → ProximaDB Migration Plan (Victor + arxive)

**Date**: 2026-05-07  
**Status**: Proposed — no code execution yet, awaiting explicit go-ahead before running migration on user data.  
**Companion docs**: [Category Review §5–6](ARXIV_CATEGORY_REVIEW_2026-05-05.md), [Roadmap Phase 4](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md), [Tech Debt RG-13](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md)

## Why a non-reindexing migration is feasible

LanceDB persists vectors as a `pa.list_(pa.float32(), D)` column inside an Arrow/Parquet-backed dataset. The vectors are byte-portable: any tool that can read Lance via PyArrow gets the exact same float32 array that was stored. ProximaDB-embedded's Python API accepts NumPy `float32` arrays directly via `db.insert(collection, ids=..., vectors=..., metadata=...)`. There is no transcoding step. The migration is therefore **schema-and-bytes**, not "extract → re-embed → reinsert."

## Targets

1. **arxive** (`/Users/vijaysingh/code/arxive`) — single LanceDB table `paper_chunks`, ~1.49M chunks, 384-dim vectors from BAAI/bge-small-en-v1.5.
2. **Victor** (`/Users/vijaysingh/code/codingagent`) — multiple LanceDB tables under `victor/storage/vector_stores/`, plus the existing `SqliteLanceDBMigration` helper in `proximadb_migration.py`.

## What already exists in Victor

`victor/storage/vector_stores/proximadb_migration.py` already does the right shape:

- `_migrate_vectors()` opens the Lance dataset, reads `vector` directly, and forwards it to `ProximaDBMultiModelProvider._client.insert_vectors(...)` with no re-embedding step (line 247: `vector = row.get("vector") or self.provider._zero_vector()`).
- It preserves IDs, content, and arbitrary metadata.
- It is async and ties into Victor's multi-model layout (graph + document + metric backfill).

Two improvements are required before we use it for non-trivial data sets:

| Issue | File:Line | Fix |
| --- | --- | --- |
| Loads all rows into memory at once via `table.head(row_count).to_list()` | `proximadb_migration.py:237` | Replace with a streaming iterator (Lance `to_arrow().to_batches(batch_size)`); insert in batches of 1000–5000 vectors |
| Couples vector migration with graph/document/metric backfill | `proximadb_migration.py:_migrate_*` | Keep multi-model migration as the Victor entry point; expose a smaller "vectors only" helper for arxive and other consumers |

## What does NOT yet exist for arxive

Arxive uses a plain `lancedb` API today (`indexer.py:VectorIndex`). It has no ProximaDB code path. We need:

1. A `proximadb_index.py` module mirroring the surface of `VectorIndex` (`index_paper`, `delete_paper`, `search`, `get_stats`).
2. A backend switch in `Config` (e.g. `vector_backend: Literal["lancedb", "proximadb"] = "lancedb"`) wired through `cli.py` so the user can flip between backends.
3. A migration script `scripts/migrate_lancedb_to_proximadb.py` that walks the Lance table once and bulk-inserts into ProximaDB.

## Pre-flight: install proximadb-embedded

ProximaDB-embedded requires a Rust build step:

```bash
cd /Users/vijaysingh/code/proximaDB/clients/python-embedded
# Inside the venv used by arxive and Victor
pip install maturin
maturin develop -m ../../Cargo.toml --release --features python,pylib -i python
```

Before any migration runs, verify:

```bash
python -c "import proximadb_embedded; print(proximadb_embedded.__version__)"
```

If the Rust toolchain or maturin is missing this fails fast — the user has to drive that one-time setup explicitly before we touch data.

## Migration: arxive

### Phase A0 — Add the backend swap (no migration yet)

Files to add or modify:

- `arxive/proximadb_index.py` (new) — `ProximaDBVectorIndex` that mirrors `VectorIndex` for `index_paper`, `search`, `delete_paper`, `get_stats`. Reuse the same `embed_model` lifecycle so live indexing keeps working through ProximaDB.
- `arxive/config.py` — add `vector_backend: str = "lancedb"`, `proximadb_path: Path = self.db_dir / "proximadb"`, `proximadb_collection: str = "paper_chunks"`, `proximadb_engine: str = "sst"`.
- `arxive/cli.py` — surface `--backend lancedb|proximadb` on `search`, `search-multi`, `index`, `pipeline`. Default keeps lancedb for safety.
- `arxive/pipeline.py` — pick backend via factory; everything else is unchanged.

This phase ships without touching the existing LanceDB store.

### Phase A1 — One-shot migration (read-only against LanceDB)

`scripts/migrate_lancedb_to_proximadb.py`:

1. Open the existing Lance dataset:
   ```python
   import lancedb
   from proximadb_embedded import ProximaDB

   db = lancedb.connect(str(config.vector_db_path))
   table = db.open_table("paper_chunks")
   ```

2. Create or open the target collection:
   ```python
   target = ProximaDB(data_dirs=str(config.proximadb_path))
   target.create_collection(
       config.proximadb_collection,
       dimension=config.embedding_dim,
       engine=config.proximadb_engine,
   )
   ```

3. Stream batches from Lance using PyArrow (NOT `.to_list()`):
   ```python
   arrow_dataset = table.to_lance()
   for batch in arrow_dataset.to_batches(batch_size=2000):
       df = batch.to_pandas()
       ids = df["chunk_id"].astype(str).tolist()
       vectors = np.stack(df["vector"].to_numpy()).astype(np.float32)
       metadata = df.drop(columns=["vector", "chunk_id"]).to_dict(orient="records")
       target.insert(
           config.proximadb_collection,
           ids=ids, vectors=vectors, metadata=metadata,
       )
   target.flush()
   ```

4. Verify counts and a small recall sample:
   - `target.stats().total_vectors == table.count_rows()`
   - For ~50 random query strings, run the same `search` against both backends and confirm overlap of top-K is above a threshold (e.g. ≥80% Jaccard at K=20). This is a sanity check; ranks may differ slightly if the two systems use different distance conventions.

### Phase A2 — Cutover

- Flip the default in `Config` to `vector_backend = "proximadb"`.
- Keep the lancedb path available behind the flag for at least one release as rollback insurance.
- After a soak period, optionally retire the LanceDB code path and reclaim disk.

## Migration: Victor

Victor is more complex because of multi-model backfill, but the vector half is already non-reindexing. Steps:

### Phase V1 — Streaming the existing migration

Modify `victor/storage/vector_stores/proximadb_migration.py:_migrate_vectors`:

- Replace `rows = table.head(row_count).to_list()` with an Arrow batch iterator from `table.to_lance().to_batches(...)`.
- Keep the existing record-shape and `insert_vectors` call. The provider already accepts batched records.

This preserves the full Victor migration semantics (graph + document + metric backfill) while fixing the memory-explosion case.

### Phase V2 — Optional vector-only entry point

Add a `migrate_vectors_only(provider, lancedb_dir, table_name, target_collection)` helper that bypasses graph and document logic. This is what arxive (or any other vector-only consumer) imports — it is the natural reuse seam for non-Victor callers.

### Phase V3 — Configuration switch in Victor

`victor/storage/vector_stores/registry.py` already lists ProximaDB as a registered provider. The remaining gap is making it the default for code-graph and conversation-embedding consumers, gated behind a settings flag (e.g. `embedding_backend`). This phase only flips the default; no data is moved here.

## Cross-cutting: applying the corpus advances

Beyond the migration, the new arxiv-validated work in [ARXIV_CATEGORY_REVIEW_2026-05-05.md §5–6](ARXIV_CATEGORY_REVIEW_2026-05-05.md) maps onto each repo as follows.

### Apply to ProximaDB itself

| Advance | ProximaDB landing zone | Notes |
| --- | --- | --- |
| Filter-strategy router (`2602.17914`, `2603.23710`, `2510.27141`) | Vector ops layer; `VECTOR_SEARCH` SQL extension | ProximaDB already exposes `<->` and metadata filters. The literature consensus says: pick pre- vs post- vs inline-filtering per query. Adding a learned planner above HNSW/IVF inside ProximaDB is a near-direct fit and fits its "one API surface" framing. |
| GLS selectivity metric (`2602.11443`) | Index statistics + SQL planner | A cheap statistic to ship with collection metadata so the router has a signal to use. |
| Hybrid retrieval reference architecture (`2604.16394`) | Cross-model query path (vector + document + observability) | ProximaDB already does cross-model joins; explicit reciprocal rank fusion + reranker hooks would round it out. |
| DSL-R1 hybrid query DSL (`2603.21018`) | SQL extensions / API surface | Validates that "structured operators + vector retrieval in one query" is the right product framing — ProximaDB is already on this path. |
| NaviX-style native vector index in graph DBMS (`2506.23397`) | ORION graph engine + vector engines | Reference for unifying graph traversal with vector neighborhoods natively. Long-range, not a near-term ask. |

### Apply to Victor

| Advance | Victor landing zone | Notes |
| --- | --- | --- |
| Filter-strategy router | `victor/storage/vector_stores/base.py` and provider modules | Same router lives above LanceDB / ProximaDB providers, not inside them. |
| Hybrid retrieval gateway (`2604.16394`, `2603.22587`) | New `victor/storage/retrieval/gateway.py` | Single fusion seam over BM25 (`ConversationStore` FTS5), dense ANN (providers), and structured filters. |
| NaviRAG / FlexStructRAG hierarchical retrieval (`2604.12766`, `2604.16312`) | `victor/core/graph_rag/retrieval.py` and a hierarchical-summary writer for long docs | The PageIndex idea, gated under the hybrid gateway. Reuses the existing code symbol graph as the navigable structure for code. |
| Adaptive query routing (`2604.14222`) | Hybrid gateway + `RuntimeIntelligenceService` | Higher-level router that decides which retrieval strategy (flat ANN, BM25, hierarchical) to invoke. |

## Risks and gates

- **Rust build**: ProximaDB-embedded must be built locally. If this fails, no migration runs. The user should drive `maturin develop` themselves.
- **Schema mismatch**: arxive's metadata fields are simple types (string/int) which ProximaDB accepts as JSON-typed metadata. No risk expected, but the verification step in Phase A1 must confirm specific filter clauses (`primary_category = "cs.AI"`) round-trip correctly through ProximaDB's filter API.
- **Distance semantics**: LanceDB defaults to L2; ProximaDB-embedded depends on engine choice. Verify the score conversion in `arxive/indexer.py:VectorIndex.search()` (`1 / (1 + distance)`) keeps producing comparable rankings. If not, swap to cosine on both sides.
- **Disk usage**: The migration temporarily doubles disk use (LanceDB still in place + new ProximaDB store). Expected ~6–8 GB for arxive's 1.49M × 384-dim float32 + metadata text fields. Plan free space before running.
- **Atomicity**: The migration is read-only against LanceDB and append-only against ProximaDB. A failed run can be retried after `delete_collection`. We never modify the LanceDB source until the user explicitly approves cutover.

## What I will and will not do without further approval

I will:

- write the migration script and backend abstraction inside arxive (no data writes yet — code only),
- write the Victor `_migrate_vectors` streaming patch (no data writes yet — code only),
- run unit tests against a temp directory,
- present a dry-run summary listing source row count, target collection name, and disk paths.

I will not, without an explicit "yes, run the migration":

- execute migration on the live arxive `db/vectors` data,
- execute migration on Victor's `~/.victor/embeddings` data,
- delete or rename existing LanceDB datasets,
- flip default backends in committed config files.

## Suggested branches

- `arxive`: `migration/proximadb-backend`
- Victor: `research/proximadb-vector-migration`
- proximaDB (if filter-strategy router work is taken on): `feat/filter-strategy-router`

## Related documents

- [ARXIV_CATEGORY_REVIEW_2026-05-05.md](ARXIV_CATEGORY_REVIEW_2026-05-05.md) §5 (vector storage) and §6 (PageIndex / NaviRAG)
- [Roadmap Phase 4](../roadmap/research-validated-memory-context-roadmap-2026-05-05.md) (filter-strategy router, hybrid gateway)
- [Tech Debt RG-13](../tech-debt/research-validated-memory-context-gaps-2026-05-05.md) (vector-portable storage migration)
