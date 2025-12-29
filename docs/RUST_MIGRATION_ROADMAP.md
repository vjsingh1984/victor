# Victor Framework: Rust/Maturin Migration Roadmap

**Status**: Partial Implementation (8 modules migrated)
**Last Updated**: December 2025

## Current State

Victor has **8 Rust modules** compiled via Maturin with Python fallbacks:

| Module | Purpose | Performance Gain |
|--------|---------|------------------|
| `dedup.rs` | Block-based content deduplication using xxHash3 | 50x faster |
| `similarity.rs` | SIMD-optimized cosine similarity (8-lane vectors) | 5x faster |
| `json_repair.rs` | State-machine JSON parser for malformed LLM output | 10x faster |
| `hashing.rs` | xxHash3 signature hashing for loop detection | 10x faster |
| `streaming_filter.rs` | Aho-Corasick multi-pattern matching for thinking tokens | O(n) vs O(n*m) |
| `thinking.rs` | Circular phrase detection + semantic similarity | Efficient pattern matching |
| `classifier.rs` | Aho-Corasick weighted keyword classification | O(n) vs O(n*m) |

**Build**: Maturin 0.22 + PyO3 0.22, Python 3.10+

---

## Top 5 Migration Candidates

### 1. Embedding Similarity Top-K Selection (LOW COMPLEXITY)

**Current**: NumPy batch operations
**Target**: Extend existing `similarity.rs`

```rust
// rust/src/similarity.rs additions
pub fn batch_cosine_similarity_topk(
    query: &[f32],
    corpus: &[&[f32]],
    k: usize
) -> Vec<(usize, f32)>

pub fn batch_normalize_vectors(vectors: &mut [Vec<f32>])
```

**Benefits**:
- 2-3x speedup on tool selection (45-50 tools per query)
- Heap-based top-k avoids full sort
- Quick win with minimal API changes

**Estimated Effort**: 1-2 weeks

---

### 2. Pattern Matching Registry (MEDIUM COMPLEXITY)

**Current**: 528 regex patterns across codebase (Python `re` module)
**Target**: New `pattern_matcher.rs`

```rust
// rust/src/pattern_matcher.rs
pub struct PatternRegistry {
    aho_corasick: AhoCorasick,  // Multi-pattern matching
    regex_fallback: HashMap<String, Regex>,
}

pub fn scan_secrets(content: &str) -> Vec<Match>  // 12+ patterns
pub fn validate_response(content: &str) -> ValidationResult
```

**Benefits**:
- 2-10x speedup on secret scanning, response validation
- O(n) vs O(n*m) for multi-pattern matching
- Extends existing Aho-Corasick usage

**Estimated Effort**: 3-4 weeks

---

### 3. String Normalization Pipeline (MEDIUM COMPLEXITY)

**Current**: Multi-pass text processing in Python
**Target**: Extend `dedup.rs` with single-pass streaming

```rust
// rust/src/dedup.rs additions
pub struct StreamingDeduplicator {
    context_window: CircularBuffer<Block>,
    hash_cache: HashSet<u64>,
}

pub fn deduplicate_streaming(stream: impl Iterator<Item=&str>) -> String
```

**Benefits**:
- 3-5x speedup for LLM response processing
- Single traversal vs multiple passes
- Streaming for real-time deduplication

**Estimated Effort**: 3-4 weeks

---

### 4. Graph PageRank Engine (MEDIUM COMPLEXITY)

**Current**: NetworkX Python implementation
**Target**: New `pagerank.rs`

```rust
// rust/src/pagerank.rs
pub fn compute_pagerank(
    edges: &[(usize, usize)],
    num_nodes: usize,
    iterations: u32,
    damping: f32,
) -> Vec<f32>
```

**Benefits**:
- 3-8x speedup on architecture analysis queries
- SIMD-optimized matrix operations
- Parallel iteration with rayon

**Estimated Effort**: 3-4 weeks

---

### 5. Code Indexing Acceleration (HIGH COMPLEXITY)

**Current**: Tree-sitter + Python post-processing
**Target**: Rust Tree-sitter wrapper with parallel processing

```rust
// rust/src/indexer.rs
pub fn index_codebase(
    root: &Path,
    languages: &[Language],
) -> CodebaseIndex

pub fn extract_symbols(source: &str, language: Language) -> Vec<Symbol>
```

**Benefits**:
- 2-4x speedup on large codebases (1000+ files)
- Parallel file processing with rayon
- Reduced Python-C boundary crossings

**Estimated Effort**: 6-8 weeks

---

## Implementation Roadmap

### Phase 1: Quick Wins (2 weeks)
- [ ] Add `top_k_similar()` heap selection to similarity.rs
- [ ] Add `batch_normalize_vectors()` for efficiency
- **Expected**: 10-15% tool selection speedup

### Phase 2: Pattern Optimization (4 weeks)
- [ ] Create pattern_matcher.rs for secret detection
- [ ] Expand dedup.rs with streaming consolidation
- **Expected**: 3-5x response sanitization, 5-10x secret scanning

### Phase 3: Compute Acceleration (4 weeks)
- [ ] Implement GraphPageRank module
- [ ] Optional GPU pipeline bridge (cuBLAS)
- **Expected**: 3-8x architecture analysis queries

### Phase 4: Long-term (Q2 2026)
- [ ] Full code indexer optimization
- [ ] Approximate similarity for 10K+ vector corpus
- **Expected**: 2-4x large codebase indexing

---

## Build & Testing

```bash
# Development build
cd rust
maturin develop --release

# Production wheel
maturin build --release

# Verify native extensions
python -c "from victor.native import is_native_available; print('Native:', is_native_available())"

# Benchmark
python scripts/benchmark_native.py
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking Python API | Maintain fallback wrappers; semantic versioning |
| Platform incompatibility | Test Linux/macOS/Windows; pure Python fallback always available |
| Maintenance burden | Clear documentation; automated cross-platform CI/CD |
| Diminishing returns | Prioritize CPU-bound over I/O-bound operations |

---

## Priority Summary

| Candidate | Speedup | Complexity | ROI |
|-----------|---------|------------|-----|
| **Embedding Top-K** | 2-3x | LOW | VERY HIGH |
| **Pattern Matching** | 2-10x | MEDIUM | HIGH |
| **String Normalization** | 3-5x | MEDIUM | HIGH |
| **Graph PageRank** | 3-8x | MEDIUM | HIGH |
| **Code Indexing** | 2-4x | HIGH | MEDIUM |

---

*Document version: 1.0*
