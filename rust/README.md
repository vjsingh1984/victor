# Victor Native Extensions

High-performance Rust implementations of CPU-intensive operations for the Victor AI coding assistant.

## Features

- **Deduplication** (`dedup`): Rolling hash-based content deduplication using xxHash3
- **Similarity** (`similarity`): SIMD-optimized cosine similarity for embeddings
- **JSON Repair** (`json_repair`): Fast JSON repair with streaming parser
- **Hashing** (`hashing`): High-performance signature hashing for loop detection

## Performance

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Block deduplication | 1.0ms | 0.02ms | ~50x |
| Cosine similarity (384d, 100 vectors) | 5.0ms | 1.0ms | ~5x |
| JSON repair | 0.5ms | 0.05ms | ~10x |
| Signature hashing | 0.2ms | 0.02ms | ~10x |

## Building

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.10+
- maturin (`pip install maturin`)

### Development Build

```bash
cd rust

# Build and install in development mode
maturin develop

# Build with optimizations
maturin develop --release

# Build wheel for distribution
maturin build --release
```

### Install Pre-built Wheel

```bash
# From the rust directory after building
pip install target/wheels/victor_native-*.whl
```

## Usage

The native extensions are automatically used when available:

```python
from victor.native import (
    rolling_hash_blocks,
    batch_cosine_similarity,
    repair_json,
    compute_signature,
    is_native_available,
)

# Check if native extensions are loaded
print(f"Native available: {is_native_available()}")

# Deduplication
blocks = rolling_hash_blocks(content, min_block_length=50)
for hash_str, block, is_duplicate in blocks:
    if is_duplicate:
        print(f"Duplicate block: {block[:50]}...")

# Similarity
query = [0.1, 0.2, ...]  # 384-dim embedding
corpus = [[...], [...], ...]  # List of embeddings
similarities = batch_cosine_similarity(query, corpus)

# JSON repair
fixed = repair_json("{'key': 'value', 'active': True}")
# '{"key": "value", "active": true}'

# Signature hashing
sig = compute_signature("read_file", {"path": "/test.py"})
```

## Fallback Behavior

If the native extension is not available (not installed or incompatible platform), the `victor.native` module automatically falls back to pure Python implementations with equivalent functionality.

```python
from victor.native import is_native_available

if is_native_available():
    print("Using Rust implementation (fast)")
else:
    print("Using Python fallback (compatible)")
```

## Architecture

```
rust/
├── Cargo.toml          # Rust package configuration
├── pyproject.toml      # maturin build configuration
├── README.md           # This file
└── src/
    ├── lib.rs          # PyO3 module definition
    ├── dedup.rs        # Deduplication module
    ├── similarity.rs   # Cosine similarity module
    ├── json_repair.rs  # JSON repair module
    └── hashing.rs      # Signature hashing module
```

## Testing

```bash
# Run Rust tests
cargo test

# Run Python integration tests
pytest ../tests/unit/test_native.py -v
```

## License

Apache License 2.0
