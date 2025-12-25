// Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Victor Native Extensions
//!
//! High-performance Rust implementations of CPU-intensive operations
//! for the Victor AI coding assistant.
//!
//! # Modules
//!
//! - `dedup`: Rolling hash-based deduplication for output processing
//! - `similarity`: SIMD-optimized cosine similarity for embeddings
//! - `json_repair`: Fast JSON repair via streaming parser
//! - `hashing`: High-performance signature hashing for loop detection

use pyo3::prelude::*;

mod dedup;
mod hashing;
mod json_repair;
mod similarity;

/// Victor Native Extensions Module
///
/// This module exposes high-performance Rust implementations to Python
/// via PyO3 bindings. All functions gracefully handle errors and return
/// appropriate Python exceptions when needed.
#[pymodule]
fn victor_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Deduplication functions
    m.add_function(wrap_pyfunction!(dedup::rolling_hash_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(dedup::normalize_block, m)?)?;
    m.add_function(wrap_pyfunction!(dedup::find_duplicate_blocks, m)?)?;

    // Similarity functions
    m.add_function(wrap_pyfunction!(similarity::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::top_k_similar, m)?)?;

    // JSON repair functions
    m.add_function(wrap_pyfunction!(json_repair::repair_json, m)?)?;
    m.add_function(wrap_pyfunction!(json_repair::extract_json_objects, m)?)?;

    // Hashing functions
    m.add_function(wrap_pyfunction!(hashing::compute_signature, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::compute_batch_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::signature_similarity, m)?)?;

    Ok(())
}
