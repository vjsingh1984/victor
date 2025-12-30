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

// Allow false positive from pyo3 macro-generated code
#![allow(clippy::useless_conversion)]

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
//! - `streaming_filter`: Fast thinking token detection for streaming
//! - `thinking`: Thinking pattern detection for breaking loops
//! - `classifier`: Fast task classification with weighted patterns
//! - `chunking`: High-performance document chunking for RAG
//! - `secrets`: High-performance secret detection using compiled regex
//! - `pattern_match`: Aho-Corasick multi-pattern matching for tool/intent detection

use pyo3::prelude::*;

mod chunking;
mod classifier;
mod dedup;
mod hashing;
mod json_repair;
mod pattern_match;
mod secrets;
mod similarity;
mod streaming_filter;
mod thinking;

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
    m.add_function(wrap_pyfunction!(similarity::batch_normalize_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(
        similarity::batch_cosine_similarity_normalized,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(similarity::top_k_similar_normalized, m)?)?;

    // JSON repair functions
    m.add_function(wrap_pyfunction!(json_repair::repair_json, m)?)?;
    m.add_function(wrap_pyfunction!(json_repair::extract_json_objects, m)?)?;

    // Hashing functions
    m.add_function(wrap_pyfunction!(hashing::compute_signature, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::compute_batch_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::signature_similarity, m)?)?;

    // Streaming filter classes and functions
    m.add_class::<streaming_filter::StreamingFilter>()?;
    m.add_class::<streaming_filter::StreamingChunkResult>()?;
    m.add_function(wrap_pyfunction!(
        streaming_filter::strip_thinking_tokens,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        streaming_filter::contains_thinking_tokens,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(streaming_filter::find_thinking_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(
        streaming_filter::extract_thinking_content,
        m
    )?)?;

    // Thinking pattern detection classes and functions
    m.add_class::<thinking::ThinkingDetector>()?;
    m.add_class::<thinking::PatternAnalysis>()?;
    m.add_function(wrap_pyfunction!(thinking::detect_circular_phrases, m)?)?;
    m.add_function(wrap_pyfunction!(thinking::count_circular_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(thinking::find_circular_patterns, m)?)?;

    // Task classification classes and functions
    m.add_class::<classifier::TaskClassifier>()?;
    m.add_class::<classifier::ClassificationResult>()?;
    m.add_class::<classifier::TaskType>()?;
    m.add_function(wrap_pyfunction!(classifier::classify_task, m)?)?;
    m.add_function(wrap_pyfunction!(classifier::has_action_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(classifier::has_analysis_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(classifier::has_generation_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(classifier::has_negation, m)?)?;
    m.add_function(wrap_pyfunction!(classifier::find_all_keywords, m)?)?;

    // Document chunking functions (HIGH impact, LOW complexity)
    m.add_function(wrap_pyfunction!(chunking::chunk_by_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::chunk_by_chars, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::chunk_by_paragraphs, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::detect_doc_type, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::count_tokens_approx, m)?)?;

    // Secret detection functions (HIGH impact, MEDIUM complexity)
    m.add_class::<secrets::SecretMatch>()?;
    m.add_function(wrap_pyfunction!(secrets::scan_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::has_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::get_secret_types, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::mask_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::list_secret_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::scan_secrets_summary, m)?)?;

    // Pattern matching functions (HIGH impact, MEDIUM complexity)
    m.add_class::<pattern_match::PatternMatcher>()?;
    m.add_class::<pattern_match::PatternMatch>()?;
    m.add_function(wrap_pyfunction!(pattern_match::contains_any_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_match::find_all_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_match::count_pattern_matches, m)?)?;
    m.add_function(wrap_pyfunction!(
        pattern_match::get_matched_pattern_indices,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pattern_match::batch_contains_any, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_match::weighted_pattern_score, m)?)?;

    Ok(())
}
