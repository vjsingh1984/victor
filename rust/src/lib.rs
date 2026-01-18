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
//! - `extractor`: High-performance tool call extraction from model output
//! - `sanitizer`: High-performance response sanitization
//! - `embeddings`: Quantized embeddings, matrix ops, KNN (v0.4.0)
//! - `yaml_loader`: Fast YAML parsing for workflow definitions (v0.4.0)
//! - `ast_indexer`: Fast stdlib detection and identifier extraction (v0.4.0)
//! - `arg_normalizer`: Fast JSON repair and type coercion (v0.4.0)
//! - `tool_selector`: High-performance tool selection with caching (v0.5.0)
//! - `ast_processor`: High-performance tree-sitter AST parsing with caching (v0.5.1)
//! - `embedding_ops`: High-performance embedding operations with SIMD (v0.5.1)
//! - `signature`: Fast tool call signature computation for deduplication (v0.5.1)
//! - `file_ops`: High-performance parallel file system operations (v0.5.1)
//! - `batch_processor`: High-performance batch processing coordinator (v0.5.1)
//! - `graph_algorithms`: High-performance graph algorithms for code analysis (v0.5.1)
//! - `serialization`: High-performance JSON/YAML parsing and serialization (v0.5.1)

use pyo3::prelude::*;

mod arg_normalizer;
mod ast_indexer;
mod ast_processor;
mod batch_processor;
mod chunking;
mod classifier;
mod dedup;
mod embeddings;
mod embedding_ops;
mod extractor;
mod file_ops;
mod graph_algorithms;
mod hashing;
mod json_repair;
mod pattern_match;
// mod regex_engine;  // TODO: Fix compilation errors in regex_engine.rs
mod sanitizer;
// mod serialization;  // TODO: Fix compilation errors in serialization.rs
mod secrets;
// mod signature;
mod similarity;
mod streaming_filter;
mod thinking;
mod tool_selector;
mod yaml_loader;

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

    // Line-aware chunking functions (Protocol-compliant)
    m.add_class::<chunking::ChunkInfoRust>()?;
    m.add_function(wrap_pyfunction!(chunking::count_lines, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::find_line_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::line_at_offset, m)?)?;
    m.add_function(wrap_pyfunction!(chunking::chunk_with_overlap, m)?)?;

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

    // Tool call extraction functions (HIGH impact, MEDIUM complexity)
    m.add_function(wrap_pyfunction!(extractor::extract_file_path, m)?)?;
    m.add_function(wrap_pyfunction!(extractor::extract_code_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(extractor::extract_shell_commands, m)?)?;
    m.add_function(wrap_pyfunction!(extractor::extract_tool_call, m)?)?;
    m.add_function(wrap_pyfunction!(extractor::batch_extract_file_paths, m)?)?;

    // Response sanitization functions (HIGH impact, MEDIUM complexity)
    m.add_function(wrap_pyfunction!(sanitizer::sanitize_response, m)?)?;
    m.add_function(wrap_pyfunction!(sanitizer::is_garbage_content, m)?)?;
    m.add_function(wrap_pyfunction!(sanitizer::detect_leakage_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(sanitizer::strip_markup, m)?)?;
    m.add_function(wrap_pyfunction!(sanitizer::validate_tool_name, m)?)?;

    // Embedding functions (v0.4.0 - quantization, matrix ops, KNN)
    m.add_class::<embeddings::QuantizedEmbedding>()?;
    m.add_function(wrap_pyfunction!(embeddings::quantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::batch_quantize_embeddings, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::dequantize_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::quantized_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(
        embeddings::batch_quantized_cosine_similarity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(embeddings::matmul_vector, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::batch_matmul_vector, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::random_projection, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::pairwise_distances, m)?)?;
    m.add_function(wrap_pyfunction!(embeddings::knn_graph, m)?)?;

    // YAML parsing functions (v0.4.0 - workflow acceleration)
    m.add_function(wrap_pyfunction!(yaml_loader::parse_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_loader::parse_yaml_with_env, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_loader::parse_yaml_file, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_loader::parse_yaml_file_with_env, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_loader::validate_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_loader::extract_workflow_names, m)?)?;

    // AST indexer functions (v0.4.0 - codebase indexing acceleration)
    m.add_function(wrap_pyfunction!(ast_indexer::is_stdlib_module, m)?)?;
    m.add_function(wrap_pyfunction!(ast_indexer::batch_is_stdlib_modules, m)?)?;
    m.add_function(wrap_pyfunction!(ast_indexer::filter_stdlib_imports, m)?)?;
    m.add_function(wrap_pyfunction!(ast_indexer::extract_identifiers, m)?)?;
    m.add_function(wrap_pyfunction!(
        ast_indexer::extract_identifiers_with_positions,
        m
    )?)?;

    // Argument normalizer functions (v0.4.0 - tool call acceleration)
    m.add_function(wrap_pyfunction!(arg_normalizer::coerce_string_type, m)?)?;
    m.add_function(wrap_pyfunction!(
        arg_normalizer::batch_coerce_string_types,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(arg_normalizer::normalize_json_string, m)?)?;
    m.add_function(wrap_pyfunction!(
        arg_normalizer::batch_normalize_json_strings,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(arg_normalizer::repair_quotes, m)?)?;
    m.add_function(wrap_pyfunction!(arg_normalizer::is_valid_json, m)?)?;
    m.add_function(wrap_pyfunction!(arg_normalizer::get_json_type, m)?)?;

    // Tool selector functions (v0.5.0 - high-performance tool selection)
    m.add_function(wrap_pyfunction!(tool_selector::cosine_similarity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(tool_selector::topk_indices, m)?)?;
    m.add_function(wrap_pyfunction!(tool_selector::filter_by_category, m)?)?;
    m.add_function(wrap_pyfunction!(tool_selector::topk_tools_by_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(tool_selector::filter_by_similarity_threshold, m)?)?;

    // AST processor classes and functions (v0.5.1 - tree-sitter acceleration)
    m.add_class::<ast_processor::PyAstProcessor>()?;
    m.add_class::<ast_processor::PyAstTree>()?;
    m.add_class::<ast_processor::PyNodeMatch>()?;
    m.add_class::<ast_processor::PyExtractedSymbol>()?;
    m.add_function(wrap_pyfunction!(ast_processor::parse_to_ast, m)?)?;
    m.add_function(wrap_pyfunction!(ast_processor::parse_to_ast_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ast_processor::execute_query, m)?)?;
    m.add_function(wrap_pyfunction!(ast_processor::extract_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(ast_processor::extract_symbols_batch, m)?)?;

    // Embedding operations (v0.5.1 - SIMD-optimized embedding operations)
    m.add_function(wrap_pyfunction!(embedding_ops::batch_cosine_similarity_simd, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_ops::topk_indices_partial, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_ops::similarity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(
        embedding_ops::batch_cosine_similarity_parallel,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(embedding_ops::batch_top_k_similar, m)?)?;
    m.add_class::<embedding_ops::EmbeddingCache>()?;

    // Signature computation (v0.5.1 - fast tool call deduplication)
//     m.add_function(wrap_pyfunction!(signature::compute_tool_call_signature, m)?)?;
//     m.add_function(wrap_pyfunction!(
//         signature::batch_compute_tool_call_signatures,
//         m
//     )?)?;
//     m.add_class::<signature::ToolCallData>()?;
//     m.add_function(wrap_pyfunction!(signature::deduplicate_tool_calls, m)?)?;
//     m.add_function(wrap_pyfunction!(signature::deduplicate_tool_calls_dict, m)?)?;
// 
//     // Regex engine (v0.5.1 - high-performance code pattern matching)
// //     m.add_class::<regex_engine::CompiledRegexSet>()?;
// //     m.add_class::<regex_engine::MatchResult>()?;
// //     m.add_function(wrap_pyfunction!(regex_engine::compile_language_patterns, m)?)?;
// //     m.add_function(wrap_pyfunction!(regex_engine::list_supported_languages, m)?)?;
// //     m.add_function(wrap_pyfunction!(regex_engine::get_language_categories, m)?)?;

    // File operations (v0.5.1 - high-performance parallel file system operations)
    m.add_class::<file_ops::FileInfo>()?;
    m.add_class::<file_ops::FileMetadata>()?;
    m.add_function(wrap_pyfunction!(file_ops::walk_directory_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::collect_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::filter_by_extension, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::filter_by_size, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::get_directory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::group_by_directory, m)?)?;
    m.add_function(wrap_pyfunction!(file_ops::filter_by_modified_time, m)?)?;

    // Graph algorithms (v0.5.1 - high-performance graph algorithms for code analysis)
    m.add_class::<graph_algorithms::Graph>()?;
    m.add_function(wrap_pyfunction!(graph_algorithms::graph_from_edge_list, m)?)?;
    m.add_function(wrap_pyfunction!(graph_algorithms::graph_from_adjacency_matrix, m)?)?;

    // Batch processor (v0.5.1 - high-performance parallel task coordination)
    m.add_class::<batch_processor::BatchTask>()?;
    m.add_class::<batch_processor::BatchResult>()?;
    m.add_class::<batch_processor::BatchProcessor>()?;
    m.add_class::<batch_processor::BatchProcessSummary>()?;
    m.add_class::<batch_processor::BatchProgress>()?;
    m.add_function(wrap_pyfunction!(batch_processor::create_task_batches, m)?)?;
    m.add_function(wrap_pyfunction!(batch_processor::merge_batch_summaries, m)?)?;
    m.add_function(wrap_pyfunction!(batch_processor::calculate_optimal_batch_size, m)?)?;
    m.add_function(wrap_pyfunction!(batch_processor::estimate_batch_duration, m)?)?;

    // Serialization (v0.5.1 - high-performance JSON/YAML parsing and serialization)
    Ok(())
}
