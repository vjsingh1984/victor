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

//! High-performance cosine similarity computation.
//!
//! This module provides SIMD-optimized cosine similarity calculations
//! for embedding vectors, commonly used in semantic search and tool selection.
//!
//! Features:
//! - Portable SIMD using the `wide` crate (works on x86, ARM, etc.)
//! - Batch processing with parallelization via rayon
//! - Memory-efficient operations avoiding unnecessary allocations

use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f32x8;

const EPSILON: f32 = 1e-9;
const SIMD_WIDTH: usize = 8;

/// Compute the L2 norm of a vector using SIMD.
#[inline]
fn simd_norm(vec: &[f32]) -> f32 {
    let chunks = vec.chunks_exact(SIMD_WIDTH);
    let remainder = chunks.remainder();

    let mut sum = f32x8::ZERO;
    for chunk in chunks {
        let v = f32x8::from(unsafe { *(chunk.as_ptr() as *const [f32; 8]) });
        sum += v * v;
    }

    let mut total: f32 = sum.reduce_add();

    // Handle remainder
    for &x in remainder {
        total += x * x;
    }

    total.sqrt().max(EPSILON)
}

/// Compute dot product using SIMD.
#[inline]
fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let a_chunks = a.chunks_exact(SIMD_WIDTH);
    let b_chunks = b.chunks_exact(SIMD_WIDTH);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    let mut sum = f32x8::ZERO;
    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let av = f32x8::from(unsafe { *(a_chunk.as_ptr() as *const [f32; 8]) });
        let bv = f32x8::from(unsafe { *(b_chunk.as_ptr() as *const [f32; 8]) });
        sum += av * bv;
    }

    let mut total: f32 = sum.reduce_add();

    // Handle remainder
    for (a, b) in a_remainder.iter().zip(b_remainder.iter()) {
        total += a * b;
    }

    total
}

/// Compute cosine similarity between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity value between -1 and 1
///
/// # Raises
/// * `ValueError` - If vectors have different lengths
#[pyfunction]
pub fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vectors must have same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    if a.is_empty() {
        return Ok(0.0);
    }

    let norm_a = simd_norm(&a);
    let norm_b = simd_norm(&b);
    let dot = simd_dot(&a, &b);

    Ok(dot / (norm_a * norm_b))
}

/// Compute cosine similarity between a query vector and multiple corpus vectors.
///
/// This function is optimized for the common case of comparing one query
/// against many candidates, using parallel processing for large corpora.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `corpus` - List of corpus embedding vectors
///
/// # Returns
/// List of similarity scores, one per corpus vector
///
/// # Raises
/// * `ValueError` - If query dimension doesn't match corpus dimensions
#[pyfunction]
pub fn batch_cosine_similarity(query: Vec<f32>, corpus: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    if corpus.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions
    let query_dim = query.len();
    for (i, vec) in corpus.iter().enumerate() {
        if vec.len() != query_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: query has {} dims, corpus[{}] has {} dims",
                query_dim,
                i,
                vec.len()
            )));
        }
    }

    // Pre-compute query norm
    let query_norm = simd_norm(&query);

    // Use parallel processing for larger corpora
    let results: Vec<f32> = if corpus.len() > 100 {
        corpus
            .par_iter()
            .map(|vec| {
                let corpus_norm = simd_norm(vec);
                let dot = simd_dot(&query, vec);
                dot / (query_norm * corpus_norm)
            })
            .collect()
    } else {
        corpus
            .iter()
            .map(|vec| {
                let corpus_norm = simd_norm(vec);
                let dot = simd_dot(&query, vec);
                dot / (query_norm * corpus_norm)
            })
            .collect()
    };

    Ok(results)
}

/// Find top-k most similar vectors from a corpus.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `corpus` - List of corpus embedding vectors
/// * `k` - Number of top results to return
///
/// # Returns
/// List of (index, similarity) tuples, sorted by similarity descending
#[pyfunction]
#[pyo3(signature = (query, corpus, k = 10))]
pub fn top_k_similar(
    query: Vec<f32>,
    corpus: Vec<Vec<f32>>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let similarities = batch_cosine_similarity(query, corpus)?;

    // Create (index, similarity) pairs
    let mut indexed: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();

    // Partial sort for efficiency - only need top k
    let k = k.min(indexed.len());
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top k and sort them
    let mut top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(top_k)
}

/// Normalize vectors in-place to unit length.
///
/// This is useful for pre-processing embeddings before similarity computation,
/// reducing redundant norm calculations in batch operations.
///
/// # Arguments
/// * `vectors` - List of vectors to normalize in-place
///
/// # Returns
/// List of normalized vectors (unit length)
#[pyfunction]
pub fn batch_normalize_vectors(vectors: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }

    // Use parallel processing for larger batches
    let results: Vec<Vec<f32>> = if vectors.len() > 100 {
        vectors
            .par_iter()
            .map(|vec| normalize_vector(vec))
            .collect()
    } else {
        vectors.iter().map(|vec| normalize_vector(vec)).collect()
    };

    Ok(results)
}

/// Normalize a single vector to unit length using SIMD.
#[inline]
fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm = simd_norm(vec);
    vec.iter().map(|&x| x / norm).collect()
}

/// Compute cosine similarities between a query and pre-normalized corpus vectors.
///
/// This is faster than batch_cosine_similarity when the corpus is already normalized,
/// as it avoids redundant norm calculations.
///
/// # Arguments
/// * `query` - Query embedding vector (will be normalized internally)
/// * `normalized_corpus` - List of pre-normalized corpus vectors (unit length)
///
/// # Returns
/// List of similarity scores, one per corpus vector
#[pyfunction]
pub fn batch_cosine_similarity_normalized(
    query: Vec<f32>,
    normalized_corpus: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    if normalized_corpus.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions
    let query_dim = query.len();
    for (i, vec) in normalized_corpus.iter().enumerate() {
        if vec.len() != query_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: query has {} dims, corpus[{}] has {} dims",
                query_dim,
                i,
                vec.len()
            )));
        }
    }

    // Normalize query once
    let query_normalized = normalize_vector(&query);

    // For pre-normalized vectors, similarity is just dot product
    let results: Vec<f32> = if normalized_corpus.len() > 100 {
        normalized_corpus
            .par_iter()
            .map(|vec| simd_dot(&query_normalized, vec))
            .collect()
    } else {
        normalized_corpus
            .iter()
            .map(|vec| simd_dot(&query_normalized, vec))
            .collect()
    };

    Ok(results)
}

/// Find top-k similar vectors from a pre-normalized corpus.
///
/// More efficient version of top_k_similar when corpus is already normalized.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `normalized_corpus` - List of pre-normalized corpus vectors
/// * `k` - Number of top results to return
///
/// # Returns
/// List of (index, similarity) tuples, sorted by similarity descending
#[pyfunction]
#[pyo3(signature = (query, normalized_corpus, k = 10))]
pub fn top_k_similar_normalized(
    query: Vec<f32>,
    normalized_corpus: Vec<Vec<f32>>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let similarities = batch_cosine_similarity_normalized(query, normalized_corpus)?;

    // Use a binary heap for efficient top-k selection
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct ScoredIndex(usize, f32);

    impl Eq for ScoredIndex {}

    impl Ord for ScoredIndex {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse order for min-heap behavior
            other.1.partial_cmp(&self.1).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for ScoredIndex {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let k = k.min(similarities.len());
    let mut heap: BinaryHeap<ScoredIndex> = BinaryHeap::with_capacity(k + 1);

    for (idx, sim) in similarities.into_iter().enumerate() {
        heap.push(ScoredIndex(idx, sim));
        if heap.len() > k {
            heap.pop(); // Remove smallest
        }
    }

    // Extract and sort results
    let mut results: Vec<(usize, f32)> = heap.into_iter().map(|si| (si.0, si.1)).collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(v.clone(), v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let corpus = vec![
            vec![1.0, 0.0, 0.0],  // identical
            vec![0.0, 1.0, 0.0],  // orthogonal
            vec![-1.0, 0.0, 0.0], // opposite
        ];
        let sims = batch_cosine_similarity(query, corpus).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 1e-6);
        assert!(sims[1].abs() < 1e-6);
        assert!((sims[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_similar() {
        let query = vec![1.0, 0.0];
        let corpus = vec![
            vec![0.5, 0.5],
            vec![1.0, 0.0], // Most similar
            vec![0.0, 1.0],
            vec![0.9, 0.1], // Second most similar
        ];
        let top = top_k_similar(query, corpus, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // Index of most similar
        assert_eq!(top[1].0, 3); // Index of second most similar
    }

    #[test]
    fn test_simd_norm() {
        // Test with various sizes to exercise SIMD and remainder
        let v3 = vec![3.0, 4.0, 0.0];
        assert!((simd_norm(&v3) - 5.0).abs() < 1e-6);

        let v9 = vec![1.0; 9];
        assert!((simd_norm(&v9) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(a, b).is_err());
    }

    #[test]
    fn test_batch_normalize_vectors() {
        let vectors = vec![
            vec![3.0, 4.0], // norm = 5
            vec![1.0, 0.0], // norm = 1
            vec![0.0, 2.0], // norm = 2
        ];
        let normalized = batch_normalize_vectors(vectors).unwrap();

        assert_eq!(normalized.len(), 3);

        // Check first vector is normalized to unit length
        let norm0: f32 = normalized[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm0 - 1.0).abs() < 1e-6);

        // Check values
        assert!((normalized[0][0] - 0.6).abs() < 1e-6);
        assert!((normalized[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity_normalized() {
        // Pre-normalized vectors (unit length)
        let corpus = vec![
            vec![1.0, 0.0, 0.0], // unit vector along x
            vec![0.0, 1.0, 0.0], // unit vector along y
            vec![0.0, 0.0, 1.0], // unit vector along z
        ];
        let query = vec![1.0, 0.0, 0.0]; // Will be normalized internally

        let sims = batch_cosine_similarity_normalized(query, corpus).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 1e-6); // Identical
        assert!(sims[1].abs() < 1e-6); // Orthogonal
        assert!(sims[2].abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_top_k_similar_normalized() {
        // Pre-normalized vectors
        let corpus = vec![
            vec![0.6, 0.8], // ~53 degrees from x-axis
            vec![1.0, 0.0], // x-axis (most similar to query)
            vec![0.0, 1.0], // y-axis (least similar)
            vec![0.8, 0.6], // ~37 degrees (second most similar)
        ];
        let query = vec![1.0, 0.0]; // x-axis

        let top = top_k_similar_normalized(query, corpus, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // Index 1 is most similar (x-axis)
        assert_eq!(top[1].0, 3); // Index 3 is second most similar
    }

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
