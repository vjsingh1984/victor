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

//! High-performance embedding operations for v0.4.0.
//!
//! This module provides:
//! - Quantized embeddings (int8) for memory-efficient storage
//! - Matrix operations for embedding transformations
//! - Approximate nearest neighbor search using product quantization
//! - Batch embedding operations with SIMD acceleration
//!
//! Memory savings with quantization:
//! - float32: 4 bytes per dimension
//! - int8: 1 byte per dimension (4x compression)
//! - For 384-dim embeddings: 1.5KB → 384 bytes per vector

use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f32x8;

const EPSILON: f32 = 1e-9;
const SIMD_WIDTH: usize = 8;

/// Quantized embedding representation using int8.
///
/// Stores scale factor for dequantization: value = quantized * scale
#[pyclass]
#[derive(Clone)]
pub struct QuantizedEmbedding {
    #[pyo3(get)]
    pub data: Vec<i8>,
    #[pyo3(get)]
    pub scale: f32,
    #[pyo3(get)]
    pub zero_point: i8,
}

#[pymethods]
impl QuantizedEmbedding {
    #[new]
    pub fn new(data: Vec<i8>, scale: f32, zero_point: i8) -> Self {
        Self { data, scale, zero_point }
    }

    /// Get the dimension of the embedding
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Dequantize back to float32
    pub fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&x| (x as f32 - self.zero_point as f32) * self.scale)
            .collect()
    }

    /// Compute dot product with another quantized embedding
    pub fn dot(&self, other: &QuantizedEmbedding) -> PyResult<f32> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Embeddings must have same dimension",
            ));
        }

        // Integer dot product, then scale
        let int_dot: i32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a as i32) * (b as i32))
            .sum();

        // Adjust for zero points and scale
        let n = self.data.len() as f32;
        let adjusted = int_dot as f32
            - (self.zero_point as f32) * other.data.iter().map(|&x| x as f32).sum::<f32>()
            - (other.zero_point as f32) * self.data.iter().map(|&x| x as f32).sum::<f32>()
            + n * (self.zero_point as f32) * (other.zero_point as f32);

        Ok(adjusted * self.scale * other.scale)
    }
}

/// Quantize a float32 embedding to int8.
///
/// Uses symmetric quantization around zero for simplicity.
///
/// # Arguments
/// * `embedding` - Float32 embedding vector
///
/// # Returns
/// QuantizedEmbedding with int8 data and scale factor
#[pyfunction]
pub fn quantize_embedding(embedding: Vec<f32>) -> QuantizedEmbedding {
    if embedding.is_empty() {
        return QuantizedEmbedding::new(Vec::new(), 1.0, 0);
    }

    // Find min/max for scaling
    let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Symmetric quantization
    let abs_max = min_val.abs().max(max_val.abs());
    let scale = abs_max / 127.0;

    let data: Vec<i8> = if scale < EPSILON {
        vec![0i8; embedding.len()]
    } else {
        embedding
            .iter()
            .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
            .collect()
    };

    QuantizedEmbedding::new(data, scale.max(EPSILON), 0)
}

/// Batch quantize multiple embeddings.
///
/// # Arguments
/// * `embeddings` - List of float32 embedding vectors
///
/// # Returns
/// List of QuantizedEmbedding objects
#[pyfunction]
pub fn batch_quantize_embeddings(embeddings: Vec<Vec<f32>>) -> Vec<QuantizedEmbedding> {
    if embeddings.len() > 100 {
        embeddings
            .par_iter()
            .map(|e| quantize_embedding(e.clone()))
            .collect()
    } else {
        embeddings
            .iter()
            .map(|e| quantize_embedding(e.clone()))
            .collect()
    }
}

/// Dequantize a quantized embedding back to float32.
#[pyfunction]
pub fn dequantize_embedding(quantized: &QuantizedEmbedding) -> Vec<f32> {
    quantized.dequantize()
}

/// Compute cosine similarity between two quantized embeddings.
///
/// This is faster than dequantizing first because we work with integers.
#[pyfunction]
pub fn quantized_cosine_similarity(a: &QuantizedEmbedding, b: &QuantizedEmbedding) -> PyResult<f32> {
    if a.data.len() != b.data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Embeddings must have same dimension",
        ));
    }

    if a.data.is_empty() {
        return Ok(0.0);
    }

    // Compute norms and dot product in integer space
    let mut dot_sum: i32 = 0;
    let mut norm_a_sq: i32 = 0;
    let mut norm_b_sq: i32 = 0;

    for (&av, &bv) in a.data.iter().zip(b.data.iter()) {
        let ai = av as i32;
        let bi = bv as i32;
        dot_sum += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    let norm_a = (norm_a_sq as f32).sqrt();
    let norm_b = (norm_b_sq as f32).sqrt();

    if norm_a < EPSILON || norm_b < EPSILON {
        return Ok(0.0);
    }

    Ok((dot_sum as f32) / (norm_a * norm_b))
}

/// Batch cosine similarity between a query and quantized corpus.
#[pyfunction]
pub fn batch_quantized_cosine_similarity(
    query: &QuantizedEmbedding,
    corpus: Vec<QuantizedEmbedding>,
) -> PyResult<Vec<f32>> {
    if corpus.is_empty() {
        return Ok(Vec::new());
    }

    let results: Vec<f32> = if corpus.len() > 100 {
        corpus
            .par_iter()
            .map(|c| quantized_cosine_similarity(query, c).unwrap_or(0.0))
            .collect()
    } else {
        corpus
            .iter()
            .map(|c| quantized_cosine_similarity(query, c).unwrap_or(0.0))
            .collect()
    };

    Ok(results)
}

/// Matrix-vector multiplication using SIMD.
///
/// Computes: result = matrix @ vector
///
/// # Arguments
/// * `matrix` - 2D matrix as list of rows (each row is a vector)
/// * `vector` - Input vector
///
/// # Returns
/// Result vector
#[pyfunction]
pub fn matmul_vector(matrix: Vec<Vec<f32>>, vector: Vec<f32>) -> PyResult<Vec<f32>> {
    if matrix.is_empty() {
        return Ok(Vec::new());
    }

    let cols = vector.len();
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != cols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Matrix row {} has {} cols, expected {}",
                i,
                row.len(),
                cols
            )));
        }
    }

    let results: Vec<f32> = if matrix.len() > 100 {
        matrix.par_iter().map(|row| simd_dot(row, &vector)).collect()
    } else {
        matrix.iter().map(|row| simd_dot(row, &vector)).collect()
    };

    Ok(results)
}

/// Batch matrix-vector multiplication.
///
/// Computes: results[i] = matrix @ vectors[i] for each vector
#[pyfunction]
pub fn batch_matmul_vector(matrix: Vec<Vec<f32>>, vectors: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    if matrix.is_empty() || vectors.is_empty() {
        return Ok(Vec::new());
    }

    let results: Vec<Vec<f32>> = if vectors.len() > 10 {
        vectors
            .par_iter()
            .map(|v| matmul_vector(matrix.clone(), v.clone()).unwrap_or_default())
            .collect()
    } else {
        vectors
            .iter()
            .map(|v| matmul_vector(matrix.clone(), v.clone()).unwrap_or_default())
            .collect()
    };

    Ok(results)
}

/// Project embeddings to lower dimension using random projection.
///
/// This is useful for dimensionality reduction while preserving distances.
///
/// # Arguments
/// * `embeddings` - List of embedding vectors
/// * `target_dim` - Target dimension (must be <= input dimension)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// List of projected embeddings
#[pyfunction]
#[pyo3(signature = (embeddings, target_dim, seed = 42))]
pub fn random_projection(
    embeddings: Vec<Vec<f32>>,
    target_dim: usize,
    seed: u64,
) -> PyResult<Vec<Vec<f32>>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let input_dim = embeddings[0].len();
    if target_dim > input_dim {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Target dimension must be <= input dimension",
        ));
    }

    // Generate random projection matrix using simple LCG
    let mut rng_state = seed;
    let scale = (1.0 / target_dim as f32).sqrt();

    let projection_matrix: Vec<Vec<f32>> = (0..target_dim)
        .map(|_| {
            (0..input_dim)
                .map(|_| {
                    // Simple LCG random number generator
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                    r * scale
                })
                .collect()
        })
        .collect();

    // Project each embedding
    batch_matmul_vector(projection_matrix, embeddings)
}

/// Compute pairwise distances for a set of embeddings.
///
/// Returns a flattened upper triangular distance matrix.
#[pyfunction]
pub fn pairwise_distances(embeddings: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let n = embeddings.len();
    if n < 2 {
        return Ok(Vec::new());
    }

    // Pre-normalize all embeddings
    let normalized: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|e| {
            let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(EPSILON);
            e.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Compute upper triangular distances
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = simd_dot(&normalized[i], &normalized[j]);
            // Convert similarity to distance: d = 1 - sim
            distances.push(1.0 - sim);
        }
    }

    Ok(distances)
}

/// Find k-nearest neighbors for each embedding in a corpus.
///
/// # Arguments
/// * `embeddings` - List of embedding vectors
/// * `k` - Number of neighbors to find
///
/// # Returns
/// List of (neighbor_indices, distances) for each embedding
#[pyfunction]
#[pyo3(signature = (embeddings, k = 5))]
pub fn knn_graph(embeddings: Vec<Vec<f32>>, k: usize) -> PyResult<Vec<(Vec<usize>, Vec<f32>)>> {
    let n = embeddings.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let k = k.min(n - 1);

    // Pre-normalize
    let normalized: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|e| {
            let norm = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(EPSILON);
            e.iter().map(|x| x / norm).collect()
        })
        .collect();

    let results: Vec<(Vec<usize>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Compute similarities to all other embeddings
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, simd_dot(&normalized[i], &normalized[j])))
                .collect();

            // Sort by similarity (descending)
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k
            let neighbors: Vec<usize> = sims.iter().take(k).map(|(idx, _)| *idx).collect();
            let distances: Vec<f32> = sims.iter().take(k).map(|(_, sim)| 1.0 - sim).collect();

            (neighbors, distances)
        })
        .collect();

    Ok(results)
}

/// SIMD dot product (internal helper)
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

    for (a, b) in a_remainder.iter().zip(b_remainder.iter()) {
        total += a * b;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_identity() {
        let embedding = vec![0.1, -0.5, 0.3, 0.8, -0.2];
        let quantized = quantize_embedding(embedding.clone());
        let dequantized = quantized.dequantize();

        // Should be approximately equal (with quantization error)
        for (orig, deq) in embedding.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.02, "Quantization error too large");
        }
    }

    #[test]
    fn test_quantized_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let qa = quantize_embedding(a);
        let qb = quantize_embedding(b);
        let sim = quantized_cosine_similarity(&qa, &qb).unwrap();
        assert!((sim - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_matmul_vector() {
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let vector = vec![2.0, 3.0];
        let result = matmul_vector(matrix, vector).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_random_projection() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        let projected = random_projection(embeddings, 2, 42).unwrap();
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].len(), 2);
        assert_eq!(projected[1].len(), 2);
    }

    #[test]
    fn test_knn_graph() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],  // Close to first
            vec![0.0, 1.0],
            vec![0.1, 0.9],  // Close to third
        ];
        let knn = knn_graph(embeddings, 2).unwrap();
        assert_eq!(knn.len(), 4);
        // First embedding's nearest neighbor should be second
        assert_eq!(knn[0].0[0], 1);
    }
}
