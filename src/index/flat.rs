//! # Flat Index Implementation
//!
//! This module provides a simple flat index implementation that stores vectors
//! in a linear array and performs exact search by comparing against all vectors.
//!
//! ## Performance Characteristics
//!
//! - **Search Complexity**: O(n) - must check every vector
//! - **Insert Complexity**: O(1) - append to end of array
//! - **Memory Usage**: Linear with dataset size
//! - **Accuracy**: 100% - exact search results
//!
//! ## Use Cases
//!
//! - Small datasets (< 10K vectors)
//! - Exact search requirements
//! - Memory-constrained environments
//! - Prototype development
//!
//! # Examples
//!
//! ```rust
//! use vectorlite::{FlatIndex, Vector, SimilarityMetric, VectorIndex};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut index = FlatIndex::new(3, Vec::new());
//! let vector = Vector { id: 1, values: vec![1.0, 2.0, 3.0] };
//! 
//! index.add(vector)?;
//! let results = index.search(&[1.1, 2.1, 3.1], 5, SimilarityMetric::Cosine);
//! # Ok(())
//! # }
//! ```

use crate::{Vector, VectorIndex, SearchResult, SimilarityMetric};
use serde::{Serialize, Deserialize};


/// Flat index implementation for exact vector search
///
/// This index stores vectors in a simple array and performs linear search
/// to find the most similar vectors. While slower than approximate methods,
/// it guarantees exact results.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{FlatIndex, Vector, SimilarityMetric, VectorIndex};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut index = FlatIndex::new(3, Vec::new());
/// let vector = Vector { id: 1, values: vec![1.0, 2.0, 3.0] };
/// 
/// index.add(vector)?;
/// let results = index.search(&[1.1, 2.1, 3.1], 5, SimilarityMetric::Cosine);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct FlatIndex {
    /// Dimension of vectors stored in this index
    pub dim: usize,
    /// Storage for all vectors
    pub data: Vec<Vector>,
}

impl FlatIndex {
    pub fn new(dim: usize, data: Vec<Vector>) -> Self {
        Self { 
            dim, 
            data,
        }
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dim {
            return Err("Vector dimension mismatch".to_string());
        }
        if self.data.iter().any(|e| e.id == vector.id) {
            return Err(format!("Vector ID {} already exists", vector.id));
        }
        self.data.push(vector);
        Ok(())
    }
    
    fn delete(&mut self, id: u64) -> Result<(), String> {
        self.data.retain(|e| e.id != id);
        Ok(())
    }
    
    fn search(&self, query: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Vec<SearchResult> {
        let mut similarities: Vec<_> = self.data
        .iter()
        .map(|e| SearchResult {
            id: e.id,
            score: similarity_metric.calculate(&e.values, query)
        })
        .collect();
        
        similarities.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        similarities.truncate(k);
        similarities
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    fn get_vector(&self, id: u64) -> Option<&Vector> {
        self.data.iter().find(|e| e.id == id)
    }
    
    fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimilarityMetric;
    use serde_json;

    #[test]
    fn test_serialization_deserialization() {
        // Create a FlatIndex with some data
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 1.0, 0.0] },
            Vector { id: 3, values: vec![0.0, 0.0, 1.0] },
        ];
        let flat_index = FlatIndex::new(3, vectors);
        
        // Serialize to JSON
        let serialized = serde_json::to_string(&flat_index).expect("Serialization should work");
        
        // Deserialize from JSON
        let deserialized: FlatIndex = serde_json::from_str(&serialized).expect("Deserialization should work");
        
        // Verify the deserialized index has the same properties
        assert_eq!(deserialized.len(), 3);
        assert_eq!(deserialized.dimension(), 3);
        assert!(!deserialized.is_empty());
        
        // Verify we can retrieve vectors by ID
        assert!(deserialized.get_vector(1).is_some());
        assert!(deserialized.get_vector(2).is_some());
        assert!(deserialized.get_vector(3).is_some());
        
        // Verify search works on the deserialized index
        let query = vec![1.1, 0.1, 0.1];
        let results = deserialized.search(&query, 2, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 2);
        
        // Results should be sorted by score (highest first)
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score);
        }
        
        // The first result should be the most similar vector (ID 1)
        assert_eq!(results[0].id, 1);
        // Note: cosine similarity might not be exactly 1.0 due to floating point precision
        assert!(results[0].score > 0.99);
    }

    #[test]
    fn test_flat_index_with_cosine_similarity() {
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 1.0, 0.0] },
            Vector { id: 3, values: vec![0.0, 0.0, 1.0] },
        ];
        
        let index = FlatIndex::new(3, vectors);
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 2, SimilarityMetric::Cosine);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Most similar (identical)
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_flat_index_with_euclidean_similarity() {
        let vectors = vec![
            Vector { id: 1, values: vec![0.0, 0.0] },
            Vector { id: 2, values: vec![3.0, 4.0] },
            Vector { id: 3, values: vec![6.0, 8.0] },
        ];
        
        let index = FlatIndex::new(2, vectors);
        let query = vec![0.0, 0.0];
        let results = index.search(&query, 2, SimilarityMetric::Euclidean);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Most similar (identical)
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_flat_index_with_manhattan_similarity() {
        let vectors = vec![
            Vector { id: 1, values: vec![0.0, 0.0] },
            Vector { id: 2, values: vec![3.0, 4.0] },
            Vector { id: 3, values: vec![6.0, 8.0] },
        ];
        
        let index = FlatIndex::new(2, vectors);
        let query = vec![0.0, 0.0];
        let results = index.search(&query, 2, SimilarityMetric::Manhattan);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Most similar (identical)
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_flat_index_with_dot_product() {
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 2.0] },
            Vector { id: 2, values: vec![2.0, 1.0] },
            Vector { id: 3, values: vec![0.0, 0.0] },
        ];
        
        let index = FlatIndex::new(2, vectors);
        let query = vec![1.0, 2.0];
        let results = index.search(&query, 2, SimilarityMetric::DotProduct);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Most similar (identical)
        assert!((results[0].score - 5.0).abs() < 1e-10); // 1*1 + 2*2 = 5
    }

    #[test]
    fn test_flat_index_change_similarity_metric() {
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 2.0] },
            Vector { id: 2, values: vec![2.0, 1.0] },
        ];
        
        let index = FlatIndex::new(2, vectors);
        let query = vec![1.0, 2.0];
        
        // Test with cosine similarity
        let results_cosine = index.search(&query, 1, SimilarityMetric::Cosine);
        assert_eq!(results_cosine[0].id, 1);
        
        // Test with dot product
        let results_dot = index.search(&query, 1, SimilarityMetric::DotProduct);
        assert_eq!(results_dot[0].id, 1);
        
        // Scores should be different
        assert_ne!(results_cosine[0].score, results_dot[0].score);
    }
}
