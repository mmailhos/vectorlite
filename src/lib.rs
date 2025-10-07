pub mod index;
pub mod embeddings;

pub use index::flat::FlatIndex;
pub use index::hnsw::HNSWIndex;
pub use embeddings::EmbeddingGenerator;

use serde::{Serialize, Deserialize};

pub const DEFAULT_VECTOR_DIMENSION: usize = 768;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f64,
}

pub trait VectorIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String>;
    fn delete(&mut self, id: u64) -> Result<(), String>;
    fn search(&self, query: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_vector(&self, id: u64) -> Option<&Vector>;
    fn dimension(&self) -> usize;
}

#[derive(Debug, Serialize, Deserialize)]
pub enum VectorIndexWrapper {
    Flat(FlatIndex),
    HNSW(HNSWIndex),
}

impl VectorIndex for VectorIndexWrapper {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.add(vector),
            VectorIndexWrapper::HNSW(index) => index.add(vector),  
        }
    }

    fn delete(&mut self, id: u64) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.delete(id),
            VectorIndexWrapper::HNSW(index) => index.delete(id),
        }
    }

    fn search(&self, query: &[f64], k: usize, s: SimilarityMetric) -> Vec<SearchResult> {
        match self {
            VectorIndexWrapper::Flat(index) => index.search(query, k, s),
            VectorIndexWrapper::HNSW(index) => index.search(query, k, s),
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.len(),
            VectorIndexWrapper::HNSW(index) => index.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            VectorIndexWrapper::Flat(index) => index.is_empty(),
            VectorIndexWrapper::HNSW(index) => index.is_empty(),
        }
    }

    fn get_vector(&self, id: u64) -> Option<&Vector> {
        match self {
            VectorIndexWrapper::Flat(index) => index.get_vector(id),
            VectorIndexWrapper::HNSW(index) => index.get_vector(id),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.dimension(),
            VectorIndexWrapper::HNSW(index) => index.dimension(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    DotProduct,
}

impl SimilarityMetric {
    pub fn calculate(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        
        match self {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::Euclidean => euclidean_similarity(a, b),
            SimilarityMetric::Manhattan => manhattan_similarity(a, b),
            SimilarityMetric::DotProduct => dot_product(a, b),
        }
    }
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        SimilarityMetric::Cosine
    }
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let (mut dot, mut norm_a_sq, mut norm_b_sq) = (0.0, 0.0, 0.0);

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn euclidean_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let sum_sq = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>();
    
    let distance = sum_sq.sqrt();
    
    // Convert distance to similarity: 1 / (1 + distance)
    // This ensures similarity is in range [0, 1] with 1 being identical
    1.0 / (1.0 + distance)
}

pub fn manhattan_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let distance = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>();
    
    // Convert distance to similarity: 1 / (1 + distance)
    // This ensures similarity is in range [0, 1] with 1 being identical
    1.0 / (1.0 + distance)
}

pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((euclidean_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_similarity_different_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let expected = 1.0 / (1.0 + 5.0); // 1 / (1 + sqrt(3^2 + 4^2))
        assert!((euclidean_similarity(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((manhattan_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_similarity_different_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let expected = 1.0 / (1.0 + 7.0); // 1 / (1 + |0-3| + |0-4|)
        assert!((manhattan_similarity(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let expected = 1.0 + 4.0 + 9.0; // 14.0
        assert!((dot_product(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let expected = -1.0 - 4.0 - 9.0; // -14.0
        assert!((dot_product(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_metric_enum() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        
        // Test that all metrics work
        assert!((SimilarityMetric::Cosine.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::Euclidean.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::Manhattan.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::DotProduct.calculate(&a, &b) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_metric_default() {
        assert_eq!(SimilarityMetric::default(), SimilarityMetric::Cosine);
    }

    #[test]
    fn test_vector_store_creation() {
        let vectors = vec![
            Vector { id: 0, values: vec![1.0, 2.0, 3.0] },
            Vector { id: 1, values: vec![4.0, 5.0, 6.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_vector_store_search() {
        let vectors = vec![
            Vector { id: 0, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 1, values: vec![0.0, 1.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 0.0, 1.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2, SimilarityMetric::Cosine);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_index_wrapper_serialization() {
        use serde_json;
        
        // Test FlatIndex wrapper
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 1.0, 0.0] },
        ];
        let flat_index = FlatIndex::new(3, vectors);
        let wrapper = VectorIndexWrapper::Flat(flat_index);
        
        // Serialize wrapper
        let serialized = serde_json::to_string(&wrapper).expect("Serialization should work");
        
        // Deserialize wrapper
        let deserialized: VectorIndexWrapper = serde_json::from_str(&serialized).expect("Deserialization should work");
        
        // Verify the deserialized wrapper works
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.dimension(), 3);
        assert!(!deserialized.is_empty());
        
        // Test search through the wrapper
        let query = vec![1.1, 0.1, 0.1];
        let results = deserialized.search(&query, 1, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

}