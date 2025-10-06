pub mod index;

pub use index::flat::FlatIndex;
pub use index::hash::HashIndex;
pub use index::hnsw::HNSWIndex;

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

#[derive(Debug, Serialize, Deserialize)]
pub enum VectorIndexWrapper {
    Flat(FlatIndex),
    Hash(HashIndex),
    HNSW(HNSWIndex),
}

impl VectorIndex for VectorIndexWrapper {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.add(vector),
            VectorIndexWrapper::Hash(index) => index.add(vector),
            VectorIndexWrapper::HNSW(index) => index.add(vector),  
        }
    }

    fn delete(&mut self, id: u64) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.delete(id),
            VectorIndexWrapper::Hash(index) => index.delete(id),
            VectorIndexWrapper::HNSW(index) => index.delete(id),
        }
    }

    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult> {
        match self {
            VectorIndexWrapper::Flat(index) => index.search(query, k),
            VectorIndexWrapper::Hash(index) => index.search(query, k),
            VectorIndexWrapper::HNSW(index) => index.search(query, k),
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.len(),
            VectorIndexWrapper::Hash(index) => index.len(),
            VectorIndexWrapper::HNSW(index) => index.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            VectorIndexWrapper::Flat(index) => index.is_empty(),
            VectorIndexWrapper::Hash(index) => index.is_empty(),
            VectorIndexWrapper::HNSW(index) => index.is_empty(),
        }
    }

    fn get_vector(&self, id: u64) -> Option<&Vector> {
        match self {
            VectorIndexWrapper::Flat(index) => index.get_vector(id),
            VectorIndexWrapper::Hash(index) => index.get_vector(id),
            VectorIndexWrapper::HNSW(index) => index.get_vector(id),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.dimension(),
            VectorIndexWrapper::Hash(index) => index.dimension(),
            VectorIndexWrapper::HNSW(index) => index.dimension(),
        }
    }
}
pub trait VectorIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String>;
    fn delete(&mut self, id: u64) -> Result<(), String>;
    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_vector(&self, id: u64) -> Option<&Vector>;
    fn dimension(&self) -> usize;
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
        let results = store.search(&query, 2);
        
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
        let results = deserialized.search(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

}