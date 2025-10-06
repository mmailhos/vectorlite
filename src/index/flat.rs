use crate::{Vector, VectorIndex, SearchResult, cosine_similarity};
use serde::{Serialize, Deserialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct FlatIndex {
    pub dim: usize,
    pub data: Vec<Vector>
}

impl FlatIndex {
    pub fn new(dim: usize, data: Vec<Vector>) -> Self {
        Self { dim, data }
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
    
    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult> {
        let mut similarities: Vec<_> = self.data
        .iter()
        .map(|e| SearchResult {
            id: e.id,
            score: cosine_similarity(&e.values, query)
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

#[test]
fn test_serialization_deserialization() {
    use serde_json;
    
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
    let results = deserialized.search(&query, 2);
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
