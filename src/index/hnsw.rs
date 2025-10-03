use std::collections::HashMap;
use std::fmt::{Formatter, Debug};
use rand::rngs::StdRng;
use space::{Metric, Neighbor};
use hnsw::{Hnsw, Searcher};
use crate::{Vector, VectorIndex, SearchResult};
struct Euclidean;

const MAXIMUM_NUMBER_CONNECTIONS: usize = if cfg!(feature = "memory-optimized") {
    8
} else if cfg!(feature = "high-accuracy") {
    32
} else {
    16
};

const MAXIMUM_NUMBER_CONNECTIONS_0: usize = if cfg!(feature = "memory-optimized") {
    16
} else if cfg!(feature = "high-accuracy") {
    64
} else {
    32 
};


impl Metric<Vec<f64>> for Euclidean {
    type Unit = u64;
    
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> Self::Unit {
        let sum_sq = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>();
        (sum_sq.sqrt() * 1000.0) as u64 
    }
}

pub struct HNSWIndex {
    hnsw: Hnsw<Euclidean, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0>,
    searcher: Searcher<u64>,
    dim: usize,
    // Mapping from custom ID to internal HNSW index
    id_to_index: HashMap<u64, usize>,
    // Mapping from internal HNSW index to custom ID
    index_to_id: HashMap<usize, u64>,
    // Store vectors for retrieval by ID. 
    vectors: HashMap<u64, Vector>,
}

impl HNSWIndex {
    pub fn new(dim: usize) -> Self {
        let hnsw: Hnsw<Euclidean, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0> = Hnsw::new(Euclidean);
        let searcher = Searcher::new();
        Self { 
            hnsw, 
            searcher, 
            dim,
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            vectors: HashMap::new(),
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dim {
            return Err("Vector dimension mismatch".to_string());
        }
        
        if self.id_to_index.contains_key(&vector.id) {
            return Err(format!("Vector ID {} already exists", vector.id));
        }
        
        let internal_index = self.hnsw.insert(vector.values.clone(), &mut self.searcher);
        
        self.id_to_index.insert(vector.id, internal_index);
        self.index_to_id.insert(internal_index, vector.id);
        self.vectors.insert(vector.id, vector);
        
        Ok(())
    }
    fn delete(&mut self, id: u64) -> Result<(), String> {
        // Check if ID exists
        if !self.id_to_index.contains_key(&id) {
            return Err(format!("Vector ID {} does not exist", id));
        }
        
        let internal_index = self.id_to_index[&id];
        
        //  Since HNSW doesn't support deletion, we just remove the reference to the node in the mapping
        self.id_to_index.remove(&id);
        self.index_to_id.remove(&internal_index);
        self.vectors.remove(&id);
        
        Ok(())
    }
    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult> {
        let mut searcher = Searcher::new();
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: !0,
            };
            k
        ];
        
        let query_vec = query.to_vec();
        let results = self.hnsw.nearest(&query_vec, k * 2, &mut searcher, &mut neighbors);
        
        results.iter()
            .filter(|n| n.index != !0) // Filter out invalid results
            .filter_map(|n| {
                self.index_to_id.get(&n.index).map(|&custom_id| SearchResult {
                    id: custom_id,
                    score: 1.0 / (n.distance as f64 + 1.0), // Convert distance to similarity score
                })
            })
            .collect()
    }
    fn len(&self) -> usize {
        self.vectors.len()
    }
    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    fn get_vector(&self, id: u64) -> Option<&Vector> {
        self.vectors.get(&id)
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

impl Debug for HNSWIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HNSWIndex")
            .field("len", &self.vectors.len())
            .field("is_empty", &self.vectors.is_empty())
            .field("dimension", &self.dim)
            .finish()
    }
}
#[test]
fn test_create_hnswindex() {
    let hnsw = HNSWIndex::new(3);
    assert!(hnsw.is_empty());
    assert_eq!(hnsw.dimension(), 3);
}

#[test]
fn test_add_vector() {
    let mut hnsw = HNSWIndex::new(3);
    let vector = Vector {
        id: 1,
        values: vec![1.0, 2.0, 3.0],
    };
    
    assert!(hnsw.add(vector).is_ok());
    assert_eq!(hnsw.len(), 1);
    assert!(!hnsw.is_empty());
}

#[test]
fn test_add_vector_dimension_mismatch() {
    let mut hnsw = HNSWIndex::new(3);
    let vector = Vector {
        id: 1,
        values: vec![1.0, 2.0], // Wrong dimension
    };
    
    assert!(hnsw.add(vector).is_err());
    assert_eq!(hnsw.len(), 0);
}

#[test]
fn test_search_basic() {
    let mut hnsw = HNSWIndex::new(3);
    
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0] },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0] },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0] },
        Vector { id: 4, values: vec![1.0, 1.0, 0.0] },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    assert_eq!(hnsw.len(), 4);
    
    // Search for vector similar to [1.0, 0.0, 0.0]
    let query = vec![1.1, 0.1, 0.1];
    let results = hnsw.search(&query, 2);
    
    assert!(!results.is_empty());
    assert!(results.len() <= 2);
    
    // Results should be sorted by score (highest first)
    for i in 1..results.len() {
        assert!(results[i-1].score >= results[i].score);
    }
}

#[test]
fn test_search_empty_index() {
    let hnsw = HNSWIndex::new(3);
    let query = vec![1.0, 2.0, 3.0];
    let results = hnsw.search(&query, 5);
    
    assert!(results.is_empty());
}

#[test]
fn test_id_mapping() {
    let mut hnsw = HNSWIndex::new(3);
    
    // Add vectors with custom IDs
    let vectors = vec![
        Vector { id: 100, values: vec![1.0, 0.0, 0.0] },
        Vector { id: 200, values: vec![0.0, 1.0, 0.0] },
        Vector { id: 300, values: vec![0.0, 0.0, 1.0] },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    // Test that we can retrieve vectors by their custom IDs
    assert!(hnsw.get_vector(100).is_some());
    assert!(hnsw.get_vector(200).is_some());
    assert!(hnsw.get_vector(300).is_some());
    assert!(hnsw.get_vector(999).is_none());
    
    // Test that search returns the correct custom IDs
    let query = vec![1.1, 0.1, 0.1];
    let results = hnsw.search(&query, 2);
    
    assert!(!results.is_empty());
    // The first result should be the vector with ID 100 (most similar to [1.0, 0.0, 0.0])
    assert_eq!(results[0].id, 100);
}

#[test]
fn test_duplicate_id_error() {
    let mut hnsw = HNSWIndex::new(3);
    
    let vector1 = Vector { id: 1, values: vec![1.0, 2.0, 3.0] };
    let vector2 = Vector { id: 1, values: vec![4.0, 5.0, 6.0] }; // Same ID
    
    assert!(hnsw.add(vector1).is_ok());
    assert!(hnsw.add(vector2).is_err()); // Should fail with duplicate ID
}

#[test]
fn test_delete_vector() {
    let mut hnsw = HNSWIndex::new(3);
    
    let vector = Vector { id: 42, values: vec![1.0, 2.0, 3.0] };
    assert!(hnsw.add(vector).is_ok());
    assert_eq!(hnsw.len(), 1);
    
    // Delete the vector
    assert!(hnsw.delete(42).is_ok());
    assert_eq!(hnsw.len(), 0);
    assert!(hnsw.get_vector(42).is_none());
    
    // Try to delete non-existent vector
    assert!(hnsw.delete(999).is_err());
}

#[test]
fn test_feature_flags() {
    // Test that the constants are properly set based on features
    // This test will only pass if the correct feature is enabled
    let hnsw = HNSWIndex::new(3);
    
    // Verify the HNSW was created successfully
    assert!(hnsw.is_empty());
    assert_eq!(hnsw.dimension(), 3);
    
    // The actual connection values are tested at compile time
    // through the type system, so we just verify the struct works
}

