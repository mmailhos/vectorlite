use std::usize;
use std::fmt::{Formatter, Debug};


use rand::rngs::StdRng;
use space::{Metric, Neighbor};
use hnsw::{Hnsw, Searcher};
use crate::{Vector, VectorIndex, SearchResult};
struct Euclidean;

// Maximum number of connections each node can have in all layers except the 0 layer
const MAXIMUM_NUMBER_CONNECTIONS: usize = 16;
// Maximum number of connections for the bottom layer
const MAXIMUM_NUMBER_CONNECTIONS_0: usize = 32;


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
    dim: usize
}

impl HNSWIndex {
    pub fn new(dim: usize) -> Self {
        let hnsw: Hnsw<Euclidean, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0> = Hnsw::new(Euclidean);
        let searcher = Searcher::new();
        Self { hnsw, searcher, dim}
    }
}

impl VectorIndex for HNSWIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dim {
            return Err(format!("Vector dimension mismatch"));
        }
        self.hnsw.insert(vector.values, &mut self.searcher);
        Ok(())
    }
    fn delete(&mut self, id: u64) -> Result<(), String> {
        unimplemented!()
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
            .map(|n| SearchResult {
                id: n.index as u64,
                score: 1.0 / (n.distance as f64 + 1.0), // Convert distance to similarity score
            })
            .collect()
    }
    fn len(&self) -> usize {
        self.hnsw.len()
    }
    fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }
    fn get_vector(&self, id: u64) -> Option<&Vector> {
        unimplemented!();
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

impl Debug for HNSWIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HNSWIndex")
            .field("len", &self.hnsw.len())
            .field("is_empty", &self.hnsw.is_empty())
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
    
    // Add some test vectors
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

