
use serde::{Deserialize, Serialize};

use crate::{Vector, VectorIndex, SearchResult, cosine_similarity};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct HashIndex {
    pub dim: usize,
    pub data: HashMap<u64, Vector>
}

impl HashIndex {
    pub fn new(dim: usize, data: HashMap<u64, Vector>) -> Self {
        Self { 
            dim, 
            data 
        }
    }
}

impl VectorIndex for HashIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dim {
            return Err("Vector dimension mismatch".to_string());
        }
        if self.data.contains_key(&vector.id) {
            return Err(format!("Vector ID {} already exists", vector.id));
        }
        self.data.insert(vector.id, vector);
        Ok(())
    }

    fn delete(&mut self, id: u64) -> Result<(), String> {
        if !self.data.contains_key(&id) {
            return Err(format!("Vector ID {} does not exist", id));
        }
        self.data.remove(&id);
        Ok(())
    }

    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = Vec::new();
        for (id, vector) in &self.data {
            results.push(SearchResult { id: *id, score: cosine_similarity(&vector.values, query) });
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn get_vector(&self, id: u64) -> Option<&Vector> {
        self.data.get(&id)
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}