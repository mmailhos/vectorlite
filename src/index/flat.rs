use crate::{Vector, VectorIndex, SearchResult, cosine_similarity};

#[derive(Debug)]
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
            return Err(format!("Vector dimension mismatch"));
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
