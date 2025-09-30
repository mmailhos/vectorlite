use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub const DEFAULT_EMBEDDING_DIMENSION: usize = 768;

#[derive(serde::Deserialize)]
pub struct Issue {
    pub html_url: String,
    pub title: String,
    pub comments: String,
    pub body: String,
    pub comment_length: u32,
    pub text: String,
    pub embeddings: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: u64,
    pub embedding: Vec<f64>,
}

#[derive(Debug)]
pub struct VectorStore {
    pub dim: usize,
    pub data: Vec<Embedding>,
}

impl VectorStore {
    pub fn new(dim: usize, data: Vec<Embedding>) -> Self {
        Self { dim, data }
    }

    pub fn insert(&mut self, embedding: Embedding) {
        assert_eq!(embedding.embedding.len(), self.dim, "Embedding dimension mismatch");
        self.data.push(embedding);
    }

    pub fn search(&self, query: &[f64], k: usize) -> Vec<(u64, f64)> {
        let mut similarities = self.data
            .iter()
            .map(|e| (e.id, cosine_similarity(&e.embedding, query)))
            .collect::<Vec<_>>();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_embedding(&self, id: u64) -> Option<&Embedding> {
        self.data.iter().find(|e| e.id == id)
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

pub fn load_test_dataset(path: &str, dimension: usize) -> Result<VectorStore, io::Error> {
    let mut vector_store = VectorStore::new(dimension, Vec::new());
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    for (id_counter, line) in reader.lines().enumerate() {
        let issue = serde_json::from_str::<Issue>(&line?)?;
        vector_store.insert(Embedding { 
            id: id_counter as u64,
            embedding: issue.embeddings,
        });
    }
    
    Ok(vector_store)
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
        let embeddings = vec![
            Embedding { id: 0, embedding: vec![1.0, 2.0, 3.0] },
            Embedding { id: 1, embedding: vec![4.0, 5.0, 6.0] },
        ];
        let store = VectorStore::new(3, embeddings);
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_vector_store_search() {
        let embeddings = vec![
            Embedding { id: 0, embedding: vec![1.0, 0.0, 0.0] },
            Embedding { id: 1, embedding: vec![0.0, 1.0, 0.0] },
            Embedding { id: 2, embedding: vec![0.0, 0.0, 1.0] },
        ];
        let store = VectorStore::new(3, embeddings);
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Should match the first vector
        assert!((results[0].1 - 1.0).abs() < 1e-10); // Perfect similarity
    }
}