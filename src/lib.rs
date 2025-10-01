pub const DEFAULT_VECTOR_DIMENSION: usize = 768;

#[derive(Debug, Clone)]
pub struct Vector {
    pub id: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f64,
}

pub trait VectorIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String>;
    fn search(&self, query: &[f64], k: usize) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_vector(&self, id: u64) -> Option<&Vector>;
}

#[derive(Debug)]
pub struct FlatIndex {
    pub dim: usize,
    pub data: Vec<Vector>
}
impl FlatIndex {
    fn new(dim: usize, data: Vec<Vector>) -> Self {
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
            Vector { id: 0, vector: vec![1.0, 2.0, 3.0] },
            Vector { id: 1, vector: vec![4.0, 5.0, 6.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_vector_store_search() {
        let vectors = vec![
            Vector { id: 0, vector: vec![1.0, 0.0, 0.0] },
            Vector { id: 1, vector: vec![0.0, 1.0, 0.0] },
            Vector { id: 2, vector: vec![0.0, 0.0, 1.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Should match the first vector
        assert!((results[0].1 - 1.0).abs() < 1e-10); // Perfect similarity
    }
}