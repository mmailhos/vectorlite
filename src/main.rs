use std::fs::File;
use std::io::{ BufReader, BufRead, Error};

#[derive(serde::Deserialize)]
struct Issue {
    html_url: String,
    title: String,
    comments: String,
    body: String,
    comment_length: u32,
    text: String,
    embeddings: Vec<f64>,
}

struct Embedding {
    id: u64,
    embedding: Vec<f64>,
}

struct VectorStore {
    dim: usize,
    data: Vec<Embedding>,
}

impl VectorStore {
    fn new(dim: usize, data: Vec<Embedding>) -> Self {
        Self { dim, data }
    }

    fn insert(&mut self, embedding: Embedding) {
        assert_eq!(embedding.embedding.len(), self.dim, "Embedding dimension mismatch");
        self.data.push(embedding);
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
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



    fn search(&self, query: &[f64], k: usize) -> Vec<(u64, f64)> {
        let mut similarities: Vec<(u64, f64)> = self.data
            .iter()
            .map(|e| (e.id, Self::cosine_similarity(&e.embedding, query)))
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }
}


fn load_test_dataset(path: String, dimension: usize) -> Result<VectorStore, Error> {
    let mut vector_store = VectorStore::new(dimension, Vec::new());
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut id_counter: u64 = 0;
    for line in reader.lines() {
        let issue = serde_json::from_str::<Issue>(&line?)?;
        vector_store.insert(Embedding { 
            id: id_counter,
            embedding: issue.embeddings,
        });
        id_counter += 1;
    }
    return Ok(vector_store);
}
fn main() {
    let vector_store = load_test_dataset("dataset/github-embeddings-doy/issues-datasets-embedded.jsonl".to_string(), 768).unwrap();
    println!("{}", vector_store.data.len());
}