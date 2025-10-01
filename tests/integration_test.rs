use vectorlite::*;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

const TEST_DATASET_PATH: &str = "tests/datasets/github-issues.jsonl";

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

pub fn load_test_dataset(path: &str, dimension: usize) -> Result<FlatIndex, io::Error> {
    let mut vector_store = FlatIndex::new(dimension, Vec::new());
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    for (id_counter, line) in reader.lines().enumerate() {
        let issue = serde_json::from_str::<Issue>(&line?)?;
        let vector = Vector { 
            id: id_counter as u64,
            values: issue.embeddings,
        };
        vector_store.add(vector).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    
    Ok(vector_store)
}


#[test]
fn test_load_real_dataset() {
    // Test that we can load the real dataset without errors
    let vector_store = load_test_dataset(TEST_DATASET_PATH, DEFAULT_VECTOR_DIMENSION)
        .expect("Failed to load real dataset");
    
    // Verify we loaded some data
    assert!(!vector_store.is_empty(), "Dataset should not be empty");
    assert!(vector_store.len() == 2175, "Should have 2175 embeddings");
    
    // Verify all embeddings have the correct dimension
    for vector in &vector_store.data {
        assert_eq!(
            vector.values.len(), 
            DEFAULT_VECTOR_DIMENSION,
            "All vectors should have dimension {}",
            DEFAULT_VECTOR_DIMENSION
        );
    }
    
    println!("Successfully loaded {} embeddings from real dataset", vector_store.len());
}

#[test]
fn test_dataset_file_exists() {
    // Test that the dataset file actually exists
    assert!(
        Path::new(TEST_DATASET_PATH).exists(),
        "Dataset file should exist at {}", 
        TEST_DATASET_PATH
    );
}