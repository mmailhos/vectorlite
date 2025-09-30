use vectorlite::*;
use std::path::Path;

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

#[test]
fn test_load_real_dataset() {
    // Test that we can load the real dataset without errors
    let vector_store = load_test_dataset(TEST_DATASET_PATH, DEFAULT_EMBEDDING_DIMENSION)
        .expect("Failed to load real dataset");
    
    // Verify we loaded some data
    assert!(!vector_store.is_empty(), "Dataset should not be empty");
    assert!(vector_store.len() == 2175, "Should have 2175 embeddings");
    
    // Verify all embeddings have the correct dimension
    for embedding in &vector_store.data {
        assert_eq!(
            embedding.embedding.len(), 
            DEFAULT_EMBEDDING_DIMENSION,
            "All embeddings should have dimension {}",
            DEFAULT_EMBEDDING_DIMENSION
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