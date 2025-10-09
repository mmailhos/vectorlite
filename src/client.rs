use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::{VectorIndexWrapper, VectorIndex, Vector, SearchResult, SimilarityMetric, EmbeddingFunction};

pub struct VectorLiteClient {
    collections: HashMap<String, Collection>,
    embedding_function: Arc<dyn EmbeddingFunction>,
}

pub struct Settings {}

impl VectorLiteClient {
    pub fn new(embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn create_collection(&mut self, name: &str, index_type: IndexType) -> Result<(), String> {
        if self.collections.contains_key(name) {
            return Err(format!("Collection '{}' already exists", name));
        }

        let dimension = self.embedding_function.dimension();
        let index = match index_type {
            IndexType::Flat => VectorIndexWrapper::Flat(crate::FlatIndex::new(dimension, Vec::new())),
            IndexType::HNSW => VectorIndexWrapper::HNSW(crate::HNSWIndex::new(dimension)),
        };

        let collection = InnerCollection {
            name: name.to_string(),
            index: Arc::new(RwLock::new(index)),
            next_id: Arc::new(AtomicU64::new(0)),
        };

        self.collections.insert(name.to_string(), Arc::new(collection));
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<(), String> {
        if self.collections.remove(name).is_some() {
            Ok(())
        } else {
            Err(format!("Collection '{}' not found", name))
        }
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    pub fn add_text_to_collection(&self, collection_name: &str, text: &str) -> Result<u64, String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.add_text(text, self.embedding_function.as_ref())
    }

    pub fn add_vector_to_collection(&self, collection_name: &str, vector: Vector) -> Result<(), String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.add_vector(vector)
    }

    pub fn search_text_in_collection(&self, collection_name: &str, query_text: &str, k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.search_text(query_text, k, similarity_metric, self.embedding_function.as_ref())
    }

    pub fn search_vector_in_collection(&self, collection_name: &str, query_vector: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.search_vector(query_vector, k, similarity_metric)
    }

    pub fn delete_from_collection(&self, collection_name: &str, id: u64) -> Result<(), String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.delete(id)
    }

    pub fn get_vector_from_collection(&self, collection_name: &str, id: u64) -> Result<Option<Vector>, String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.get_vector(id)
    }

    pub fn get_collection_info(&self, collection_name: &str) -> Result<CollectionInfo, String> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        
        collection.get_info()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    Flat,
    HNSW,
}

type Collection = Arc<InnerCollection>;

pub struct InnerCollection {
    name: String,
    index: Arc<RwLock<VectorIndexWrapper>>,
    next_id: Arc<AtomicU64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub count: usize,
    pub is_empty: bool,
    pub dimension: usize,
}

impl std::fmt::Debug for InnerCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl InnerCollection {
    pub fn add_text(&self, text: &str, embedding_function: &dyn EmbeddingFunction) -> Result<u64, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        // Generate embedding outside the lock
        let embedding = embedding_function.generate_embedding(text)
            .map_err(|e| e.to_string())?;
        
        let vector = Vector { id, values: embedding };
        
        // Acquire write lock only for the index operation
        let mut index = self.index.write().map_err(|_| "Failed to acquire write lock")?;
        index.add(vector)?;
        Ok(id)
    }

    pub fn add_vector(&self, vector: Vector) -> Result<(), String> {
        let mut index = self.index.write().map_err(|_| "Failed to acquire write lock")?;
        index.add(vector)
    }

    pub fn delete(&self, id: u64) -> Result<(), String> {
        let mut index = self.index.write().map_err(|_| "Failed to acquire write lock")?;
        index.delete(id)
    }

    pub fn search_text(&self, query_text: &str, k: usize, similarity_metric: SimilarityMetric, embedding_function: &dyn EmbeddingFunction) -> Result<Vec<SearchResult>, String> {
        // Generate embedding outside the lock
        let query_embedding = embedding_function.generate_embedding(query_text)
            .map_err(|e| e.to_string())?;
        
        // Acquire read lock for search
        let index = self.index.read().map_err(|_| "Failed to acquire read lock")?;
        Ok(index.search(&query_embedding, k, similarity_metric))
    }

    pub fn search_vector(&self, query_vector: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, String> {
        let index = self.index.read().map_err(|_| "Failed to acquire read lock")?;
        Ok(index.search(query_vector, k, similarity_metric))
    }

    pub fn get_vector(&self, id: u64) -> Result<Option<Vector>, String> {
        let index = self.index.read().map_err(|_| "Failed to acquire read lock")?;
        Ok(index.get_vector(id).cloned())
    }

    pub fn get_info(&self) -> Result<CollectionInfo, String> {
        let index = self.index.read().map_err(|_| "Failed to acquire read lock")?;
        Ok(CollectionInfo {
            name: self.name.clone(),
            count: index.len(),
            is_empty: index.is_empty(),
            dimension: index.dimension(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock embedding function for testing
    struct MockEmbeddingFunction {
        dimension: usize,
    }

    impl MockEmbeddingFunction {
        fn new(dimension: usize) -> Self {
            Self { dimension }
        }
    }

    impl EmbeddingFunction for MockEmbeddingFunction {
        fn generate_embedding(&self, _text: &str) -> crate::embeddings::Result<Vec<f64>> {
            // Return a simple mock embedding
            Ok(vec![1.0, 2.0, 3.0])
        }

        fn dimension(&self) -> usize {
            self.dimension
        }
    }

    #[test]
    fn test_client_creation() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let client = VectorLiteClient::new(Box::new(embedding_fn));
        
        assert!(client.collections.is_empty());
        assert!(client.list_collections().is_empty());
    }

    #[test]
    fn test_create_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        let result = client.create_collection("test_collection", IndexType::Flat);
        assert!(result.is_ok());
        
        // Check collection exists
        assert!(client.has_collection("test_collection"));
        assert_eq!(client.list_collections(), vec!["test_collection"]);
    }

    #[test]
    fn test_create_duplicate_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create first collection
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Try to create duplicate
        let result = client.create_collection("test_collection", IndexType::Flat);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_get_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Get collection
        let collection = client.get_collection("test_collection");
        assert!(collection.is_some());
        assert_eq!(collection.unwrap().name(), "test_collection");
        
        // Get non-existent collection
        let collection = client.get_collection("non_existent");
        assert!(collection.is_none());
    }

    #[test]
    fn test_delete_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        assert!(client.has_collection("test_collection"));
        
        // Delete collection
        let result = client.delete_collection("test_collection");
        assert!(result.is_ok());
        assert!(!client.has_collection("test_collection"));
        
        // Try to delete non-existent collection
        let result = client.delete_collection("non_existent");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_text_to_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Add text
        let result = client.add_text_to_collection("test_collection", "Hello world");
        assert!(result.is_ok());
        let id = result.unwrap();
        assert_eq!(id, 0); // First ID is 0
        
        // Add another text
        let result = client.add_text_to_collection("test_collection", "Another text");
        assert!(result.is_ok());
        let id = result.unwrap();
        assert_eq!(id, 1);
        
        // Check collection count
        let info = client.get_collection_info("test_collection").unwrap();
        assert_eq!(info.count, 2);
    }

    #[test]
    fn test_add_text_to_nonexistent_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Try to add to non-existent collection
        let result = client.add_text_to_collection("non_existent", "Hello world");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_collection_operations() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Test initial state
        let info = client.get_collection_info("test_collection").unwrap();
        assert!(info.is_empty);
        assert_eq!(info.count, 0);
        assert_eq!(info.name, "test_collection");
        
        // Add text
        let id = client.add_text_to_collection("test_collection", "Hello world").unwrap();
        assert_eq!(id, 0);
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert!(!info.is_empty);
        assert_eq!(info.count, 1);
        
        // Add another text
        let id = client.add_text_to_collection("test_collection", "Another text").unwrap();
        assert_eq!(id, 1);
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert_eq!(info.count, 2);
        
        // Test search
        let results = client.search_text_in_collection("test_collection", "Hello", 1, SimilarityMetric::Cosine).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        
        // Test get vector
        let vector = client.get_vector_from_collection("test_collection", 0).unwrap();
        assert!(vector.is_some());
        assert_eq!(vector.unwrap().id, 0);
        
        // Test delete
        client.delete_from_collection("test_collection", 0).unwrap();
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert_eq!(info.count, 1);
        
        // Verify vector is gone
        let vector = client.get_vector_from_collection("test_collection", 0).unwrap();
        assert!(vector.is_none());
    }

    #[test]
    fn test_collection_with_hnsw_index() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create HNSW collection
        client.create_collection("hnsw_collection", IndexType::HNSW).unwrap();
        
        // Add some text
        let id1 = client.add_text_to_collection("hnsw_collection", "First document").unwrap();
        let id2 = client.add_text_to_collection("hnsw_collection", "Second document").unwrap();
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        
        let info = client.get_collection_info("hnsw_collection").unwrap();
        assert_eq!(info.count, 2);
        
        // Test search
        let results = client.search_text_in_collection("hnsw_collection", "First", 1, SimilarityMetric::Cosine).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_add_vector_directly() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Add vector directly
        let vector = Vector {
            id: 42,
            values: vec![1.0, 2.0, 3.0],
        };
        
        client.add_vector_to_collection("test_collection", vector).unwrap();
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert_eq!(info.count, 1);
        
        // Verify vector exists
        let retrieved = client.get_vector_from_collection("test_collection", 42).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 42);
    }

    #[test]
    fn test_search_vector_directly() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        
        // Add some vectors
        client.add_text_to_collection("test_collection", "Hello world").unwrap();
        client.add_text_to_collection("test_collection", "Another text").unwrap();
        
        // Search with vector directly
        let query_vector = vec![1.0, 2.0, 3.0];
        let results = client.search_vector_in_collection("test_collection", &query_vector, 2, SimilarityMetric::Cosine).unwrap();
        
        assert_eq!(results.len(), 2);
        // Results should be sorted by score (highest first)
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score);
        }
    }
}