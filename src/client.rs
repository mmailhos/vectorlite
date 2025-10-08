use std::collections::HashMap;
use std::sync::Arc;
use crate::{VectorIndexWrapper, VectorIndex, Vector, SearchResult, SimilarityMetric, EmbeddingFunction};

pub struct VectorLiteClient {
    collections: HashMap<String, Collection>,
    settings: Settings,
    embedding_function: Arc<dyn EmbeddingFunction>,
}

pub struct Settings {}

impl VectorLiteClient {
    pub fn new(embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            settings: Settings {},
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn new_with_settings(settings: Settings, embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            settings,
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn create_collection(&mut self, name: &str, index_type: IndexType) -> Result<&mut Collection, String> {
        if self.collections.contains_key(name) {
            return Err(format!("Collection '{}' already exists", name));
        }

        let dimension = self.embedding_function.dimension();
        let index = match index_type {
            IndexType::Flat => VectorIndexWrapper::Flat(crate::FlatIndex::new(dimension, Vec::new())),
            IndexType::HNSW => VectorIndexWrapper::HNSW(crate::HNSWIndex::new(dimension)),
        };

        let collection = Collection {
            name: name.to_string(),
            index,
            embedding_function: Arc::clone(&self.embedding_function),
            next_id: 1,
        };

        self.collections.insert(name.to_string(), collection);
        Ok(self.collections.get_mut(name).unwrap())
    }

    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn get_collection_mut(&mut self, name: &str) -> Option<&mut Collection> {
        self.collections.get_mut(name)
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

    pub fn add_text_to_collection(&mut self, collection_name: &str, text: &str) -> Result<u64, String> {
        if let Some(collection) = self.collections.get_mut(collection_name) {
            collection.add_text(text)
        } else {
            Err(format!("Collection '{}' not found", collection_name))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    Flat,
    HNSW,
}

pub struct Collection {
    name: String,
    index: VectorIndexWrapper,
    embedding_function: Arc<dyn EmbeddingFunction>,
    next_id: u64,
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("index", &self.index)
            .field("next_id", &self.next_id)
            .field("embedding_function", &"<EmbeddingFunction>")
            .finish()
    }
}

impl Collection {
    pub fn add_text(&mut self, text: &str) -> Result<u64, String> {
        let id = self.next_id;
        self.next_id += 1;
        
        let embedding = self.embedding_function.generate_embedding(text)
            .map_err(|e| e.to_string())?;
        let vector = Vector { id, values: embedding };
        self.index.add(vector)?;
        Ok(id)
    }

    pub fn add_vector(&mut self, vector: Vector) -> Result<(), String> {
        self.index.add(vector)
    }

    pub fn delete(&mut self, id: u64) -> Result<(), String> {
        self.index.delete(id)
    }

    pub fn search_text(&self, query_text: &str, k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, String> {
        let query_embedding = self.embedding_function.generate_embedding(query_text)
            .map_err(|e| e.to_string())?;
        Ok(self.index.search(&query_embedding, k, similarity_metric))
    }

    pub fn search_vector(&self, query_vector: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Vec<SearchResult> {
        self.index.search(query_vector, k, similarity_metric)
    }

    pub fn get_vector(&self, id: u64) -> Option<&Vector> {
        self.index.get_vector(id)
    }

    pub fn count(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
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
        assert_eq!(id, 1);
        
        // Add another text
        let result = client.add_text_to_collection("test_collection", "Another text");
        assert!(result.is_ok());
        let id = result.unwrap();
        assert_eq!(id, 2);
        
        // Check collection count
        let collection = client.get_collection("test_collection").unwrap();
        assert_eq!(collection.count(), 2);
    }

    #[test]
    fn test_add_text_to_nonexistent_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
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
        
        // Get mutable reference
        let collection = client.get_collection_mut("test_collection").unwrap();
        
        // Test initial state
        assert!(collection.is_empty());
        assert_eq!(collection.count(), 0);
        assert_eq!(collection.name(), "test_collection");
        
        // Add text
        let id = collection.add_text("Hello world").unwrap();
        assert_eq!(id, 1);
        assert!(!collection.is_empty());
        assert_eq!(collection.count(), 1);
        
        // Add another text
        let id = collection.add_text("Another text").unwrap();
        assert_eq!(id, 2);
        assert_eq!(collection.count(), 2);
        
        // Test search
        let results = collection.search_text("Hello", 1, SimilarityMetric::Cosine).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
        
        // Test get vector
        let vector = collection.get_vector(1);
        assert!(vector.is_some());
        assert_eq!(vector.unwrap().id, 1);
        
        // Test delete
        collection.delete(1).unwrap();
        assert_eq!(collection.count(), 1);
        
        // Verify vector is gone
        let vector = collection.get_vector(1);
        assert!(vector.is_none());
    }

    #[test]
    fn test_collection_with_hnsw_index() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create HNSW collection
        client.create_collection("hnsw_collection", IndexType::HNSW).unwrap();
        
        let collection = client.get_collection_mut("hnsw_collection").unwrap();
        
        // Add some text
        let id1 = collection.add_text("First document").unwrap();
        let id2 = collection.add_text("Second document").unwrap();
        
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(collection.count(), 2);
        
        // Test search
        let results = collection.search_text("First", 1, SimilarityMetric::Cosine).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_add_vector_directly() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        let collection = client.get_collection_mut("test_collection").unwrap();
        
        // Add vector directly
        let vector = Vector {
            id: 42,
            values: vec![1.0, 2.0, 3.0],
        };
        
        collection.add_vector(vector).unwrap();
        assert_eq!(collection.count(), 1);
        
        // Verify vector exists
        let retrieved = collection.get_vector(42);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 42);
    }

    #[test]
    fn test_search_vector_directly() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        client.create_collection("test_collection", IndexType::Flat).unwrap();
        let collection = client.get_collection_mut("test_collection").unwrap();
        
        // Add some vectors
        collection.add_text("Hello world").unwrap();
        collection.add_text("Another text").unwrap();
        
        // Search with vector directly
        let query_vector = vec![1.0, 2.0, 3.0];
        let results = collection.search_vector(&query_vector, 2, SimilarityMetric::Cosine);
        
        assert_eq!(results.len(), 2);
        // Results should be sorted by score (highest first)
        for i in 1..results.len() {
            assert!(results[i-1].score >= results[i].score);
        }
    }
}