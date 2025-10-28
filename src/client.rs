//! # Client Module
//!
//! This module provides the main client interface for VectorLite, including collection management,
//! vector operations, and search functionality.
//!
//! The `VectorLiteClient` is the primary entry point for interacting with the vector database.
//! It manages collections of vectors and provides thread-safe operations for adding, searching,
//! and deleting vectors.
//!
//! # Examples
//!
//! ```rust
//! use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a client with an embedding function
//! let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
//!
//! // Create a collection
//! client.create_collection("documents", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
//!
//! // Add text (auto-generates embedding)
//! let id = client.add_text_to_collection("documents", "Hello world", None)?;
//!
//! // Search for similar text
//! let results = client.search_text_in_collection(
//!     "documents", 
//!     "hello", 
//!     5, 
//!     None  // Auto-detects from HNSW index metric
//! )?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::path::Path;
use crate::{VectorIndexWrapper, VectorIndex, Vector, SearchResult, SimilarityMetric, EmbeddingFunction, PersistenceError, save_collection_to_file, load_collection_from_file};
use crate::errors::{VectorLiteError, VectorLiteResult};

/// Main client for interacting with VectorLite
///
/// The `VectorLiteClient` provides a high-level interface for managing vector collections,
/// performing searches, and handling embeddings. It's designed to be thread-safe and
/// efficient for AI agent workloads.
///
/// # Thread Safety
///
/// The client uses `Arc<RwLock<>>` for collections and atomic counters for ID generation,
/// making it safe to use across multiple threads.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
/// client.create_collection("docs", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
/// # Ok(())
/// # }
/// ```
pub struct VectorLiteClient {
    collections: HashMap<String, CollectionRef>,
    embedding_function: Arc<dyn EmbeddingFunction>,
}

/// Configuration settings for VectorLite
///
/// Currently unused but reserved for future configuration options.
pub struct Settings {}

impl VectorLiteClient {
    pub fn new(embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn create_collection(&mut self, name: &str, index_type: IndexType, metric: Option<SimilarityMetric>) -> VectorLiteResult<()> {
        if self.collections.contains_key(name) {
            return Err(VectorLiteError::CollectionAlreadyExists { name: name.to_string() });
        }

        let dimension = self.embedding_function.dimension();
        let index = match index_type {
            IndexType::Flat => {
                VectorIndexWrapper::Flat(crate::FlatIndex::new(dimension, Vec::new()))
            },
            IndexType::HNSW => {
                // HNSW requires a metric to build the graph structure
                // No default is provided to force explicit specification
                let used_metric = metric.ok_or(VectorLiteError::MetricRequired)?;
                VectorIndexWrapper::HNSW(Box::new(crate::HNSWIndex::new(dimension, used_metric)))
            },
        };

        let collection = Collection {
            name: name.to_string(),
            index: Arc::new(RwLock::new(index)),
            next_id: Arc::new(AtomicU64::new(0)),
        };

        self.collections.insert(name.to_string(), Arc::new(collection));
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<&CollectionRef> {
        self.collections.get(name)
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    pub fn delete_collection(&mut self, name: &str) -> VectorLiteResult<()> {
        if self.collections.remove(name).is_some() {
            Ok(())
        } else {
            Err(VectorLiteError::CollectionNotFound { name: name.to_string() })
        }
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    pub fn add_text_to_collection(&self, collection_name: &str, text: &str, metadata: Option<serde_json::Value>) -> VectorLiteResult<u64> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| VectorLiteError::CollectionNotFound { name: collection_name.to_string() })?;
        
        collection.add_text_with_metadata(text, metadata, self.embedding_function.as_ref())
    }


    pub fn search_text_in_collection(&self, collection_name: &str, query_text: &str, k: usize, similarity_metric: Option<SimilarityMetric>) -> VectorLiteResult<Vec<SearchResult>> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| VectorLiteError::CollectionNotFound { name: collection_name.to_string() })?;
        
        let metric = match similarity_metric {
            Some(m) => m,
            None => {
                let index_guard = collection.index.read().map_err(|_| {
                    VectorLiteError::LockError("Failed to acquire read lock for metric detection".to_string())
                })?;
                
                match index_guard.metric() {
                    Some(m) => m,
                    None => SimilarityMetric::Cosine, 
                }
            }
        };
        
        collection.search_text(query_text, k, metric, self.embedding_function.as_ref())
    }


    pub fn delete_from_collection(&self, collection_name: &str, id: u64) -> VectorLiteResult<()> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| VectorLiteError::CollectionNotFound { name: collection_name.to_string() })?;
        
        collection.delete(id)
    }

    pub fn get_vector_from_collection(&self, collection_name: &str, id: u64) -> VectorLiteResult<Option<Vector>> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| VectorLiteError::CollectionNotFound { name: collection_name.to_string() })?;
        
        collection.get_vector(id)
    }

    pub fn get_collection_info(&self, collection_name: &str) -> VectorLiteResult<CollectionInfo> {
        let collection = self.collections.get(collection_name)
            .ok_or_else(|| VectorLiteError::CollectionNotFound { name: collection_name.to_string() })?;
        
        collection.get_info()
    }

    /// Add a collection directly (used for loading from files)
    pub fn add_collection(&mut self, collection: Collection) -> VectorLiteResult<()> {
        let name = collection.name().to_string();
        if self.collections.contains_key(&name) {
            return Err(VectorLiteError::CollectionAlreadyExists { name });
        }
        self.collections.insert(name, Arc::new(collection));
        Ok(())
    }

}

/// Index types available for vector collections
///
/// Different index types offer different trade-offs between search speed, memory usage,
/// and accuracy. Choose based on your dataset size and performance requirements.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
/// 
/// // For small datasets with exact search requirements
/// // Flat index - metric is optional
/// client.create_collection("small_data", IndexType::Flat, None)?;
/// 
/// // For large datasets with approximate search tolerance
/// client.create_collection("large_data", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    /// Flat index - exact search with O(n) complexity
    /// 
    /// Best for:
    /// - Small datasets (< 10K vectors)
    /// - Exact search requirements
    /// - Memory-constrained environments
    Flat,
    /// HNSW index - approximate search with O(log n) complexity
    /// 
    /// Best for:
    /// - Large datasets (> 10K vectors)
    /// - Approximate search tolerance
    /// - High-performance requirements
    HNSW,
}

/// Collection structure containing the vector index and metadata
///
/// This struct wraps the actual vector index with thread-safe primitives
/// and provides atomic ID generation for new vectors.
///
/// # Thread Safety
///
/// Uses `Arc<RwLock<>>` for the index to allow concurrent reads and exclusive writes,
/// and `Arc<AtomicU64>` for thread-safe ID generation.
pub struct Collection {
    name: String,
    index: Arc<RwLock<VectorIndexWrapper>>,
    next_id: Arc<AtomicU64>,
}

/// Type alias for a thread-safe collection reference
type CollectionRef = Arc<Collection>;

/// Information about a collection
///
/// Contains metadata about a vector collection including its name, size,
/// vector dimension, and whether it's empty.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
/// client.create_collection("docs", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
/// 
/// let info = client.get_collection_info("docs")?;
/// println!("Collection '{}' has {} vectors of dimension {}", 
///          info.name, info.count, info.dimension);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionInfo {
    /// Name of the collection
    pub name: String,
    /// Number of vectors in the collection
    pub count: usize,
    /// Whether the collection is empty
    pub is_empty: bool,
    /// Dimension of vectors in this collection
    pub dimension: usize,
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.name)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl Collection {
    /// Create a new collection with the given name and index
    pub fn new(name: String, index: VectorIndexWrapper) -> Self {
        // Calculate next_id from the maximum ID in the index + 1
        let next_id = match &index {
            VectorIndexWrapper::Flat(flat_index) => {
                flat_index.max_id()
                    .map(|max_id| max_id + 1)
                    .unwrap_or(0)
            }
            VectorIndexWrapper::HNSW(hnsw_index) => {
                hnsw_index.max_id()
                    .map(|max_id| max_id + 1)
                    .unwrap_or(0)
            }
        };

        Self {
            name,
            index: Arc::new(RwLock::new(index)),
            next_id: Arc::new(AtomicU64::new(next_id)),
        }
    }

    pub fn add_text(&self, text: &str, embedding_function: &dyn EmbeddingFunction) -> VectorLiteResult<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        // Generate embedding outside the lock
        let embedding = embedding_function.generate_embedding(text)?;
        
        let vector = Vector { 
            id, 
            values: embedding, 
            text: text.to_string(),
            metadata: None 
        };
        let vector_dimension = vector.values.len();
        let vector_id = vector.id;
        
        // Acquire write lock only for the index operation
        let mut index = self.index.write().map_err(|_| VectorLiteError::LockError("Failed to acquire write lock for add_text".to_string()))?;
        index.add(vector).map_err(|e| {
            if e.contains("dimension") {
                VectorLiteError::DimensionMismatch { 
                    expected: index.dimension(), 
                    actual: vector_dimension 
                }
            } else if e.contains("already exists") {
                VectorLiteError::DuplicateVectorId { id: vector_id }
            } else {
                VectorLiteError::InternalError(e)
            }
        })?;
        Ok(id)
    }

    pub fn add_text_with_metadata(&self, text: &str, metadata: Option<serde_json::Value>, embedding_function: &dyn EmbeddingFunction) -> VectorLiteResult<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        // Generate embedding outside the lock
        let embedding = embedding_function.generate_embedding(text)?;
        
        let vector = Vector { 
            id, 
            values: embedding, 
            text: text.to_string(),
            metadata 
        };
        let vector_dimension = vector.values.len();
        let vector_id = vector.id;
        
        // Acquire write lock only for the index operation
        let mut index = self.index.write().map_err(|_| VectorLiteError::LockError("Failed to acquire write lock for add_text_with_metadata".to_string()))?;
        index.add(vector).map_err(|e| {
            if e.contains("dimension") {
                VectorLiteError::DimensionMismatch { 
                    expected: index.dimension(), 
                    actual: vector_dimension 
                }
            } else if e.contains("already exists") {
                VectorLiteError::DuplicateVectorId { id: vector_id }
            } else {
                VectorLiteError::InternalError(e)
            }
        })?;
        Ok(id)
    }


    pub fn delete(&self, id: u64) -> VectorLiteResult<()> {
        let mut index = self.index.write().map_err(|_| VectorLiteError::LockError("Failed to acquire write lock for delete".to_string()))?;
        index.delete(id).map_err(|e| {
            if e.contains("does not exist") {
                VectorLiteError::VectorNotFound { id }
            } else {
                VectorLiteError::InternalError(e)
            }
        })
    }

    pub fn search_text(&self, query_text: &str, k: usize, similarity_metric: SimilarityMetric, embedding_function: &dyn EmbeddingFunction) -> VectorLiteResult<Vec<SearchResult>> {
        // Generate embedding outside the lock
        let query_embedding = embedding_function.generate_embedding(query_text)?;
        
        // Acquire read lock for search
        let index = self.index.read().map_err(|_| VectorLiteError::LockError("Failed to acquire read lock for search_text".to_string()))?;

        index.search(&query_embedding, k, similarity_metric) 
    }


    pub fn get_vector(&self, id: u64) -> VectorLiteResult<Option<Vector>> {
        let index = self.index.read().map_err(|_| VectorLiteError::LockError("Failed to acquire read lock for get_vector".to_string()))?;
        Ok(index.get_vector(id))
    }

    pub fn get_info(&self) -> VectorLiteResult<CollectionInfo> {
        let index = self.index.read().map_err(|_| VectorLiteError::LockError("Failed to acquire read lock for get_info".to_string()))?;
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

    /// Get the current next ID value
    pub fn next_id(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Get a read lock on the index
    pub fn index_read(&self) -> Result<std::sync::RwLockReadGuard<'_, VectorIndexWrapper>, String> {
        self.index.read().map_err(|_| "Failed to acquire read lock".to_string())
    }

    /// Save the collection to a file
    ///
    /// This method saves the entire collection state to disk, including all vectors,
    /// the index structure, and the next ID counter. The file format is JSON-based
    /// and includes metadata for version compatibility.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the collection should be saved
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a `PersistenceError` if the operation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
    /// client.create_collection("docs", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
    /// client.add_text_to_collection("docs", "Hello world", None)?;
    ///
    /// let collection = client.get_collection("docs").unwrap();
    /// collection.save_to_file(Path::new("./docs.vlc"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_to_file(&self, path: &Path) -> Result<(), PersistenceError> {
        save_collection_to_file(self, path)
    }

    /// Load a collection from a file
    ///
    /// This method creates a new collection by loading its state from disk.
    /// The loaded collection will completely replace any existing collection
    /// with the same name (override strategy).
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from which to load the collection
    ///
    /// # Returns
    ///
    /// Returns the loaded `Collection` on success, or a `PersistenceError` if the operation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vectorlite::Collection;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let collection = Collection::load_from_file(Path::new("./docs.vlc"))?;
    /// println!("Loaded collection '{}' with {} vectors", 
    ///          collection.name(), collection.get_info()?.count);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_from_file(path: &Path) -> Result<Self, PersistenceError> {
        load_collection_from_file(path)
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
            // Return a simple mock embedding with the correct dimension
            Ok(vec![1.0; self.dimension])
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
        let result = client.create_collection("test_collection", IndexType::Flat, None);
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
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        
        // Try to create duplicate
        let result = client.create_collection("test_collection", IndexType::Flat, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VectorLiteError::CollectionAlreadyExists { .. }));
    }

    #[test]
    fn test_get_collection() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        
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
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
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
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        
        // Add text
        let result = client.add_text_to_collection("test_collection", "Hello world", None);
        assert!(result.is_ok());
        let id = result.unwrap();
        assert_eq!(id, 0); // First ID is 0
        
        // Add another text
        let result = client.add_text_to_collection("test_collection", "Another text", None);
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
        let result = client.add_text_to_collection("non_existent", "Hello world", None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VectorLiteError::CollectionNotFound { .. }));
    }

    #[test]
    fn test_collection_operations() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        
        // Test initial state
        let info = client.get_collection_info("test_collection").unwrap();
        assert!(info.is_empty);
        assert_eq!(info.count, 0);
        assert_eq!(info.name, "test_collection");
        
        // Add text
        let id = client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
        assert_eq!(id, 0);
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert!(!info.is_empty);
        assert_eq!(info.count, 1);
        
        // Add another text
        let id = client.add_text_to_collection("test_collection", "Another text", None).unwrap();
        assert_eq!(id, 1);
        
        let info = client.get_collection_info("test_collection").unwrap();
        assert_eq!(info.count, 2);
        
        // Test search
        let results = client.search_text_in_collection("test_collection", "Hello", 1, Some(SimilarityMetric::Cosine)).unwrap();
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
        client.create_collection("hnsw_collection", IndexType::HNSW, Some(SimilarityMetric::Euclidean)).unwrap();
        
        // Add some text
        let id1 = client.add_text_to_collection("hnsw_collection", "First document", None).unwrap();
        let id2 = client.add_text_to_collection("hnsw_collection", "Second document", None).unwrap();
        
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        
        let info = client.get_collection_info("hnsw_collection").unwrap();
        assert_eq!(info.count, 2);
        
        // Test search with Euclidean (must match the index metric)
        let results = client.search_text_in_collection("hnsw_collection", "First", 1, Some(SimilarityMetric::Euclidean)).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_hnsw_requires_metric() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));

        // Creating HNSW without metric should fail
        let result = client.create_collection("hnsw_collection", IndexType::HNSW, None);
        assert!(result.is_err());
        match result {
            Err(VectorLiteError::MetricRequired) => {
                // Expected error
            }
            _ => panic!("Expected MetricRequired error when creating HNSW without metric"),
        }
    }

    #[test]
    fn test_collection_save_and_load() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create collection and add some data
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
        client.add_text_to_collection("test_collection", "Another text", None).unwrap();
        
        let collection = client.get_collection("test_collection").unwrap();
        
        // Save to temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_collection.vlc");
        
        collection.save_to_file(&file_path).unwrap();
        assert!(file_path.exists());
        
        // Load the collection
        let loaded_collection = Collection::load_from_file(&file_path).unwrap();
        
        // Verify basic properties
        assert_eq!(loaded_collection.name(), "test_collection");
        
        // Verify the index works
        let info = loaded_collection.get_info().unwrap();
        assert_eq!(info.count, 2);
        assert_eq!(info.dimension, 3);
        assert!(!info.is_empty);
        
        // Test search functionality using text search
        let test_embedding_fn = MockEmbeddingFunction::new(3);
        let results = loaded_collection.search_text("Hello", 2, SimilarityMetric::Cosine, &test_embedding_fn).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_collection_save_and_load_hnsw() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Create HNSW collection and add some data
        client.create_collection("test_hnsw_collection", IndexType::HNSW, Some(SimilarityMetric::Euclidean)).unwrap();
        client.add_text_to_collection("test_hnsw_collection", "First document", None).unwrap();
        client.add_text_to_collection("test_hnsw_collection", "Second document", None).unwrap();
        
        let collection = client.get_collection("test_hnsw_collection").unwrap();
        
        // Verify the original collection works
        let info = collection.get_info().unwrap();
        assert_eq!(info.count, 2);
        assert_eq!(info.dimension, 3);
        
        // Create a separate embedding function for testing
        let test_embedding_fn = MockEmbeddingFunction::new(3);
        
        // Test search on original collection using text search 
        let results = collection.search_text("First", 1, SimilarityMetric::Euclidean, &test_embedding_fn).unwrap();
        assert_eq!(results.len(), 1);
        
        // Save to temporary file
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_hnsw_collection.vlc");
        
        collection.save_to_file(&file_path).unwrap();
        assert!(file_path.exists());
        
        // Load the collection
        let loaded_collection = Collection::load_from_file(&file_path).unwrap();
        
        // Verify basic properties
        assert_eq!(loaded_collection.name(), "test_hnsw_collection");
        
        // Verify the index works
        let info = loaded_collection.get_info().unwrap();
        assert_eq!(info.count, 2);
        assert_eq!(info.dimension, 3);
        assert!(!info.is_empty);
        
        // Test search functionality using text search
        let results = loaded_collection.search_text("First", 1, SimilarityMetric::Euclidean, &test_embedding_fn).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_collection_save_nonexistent_directory() {
        let embedding_fn = MockEmbeddingFunction::new(3);
        let mut client = VectorLiteClient::new(Box::new(embedding_fn));
        
        // Flat indexes don't need a metric parameter
        client.create_collection("test_collection", IndexType::Flat, None).unwrap();
        client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
        
        let collection = client.get_collection("test_collection").unwrap();
        
        // Try to save to a non-existent directory (should create it)
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("nonexistent").join("test_collection.vlc");
        
        let result = collection.save_to_file(&file_path);
        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[test]
    fn test_collection_load_nonexistent_file() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("nonexistent.vlc");

        let result = Collection::load_from_file(&file_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::FileNotFound(_)));
    }

    #[test]
    fn test_collection_load_invalid_json() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("invalid.vlc");
        
        // Write invalid JSON
        std::fs::write(&file_path, "invalid json content").unwrap();
        
        let result = Collection::load_from_file(&file_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::Serialization(_)));
    }
}