//! # VectorLite
//!
//! A high-performance, in-memory vector database optimized for AI agent workloads with HTTP API and thread-safe concurrency.
//!
//! ## Overview
//!
//! VectorLite is designed for **single-instance, low-latency vector operations** in AI agent environments. It prioritizes **sub-millisecond search performance** over distributed scalability, making it ideal for:
//!
//! - **AI Agent Sessions**: Session-scoped vector storage with fast retrieval
//! - **Real-time Search**: Sub-millisecond response requirements  
//! - **Prototype Development**: Rapid iteration without infrastructure complexity
//! - **Single-tenant Applications**: No multi-tenancy isolation requirements
//!
//! ## Key Features
//!
//! - **In-memory storage** for zero-latency access patterns
//! - **Native Rust ML models** using Candle framework with pluggable architecture
//! - **Thread-safe concurrency** with RwLock per collection and atomic ID generation
//! - **HNSW indexing** for approximate nearest neighbor search with configurable accuracy
//! - **HTTP API** for easy integration with AI agents and other services
//!
//! ## Quick Start
//!
//! ```rust
//! use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create client with embedding function
//! let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
//!
//! // Create collection
//! client.create_collection("documents", IndexType::HNSW)?;
//!
//! // Add text (auto-generates embedding and ID)
//! let id = client.add_text_to_collection("documents", "Hello world")?;
//!
//! // Search
//! let results = client.search_text_in_collection(
//!     "documents", 
//!     "hello", 
//!     5, 
//!     SimilarityMetric::Cosine
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Index Types
//!
//! ### FlatIndex
//! - **Complexity**: O(n) search, O(1) insert
//! - **Memory**: Linear with dataset size
//! - **Use Case**: Small datasets (< 10K vectors) or exact search requirements
//!
//! ### HNSWIndex
//! - **Complexity**: O(log n) search, O(log n) insert
//! - **Memory**: ~2-3x vector size due to graph structure
//! - **Use Case**: Large datasets with approximate search tolerance
//!
//! ## Similarity Metrics
//!
//! - **Cosine**: Default for normalized embeddings, scale-invariant
//! - **Euclidean**: Geometric distance, sensitive to vector magnitude
//! - **Manhattan**: L1 norm, robust to outliers
//! - **Dot Product**: Raw similarity, requires consistent vector scaling
//!
//! ## HTTP Server
//!
//! ```rust,no_run
//! use vectorlite::{VectorLiteClient, EmbeddingGenerator, start_server};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
//!     start_server(client, "127.0.0.1", 3001).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Configuration Profiles
//!
//! ```bash
//! # Balanced (default)
//! cargo build
//!
//! # Memory-constrained environments
//! cargo build --features memory-optimized
//!
//! # High-precision search
//! cargo build --features high-accuracy
//! ```

pub mod index;
pub mod embeddings;
pub mod client;
pub mod server;
pub mod persistence;
pub mod errors;

pub use index::flat::FlatIndex;
pub use index::hnsw::HNSWIndex;
pub use embeddings::{EmbeddingGenerator, EmbeddingFunction};
pub use client::{VectorLiteClient, Collection, Settings, IndexType};
pub use server::{create_app, start_server};
pub use persistence::{PersistenceError, save_collection_to_file, load_collection_from_file};

use serde::{Serialize, Deserialize};

/// Default vector dimension for embedding models
pub const DEFAULT_VECTOR_DIMENSION: usize = 768;

/// Represents a vector with an ID and floating-point values
///
/// # Examples
///
/// ```rust
/// use vectorlite::Vector;
///
/// let vector = Vector {
///     id: 1,
///     values: vec![0.1, 0.2, 0.3, 0.4],
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    /// Unique identifier for the vector
    pub id: u64,
    /// The vector values (embedding coordinates)
    pub values: Vec<f64>,
}

/// Search result containing a vector ID and similarity score
///
/// Results are typically sorted by score in descending order (highest similarity first).
///
/// # Examples
///
/// ```rust
/// use vectorlite::SearchResult;
///
/// let result = SearchResult {
///     id: 42,
///     score: 0.95,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The ID of the matching vector
    pub id: u64,
    /// Similarity score (higher is more similar)
    pub score: f64,
}

/// Trait for vector indexing implementations
///
/// This trait defines the common interface for different vector indexing strategies,
/// allowing for pluggable implementations like FlatIndex and HNSWIndex.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{VectorIndex, Vector, SimilarityMetric, FlatIndex};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut index = FlatIndex::new(3, Vec::new());
/// let vector = Vector { id: 1, values: vec![1.0, 2.0, 3.0] };
/// 
/// index.add(vector)?;
/// let results = index.search(&[1.1, 2.1, 3.1], 5, SimilarityMetric::Cosine);
/// # Ok(())
/// # }
/// ```
pub trait VectorIndex {
    /// Add a vector to the index
    fn add(&mut self, vector: Vector) -> Result<(), String>;
    
    /// Remove a vector from the index by ID
    fn delete(&mut self, id: u64) -> Result<(), String>;
    
    /// Search for the k most similar vectors
    fn search(&self, query: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Vec<SearchResult>;
    
    /// Get the number of vectors in the index
    fn len(&self) -> usize;
    
    /// Check if the index is empty
    fn is_empty(&self) -> bool;
    
    /// Get a vector by its ID
    fn get_vector(&self, id: u64) -> Option<&Vector>;
    
    /// Get the dimension of vectors in this index
    fn dimension(&self) -> usize;
}

/// Wrapper enum for different vector index implementations
///
/// This allows for dynamic dispatch between different indexing strategies
/// while maintaining a unified interface through the VectorIndex trait.
///
/// # Examples
///
/// ```rust
/// use vectorlite::{VectorIndexWrapper, FlatIndex, HNSWIndex, Vector, SimilarityMetric, VectorIndex};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a flat index wrapper
/// let mut wrapper = VectorIndexWrapper::Flat(FlatIndex::new(3, Vec::new()));
/// 
/// // Add a vector
/// let vector = Vector { id: 1, values: vec![1.0, 2.0, 3.0] };
/// wrapper.add(vector)?;
/// 
/// // Search using the wrapper
/// let results = wrapper.search(&[1.1, 2.1, 3.1], 5, SimilarityMetric::Cosine);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorIndexWrapper {
    /// Flat index for exact search (O(n) complexity)
    Flat(FlatIndex),
    /// HNSW index for approximate search (O(log n) complexity)
    HNSW(Box<HNSWIndex>),
}

impl VectorIndex for VectorIndexWrapper {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.add(vector),
            VectorIndexWrapper::HNSW(index) => index.add(vector),  
        }
    }

    fn delete(&mut self, id: u64) -> Result<(), String> {
        match self {
            VectorIndexWrapper::Flat(index) => index.delete(id),
            VectorIndexWrapper::HNSW(index) => index.delete(id),
        }
    }

    fn search(&self, query: &[f64], k: usize, s: SimilarityMetric) -> Vec<SearchResult> {
        match self {
            VectorIndexWrapper::Flat(index) => index.search(query, k, s),
            VectorIndexWrapper::HNSW(index) => index.search(query, k, s),
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.len(),
            VectorIndexWrapper::HNSW(index) => index.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            VectorIndexWrapper::Flat(index) => index.is_empty(),
            VectorIndexWrapper::HNSW(index) => index.is_empty(),
        }
    }

    fn get_vector(&self, id: u64) -> Option<&Vector> {
        match self {
            VectorIndexWrapper::Flat(index) => index.get_vector(id),
            VectorIndexWrapper::HNSW(index) => index.get_vector(id),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            VectorIndexWrapper::Flat(index) => index.dimension(),
            VectorIndexWrapper::HNSW(index) => index.dimension(),
        }
    }
}

/// Similarity metrics for vector comparison
///
/// Different metrics are suitable for different use cases and vector characteristics.
///
/// # Examples
///
/// ```rust
/// use vectorlite::SimilarityMetric;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.1, 2.1, 3.1];
/// 
/// let cosine_score = SimilarityMetric::Cosine.calculate(&a, &b);
/// let euclidean_score = SimilarityMetric::Euclidean.calculate(&a, &b);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity - scale-invariant, good for normalized embeddings
    /// Range: [-1, 1], where 1 is identical
    #[default]
    Cosine,
    /// Euclidean similarity - geometric distance converted to similarity
    /// Range: [0, 1], where 1 is identical
    Euclidean,
    /// Manhattan similarity - L1 norm distance converted to similarity
    /// Range: [0, 1], where 1 is identical, robust to outliers
    Manhattan,
    /// Dot product - raw similarity without normalization
    /// Range: unbounded, requires consistent vector scaling
    DotProduct,
}

impl SimilarityMetric {
    pub fn calculate(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        
        match self {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::Euclidean => euclidean_similarity(a, b),
            SimilarityMetric::Manhattan => manhattan_similarity(a, b),
            SimilarityMetric::DotProduct => dot_product(a, b),
        }
    }
}


/// Calculate cosine similarity between two vectors
///
/// Cosine similarity measures the cosine of the angle between two vectors,
/// making it scale-invariant and suitable for normalized embeddings.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Similarity score in range [-1, 1] where:
/// - 1.0 = identical vectors
/// - 0.0 = orthogonal vectors  
/// - -1.0 = opposite vectors
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use vectorlite::cosine_similarity;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0, 2.0, 3.0];
/// let similarity = cosine_similarity(&a, &b);
/// assert!((similarity - 1.0).abs() < 1e-10); // Identical vectors
/// ```
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

/// Calculate Euclidean similarity between two vectors
///
/// Euclidean similarity converts the Euclidean distance to a similarity score.
/// It's sensitive to vector magnitude and suitable for vectors with consistent scaling.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Similarity score in range [0, 1] where:
/// - 1.0 = identical vectors
/// - 0.0 = very distant vectors
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use vectorlite::euclidean_similarity;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let similarity = euclidean_similarity(&a, &b);
/// // Distance is 5.0, so similarity is 1/(1+5) = 1/6 ≈ 0.167
/// ```
pub fn euclidean_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let sum_sq = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>();
    
    let distance = sum_sq.sqrt();
    
    // Convert distance to similarity: 1 / (1 + distance)
    // This ensures similarity is in range [0, 1] with 1 being identical
    1.0 / (1.0 + distance)
}

/// Calculate Manhattan similarity between two vectors
///
/// Manhattan similarity converts the L1 norm (Manhattan distance) to a similarity score.
/// It's robust to outliers and suitable for high-dimensional sparse vectors.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Similarity score in range [0, 1] where:
/// - 1.0 = identical vectors
/// - 0.0 = very distant vectors
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use vectorlite::manhattan_similarity;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let similarity = manhattan_similarity(&a, &b);
/// // Distance is 7.0, so similarity is 1/(1+7) = 1/8 = 0.125
/// ```
pub fn manhattan_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    let distance = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>();
    
    // Convert distance to similarity: 1 / (1 + distance)
    // This ensures similarity is in range [0, 1] with 1 being identical
    1.0 / (1.0 + distance)
}

/// Calculate dot product between two vectors
///
/// Dot product is the raw similarity without normalization.
/// It requires consistent vector scaling and can produce unbounded results.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Dot product value (unbounded range):
/// - Positive values indicate similar direction
/// - Zero indicates orthogonal vectors
/// - Negative values indicate opposite direction
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use vectorlite::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0, 2.0, 3.0];
/// let product = dot_product(&a, &b);
/// assert!((product - 14.0).abs() < 1e-10); // 1*1 + 2*2 + 3*3 = 14
/// ```
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>()
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
    fn test_euclidean_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((euclidean_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_similarity_different_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let expected = 1.0 / (1.0 + 5.0); // 1 / (1 + sqrt(3^2 + 4^2))
        assert!((euclidean_similarity(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((manhattan_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_similarity_different_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let expected = 1.0 / (1.0 + 7.0); // 1 / (1 + |0-3| + |0-4|)
        assert!((manhattan_similarity(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let expected = 1.0 + 4.0 + 9.0; // 14.0
        assert!((dot_product(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((dot_product(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let expected = -1.0 - 4.0 - 9.0; // -14.0
        assert!((dot_product(&a, &b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_metric_enum() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        
        // Test that all metrics work
        assert!((SimilarityMetric::Cosine.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::Euclidean.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::Manhattan.calculate(&a, &b) - 1.0).abs() < 1e-10);
        assert!((SimilarityMetric::DotProduct.calculate(&a, &b) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_metric_default() {
        assert_eq!(SimilarityMetric::default(), SimilarityMetric::Cosine);
    }

    #[test]
    fn test_vector_store_creation() {
        let vectors = vec![
            Vector { id: 0, values: vec![1.0, 2.0, 3.0] },
            Vector { id: 1, values: vec![4.0, 5.0, 6.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_vector_store_search() {
        let vectors = vec![
            Vector { id: 0, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 1, values: vec![0.0, 1.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 0.0, 1.0] },
        ];
        let store = FlatIndex::new(3, vectors);
        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 2, SimilarityMetric::Cosine);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_index_wrapper_serialization() {
        use serde_json;
        
        // Test FlatIndex wrapper
        let vectors = vec![
            Vector { id: 1, values: vec![1.0, 0.0, 0.0] },
            Vector { id: 2, values: vec![0.0, 1.0, 0.0] },
        ];
        let flat_index = FlatIndex::new(3, vectors);
        let wrapper = VectorIndexWrapper::Flat(flat_index);
        
        // Serialize wrapper
        let serialized = serde_json::to_string(&wrapper).expect("Serialization should work");
        
        // Deserialize wrapper
        let deserialized: VectorIndexWrapper = serde_json::from_str(&serialized).expect("Deserialization should work");
        
        // Verify the deserialized wrapper works
        assert_eq!(deserialized.len(), 2);
        assert_eq!(deserialized.dimension(), 3);
        assert!(!deserialized.is_empty());
        
        // Test search through the wrapper
        let query = vec![1.1, 0.1, 0.1];
        let results = deserialized.search(&query, 1, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 1);
    }

}