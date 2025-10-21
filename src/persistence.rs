//! # Persistence Module
//!
//! This module provides functionality for saving and loading collections to/from disk.
//! It supports collection-level persistence with full override strategy.
//!
//! ## File Format
//!
//! Collections are saved as JSON files with the following structure:
//! ```json
//! {
//!   "version": "1.0.0",
//!   "metadata": {
//!     "name": "collection_name",
//!     "created_at": "2025-01-21T10:00:00Z",
//!     "vector_count": 1000,
//!     "dimension": 768,
//!     "index_type": "HNSW"
//!   },
//!   "index": {
//!     // Serialized VectorIndexWrapper
//!   },
//!   "next_id": 1001
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::{VectorIndexWrapper, Collection, VectorIndex};

/// Error types for persistence operations
#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),
    
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },
    
    #[error("Collection error: {0}")]
    Collection(String),
}

/// File header containing version and format information
#[derive(Debug, Serialize, Deserialize)]
pub struct FileHeader {
    pub version: String,
    pub format: String,
    pub created_at: DateTime<Utc>,
}

/// Collection metadata for persistence
#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub vector_count: usize,
    pub dimension: usize,
    pub index_type: String,
}

/// Complete collection data structure for persistence
#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionData {
    pub header: FileHeader,
    pub metadata: CollectionMetadata,
    pub index: VectorIndexWrapper,
    pub next_id: u64,
}

impl Default for FileHeader {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            format: "vectorlite-collection".to_string(),
            created_at: Utc::now(),
        }
    }
}

impl CollectionData {
    /// Create a new CollectionData from a Collection
    pub fn from_collection(collection: &Collection) -> Result<Self, PersistenceError> {
        let index = collection.index_read()
            .map_err(|e| PersistenceError::Collection(e))?;
        
        let next_id = collection.next_id();
        
        let index_type = match &*index {
            VectorIndexWrapper::Flat(_) => "Flat",
            VectorIndexWrapper::HNSW(_) => "HNSW",
        };
        
        Ok(CollectionData {
            header: FileHeader::default(),
            metadata: CollectionMetadata {
                name: collection.name().to_string(),
                created_at: Utc::now(),
                vector_count: index.len(),
                dimension: index.dimension(),
                index_type: index_type.to_string(),
            },
            index: (*index).clone(),
            next_id,
        })
    }
    
    /// Convert CollectionData back to a Collection
    pub fn to_collection(self) -> Collection {
        let collection = Collection::new(self.metadata.name, self.index);
        // Set the next_id to the saved value to maintain ID continuity
        collection.set_next_id(self.next_id);
        collection
    }
}

/// Save a collection to a file
pub fn save_collection_to_file(collection: &Collection, path: &Path) -> Result<(), PersistenceError> {
    let collection_data = CollectionData::from_collection(collection)?;
    
    // Create parent directories if they don't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Write to a temporary file first, then rename for atomicity
    let temp_path = path.with_extension("tmp");
    let json_data = serde_json::to_string_pretty(&collection_data)?;
    fs::write(&temp_path, json_data)?;
    
    // Atomic rename
    fs::rename(&temp_path, path)?;
    
    Ok(())
}

/// Load a collection from a file
pub fn load_collection_from_file(path: &Path) -> Result<Collection, PersistenceError> {
    let json_data = fs::read_to_string(path)?;
    let collection_data: CollectionData = serde_json::from_str(&json_data)?;
    
    // Validate version compatibility
    if collection_data.header.version != "1.0.0" {
        return Err(PersistenceError::VersionMismatch {
            expected: "1.0.0".to_string(),
            actual: collection_data.header.version,
        });
    }
    
    // Validate format
    if collection_data.header.format != "vectorlite-collection" {
        return Err(PersistenceError::InvalidFormat(format!(
            "Expected format 'vectorlite-collection', got '{}'",
            collection_data.header.format
        )));
    }
    
    Ok(collection_data.to_collection())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatIndex, HNSWIndex, Vector, SimilarityMetric};
    use tempfile::TempDir;

    fn create_test_collection() -> Collection {
        let vectors = vec![
            Vector { id: 0, values: vec![1.0, 2.0, 3.0] },
            Vector { id: 1, values: vec![4.0, 5.0, 6.0] },
        ];
        let flat_index = FlatIndex::new(3, vectors);
        let index = VectorIndexWrapper::Flat(flat_index);
        
        Collection::new("test_collection".to_string(), index)
    }

    #[test]
    fn test_collection_data_creation() {
        let collection = create_test_collection();
        let collection_data = CollectionData::from_collection(&collection).unwrap();
        
        assert_eq!(collection_data.metadata.name, "test_collection");
        assert_eq!(collection_data.metadata.vector_count, 2);
        assert_eq!(collection_data.metadata.dimension, 3);
        assert_eq!(collection_data.metadata.index_type, "Flat");
        assert_eq!(collection_data.next_id, 2);
    }

    #[test]
    fn test_collection_data_conversion() {
        let original_collection = create_test_collection();
        let collection_data = CollectionData::from_collection(&original_collection).unwrap();
        let restored_collection = collection_data.to_collection();
        
        assert_eq!(restored_collection.name(), original_collection.name());
        assert_eq!(
            restored_collection.next_id(),
            original_collection.next_id()
        );
        
        // Test that the index works
        let index = restored_collection.index_read().unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.dimension(), 3);
        
        // Test search functionality
        let results = index.search(&[1.1, 2.1, 3.1], 1, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_save_and_load_collection() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_collection.vlc");
        
        let original_collection = create_test_collection();
        
        // Save collection
        save_collection_to_file(&original_collection, &file_path).unwrap();
        assert!(file_path.exists());
        
        // Load collection
        let loaded_collection = load_collection_from_file(&file_path).unwrap();
        
        // Verify basic properties
        assert_eq!(loaded_collection.name(), original_collection.name());
        assert_eq!(
            loaded_collection.next_id(),
            original_collection.next_id()
        );
        
        // Verify index functionality
        let index = loaded_collection.index_read().unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.dimension(), 3);
        
        // Test search
        let results = index.search(&[1.1, 2.1, 3.1], 1, SimilarityMetric::Cosine);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_save_and_load_hnsw_collection() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_hnsw_collection.vlc");
        
        // Create HNSW collection
        let hnsw_index = HNSWIndex::new(3);
        let index = VectorIndexWrapper::HNSW(Box::new(hnsw_index));
        
        let collection = Collection::new("test_hnsw_collection".to_string(), index);
        
        // Add some vectors
        let vector1 = Vector { id: 0, values: vec![1.0, 2.0, 3.0] };
        let vector2 = Vector { id: 1, values: vec![4.0, 5.0, 6.0] };
        
        collection.add_vector(vector1).unwrap();
        collection.add_vector(vector2).unwrap();
        
        // Save and load
        save_collection_to_file(&collection, &file_path).unwrap();
        let loaded_collection = load_collection_from_file(&file_path).unwrap();
        
        // Verify
        assert_eq!(loaded_collection.name(), "test_hnsw_collection");
        let index = loaded_collection.index_read().unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.dimension(), 3);
    }

    #[test]
    fn test_invalid_file_format() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("invalid.vlc");
        
        // Write invalid JSON
        fs::write(&file_path, "invalid json").unwrap();
        
        let result = load_collection_from_file(&file_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::Serialization(_)));
    }

    #[test]
    fn test_version_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("version_mismatch.vlc");
        
        // Create invalid version data
        let invalid_data = CollectionData {
            header: FileHeader {
                version: "2.0.0".to_string(),
                format: "vectorlite-collection".to_string(),
                created_at: Utc::now(),
            },
            metadata: CollectionMetadata {
                name: "test".to_string(),
                created_at: Utc::now(),
                vector_count: 0,
                dimension: 3,
                index_type: "Flat".to_string(),
            },
            index: VectorIndexWrapper::Flat(FlatIndex::new(3, vec![])),
            next_id: 0,
        };
        
        let json_data = serde_json::to_string_pretty(&invalid_data).unwrap();
        fs::write(&file_path, json_data).unwrap();
        
        let result = load_collection_from_file(&file_path);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::VersionMismatch { .. }));
    }
}
