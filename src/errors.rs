//! # Error Types Module
//!
//! This module defines comprehensive error types for the VectorLite API,
//! providing structured error handling instead of string-based error matching.

use thiserror::Error;
use axum::http::StatusCode;

/// Main error type for VectorLite operations
#[derive(Error, Debug)]
pub enum VectorLiteError {
    /// Collection not found
    #[error("Collection '{name}' not found")]
    CollectionNotFound { name: String },
    
    /// Vector dimension mismatch
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    /// Vector ID already exists
    #[error("Vector ID {id} already exists")]
    DuplicateVectorId { id: u64 },
    
    /// Vector ID does not exist
    #[error("Vector ID {id} does not exist")]
    VectorNotFound { id: u64 },
    
    /// Collection already exists
    #[error("Collection '{name}' already exists")]
    CollectionAlreadyExists { name: String },
    
    /// Invalid index type
    #[error("Invalid index type: {index_type}. Must be 'flat' or 'hnsw'")]
    InvalidIndexType { index_type: String },
    
    /// Invalid similarity metric
    #[error("Invalid similarity metric: {metric}. Must be 'cosine', 'euclidean', 'manhattan', or 'dotproduct'")]
    InvalidSimilarityMetric { metric: String },
    
    /// Metric mismatch between search request and index configuration
    #[error("Metric mismatch: search requested {requested:?} but index was built for {index:?}")]
    MetricMismatch { requested: crate::SimilarityMetric, index: crate::SimilarityMetric },

    /// Metric required for HNSW index but not provided
    #[error("HNSW index requires an explicit similarity metric. Metric must be specified when creating HNSW collections.")]
    MetricRequired,

    /// Embedding generation error
    #[error("Embedding generation failed: {0}")]
    EmbeddingError(#[from] crate::embeddings::EmbeddingError),
    
    /// File not found error
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    /// Persistence error
    #[error("Persistence error: {0}")]
    PersistenceError(#[from] crate::persistence::PersistenceError),
    
    /// Lock acquisition error
    #[error("Failed to acquire lock: {0}")]
    LockError(String),
    
    /// Internal server error
    #[error("Internal server error: {0}")]
    InternalError(String),
}

impl VectorLiteError {
    /// Convert the error to an appropriate HTTP status code
    pub fn status_code(&self) -> StatusCode {
        match self {
            VectorLiteError::CollectionNotFound { .. } => StatusCode::NOT_FOUND,
            VectorLiteError::VectorNotFound { .. } => StatusCode::NOT_FOUND,
            VectorLiteError::FileNotFound { .. } => StatusCode::NOT_FOUND,
            VectorLiteError::DimensionMismatch { .. } => StatusCode::BAD_REQUEST,
            VectorLiteError::DuplicateVectorId { .. } => StatusCode::CONFLICT,
            VectorLiteError::CollectionAlreadyExists { .. } => StatusCode::CONFLICT,
            VectorLiteError::InvalidIndexType { .. } => StatusCode::BAD_REQUEST,
            VectorLiteError::InvalidSimilarityMetric { .. } => StatusCode::BAD_REQUEST,
            VectorLiteError::MetricMismatch { .. } => StatusCode::BAD_REQUEST,
            VectorLiteError::MetricRequired => StatusCode::BAD_REQUEST,
            VectorLiteError::EmbeddingError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VectorLiteError::PersistenceError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VectorLiteError::LockError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            VectorLiteError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    
    /// Check if this is a client error (4xx status codes)
    pub fn is_client_error(&self) -> bool {
        matches!(self.status_code(), StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND | StatusCode::CONFLICT)
    }
    
    /// Check if this is a server error (5xx status codes)
    pub fn is_server_error(&self) -> bool {
        matches!(self.status_code(), StatusCode::INTERNAL_SERVER_ERROR)
    }
}

/// Result type for VectorLite operations
pub type VectorLiteResult<T> = Result<T, VectorLiteError>;

/// Helper trait for converting string errors to VectorLiteError
pub trait ToVectorLiteError<T> {
    fn to_vectorlite_error(self, context: &str) -> VectorLiteResult<T>;
}

impl<T> ToVectorLiteError<T> for Result<T, String> {
    fn to_vectorlite_error(self, context: &str) -> VectorLiteResult<T> {
        self.map_err(|e| VectorLiteError::InternalError(format!("{}: {}", context, e)))
    }
}

/// Helper trait for converting lock errors to VectorLiteError
pub trait ToLockError<T> {
    fn to_lock_error(self, operation: &str) -> VectorLiteResult<T>;
}

impl<T> ToLockError<T> for Result<T, std::sync::PoisonError<std::sync::RwLockReadGuard<'_, T>>> {
    fn to_lock_error(self, operation: &str) -> VectorLiteResult<T> {
        self.map_err(|_| VectorLiteError::LockError(format!("Failed to acquire read lock for {}", operation)))
    }
}

impl<T> ToLockError<T> for Result<T, std::sync::PoisonError<std::sync::RwLockWriteGuard<'_, T>>> {
    fn to_lock_error(self, operation: &str) -> VectorLiteResult<T> {
        self.map_err(|_| VectorLiteError::LockError(format!("Failed to acquire write lock for {}", operation)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            VectorLiteError::CollectionNotFound { name: "test".to_string() }.status_code(),
            StatusCode::NOT_FOUND
        );
        
        assert_eq!(
            VectorLiteError::DimensionMismatch { expected: 384, actual: 256 }.status_code(),
            StatusCode::BAD_REQUEST
        );
        
        assert_eq!(
            VectorLiteError::DuplicateVectorId { id: 123 }.status_code(),
            StatusCode::CONFLICT
        );
    }

    #[test]
    fn test_error_classification() {
        assert!(VectorLiteError::CollectionNotFound { name: "test".to_string() }.is_client_error());
        assert!(VectorLiteError::DimensionMismatch { expected: 384, actual: 256 }.is_client_error());
        assert!(VectorLiteError::InternalError("test".to_string()).is_server_error());
    }
}
