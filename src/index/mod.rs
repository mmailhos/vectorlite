//! # Index Module
//!
//! This module contains different vector indexing implementations, each optimized
//! for different use cases and performance characteristics.
//!
//! ## Available Index Types
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
//! Both implementations implement the `VectorIndex` trait, allowing for
//! seamless switching between different indexing strategies.

pub mod flat;
pub mod hnsw;