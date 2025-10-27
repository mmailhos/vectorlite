//! # HNSW Index Implementation
//!
//! This module provides a Hierarchical Navigable Small World (HNSW) index implementation
//! for approximate nearest neighbor search. HNSW offers excellent performance for large
//! datasets with logarithmic search complexity.
//!
//! ## Performance Characteristics
//!
//! - **Search Complexity**: O(log n) - logarithmic search time
//! - **Insert Complexity**: O(log n) - logarithmic insert time
//! - **Memory Usage**: ~2-3x vector size due to graph structure
//! - **Accuracy**: High (configurable via parameters)
//!
//! ## Use Cases
//!
//! - Large datasets (> 10K vectors)
//! - Approximate search tolerance
//! - High-performance requirements
//! - Production systems
//!
//! ## Configuration
//!
//! The index behavior can be tuned via Cargo features:
//! - `memory-optimized`: Reduces memory usage with lower connection counts
//! - `high-accuracy`: Increases accuracy with higher connection counts
//!
//! # Examples
//!
//! ```rust
//! use vectorlite::{HNSWIndex, Vector, SimilarityMetric, VectorIndex};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut index = HNSWIndex::new(384, SimilarityMetric::Euclidean);
//! let vector = Vector { id: 1, values: vec![0.1; 384], text: "test".to_string(), metadata: None };
//! 
//! index.add(vector)?;
//! let results = index.search(&[0.1; 384], 5, SimilarityMetric::Euclidean);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt::{Formatter, Debug};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize, Deserializer};
use space::{Metric, Neighbor};
use hnsw::{Hnsw, Searcher};
use crate::{Vector, VectorIndex, SearchResult, SimilarityMetric};

/// Convert distance to similarity score for the given metric
fn convert_distance_to_similarity(distance: f64, metric: SimilarityMetric) -> f64 {
    match metric {
        SimilarityMetric::Euclidean => {
            // For Euclidean: similarity = 1 / (1 + distance)
            1.0 / (1.0 + distance)
        },
        SimilarityMetric::Cosine => {
            // For Cosine: distance = 1 - similarity, so similarity = 1 - distance
            // But distance is [0, 2000] scaled, so we divide by 1000
            let cos_distance = distance / 1000.0;
            1.0 - cos_distance
        },
        SimilarityMetric::Manhattan => {
            // For Manhattan: similarity = 1 / (1 + distance)
            1.0 / (1.0 + distance)
        },
        SimilarityMetric::DotProduct => {
            // For DotProduct: distance = 1000 - dot_product (clamped)
            // So: dot_product = 1000 - distance
            // We want similarity to range [0, 1] where higher dot product = higher similarity
            // Convert: similarity = (1000 - distance) / 1000, normalized to [0, 1]
            ((1000.0 - distance) / 1000.0).clamp(0.0, 1.0)
        },
    }
}

// VectorMetadata contains the metadata for a vector without the embedding values
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorMetadata {
    text: String,
    metadata: Option<serde_json::Value>,
}
#[derive(Default, Clone)]
struct Euclidean;

#[derive(Default, Clone)]
struct Cosine;

#[derive(Default, Clone)]
struct Manhattan;

#[derive(Default, Clone)]
struct DotProduct;

const MAXIMUM_NUMBER_CONNECTIONS: usize = if cfg!(feature = "memory-optimized") {
    8
} else if cfg!(feature = "high-accuracy") {
    32
} else {
    16
};

const MAXIMUM_NUMBER_CONNECTIONS_0: usize = if cfg!(feature = "memory-optimized") {
    16
} else if cfg!(feature = "high-accuracy") {
    64
} else {
    32 
};



impl Metric<Vec<f64>> for Euclidean {
    type Unit = u64;
    
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> Self::Unit {
        let sum_sq = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>();
        (sum_sq.sqrt() * 1000.0) as u64 
    }
}

impl Metric<Vec<f64>> for Cosine {
    type Unit = u64;
    
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> Self::Unit {
        // Cosine distance = 1 - cosine_similarity
        let (dot, norm_a_sq, norm_b_sq) = a.iter()
            .zip(b.iter())
            .fold((0.0, 0.0, 0.0), |(dot, a_sq, b_sq), (&x, &y)| {
                (dot + x * y, a_sq + x * x, b_sq + y * y)
            });
        
        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1000; // Maximum distance for zero vectors
        }
        
        let cosine_sim = dot / (norm_a * norm_b);
        // Convert to distance: (1 - similarity) * 1000
        let distance = (1.0 - cosine_sim) * 1000.0;
        distance as u64
    }
}

impl Metric<Vec<f64>> for Manhattan {
    type Unit = u64;
    
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> Self::Unit {
        let dist = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum::<f64>();
        (dist * 1000.0) as u64
    }
}

impl Metric<Vec<f64>> for DotProduct {
    type Unit = u64;
    
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> Self::Unit {
        // Dot product as distance (negative because higher dot product = smaller distance)
        let dot = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>();
        // Convert to positive distance: 1000 - dot (clamped)
        (1000.0 - dot.clamp(-1000.0, 1000.0)) as u64
    }
}

/// Enum to hold different HNSW index types for different metrics
#[derive(Clone)]
enum HNSWIndexInternal {
    Euclidean {
        hnsw: Hnsw<Euclidean, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0>,
        searcher: Searcher<u64>,
    },
    Cosine {
        hnsw: Hnsw<Cosine, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0>,
        searcher: Searcher<u64>,
    },
    Manhattan {
        hnsw: Hnsw<Manhattan, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0>,
        searcher: Searcher<u64>,
    },
    DotProduct {
        hnsw: Hnsw<DotProduct, Vec<f64>, StdRng, MAXIMUM_NUMBER_CONNECTIONS, MAXIMUM_NUMBER_CONNECTIONS_0>,
        searcher: Searcher<u64>,
    },
}

#[derive(Clone, Serialize)]
pub struct HNSWIndex {
    #[serde(skip)]
    index_internal: HNSWIndexInternal,

    dim: usize,
    // The similarity metric this index was optimized for
    metric: SimilarityMetric,
    // Mapping from custom ID to internal HNSW index
    id_to_index: HashMap<u64, usize>,
    // Mapping from internal HNSW index to custom ID
    index_to_id: HashMap<usize, u64>,
    // Store only metadata (text + JSON), not the full Vector
    metadata: HashMap<u64, VectorMetadata>,
    // Store vector values separately
    vector_values: HashMap<u64, Vec<f64>>,
}

impl HNSWIndex {
    pub fn new(dim: usize, metric: SimilarityMetric) -> Self {
        if dim == 0 {
            panic!("HNSW index dimension cannot be 0");
        }
        
        // Create HNSW with the specific metric for its graph structure.
        // This ensures the HNSW graph is optimized for the intended similarity metric.
        let index_internal = match metric {
            SimilarityMetric::Euclidean => {
                HNSWIndexInternal::Euclidean {
                    hnsw: Hnsw::new(Euclidean),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::Cosine => {
                HNSWIndexInternal::Cosine {
                    hnsw: Hnsw::new(Cosine),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::Manhattan => {
                HNSWIndexInternal::Manhattan {
                    hnsw: Hnsw::new(Manhattan),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::DotProduct => {
                HNSWIndexInternal::DotProduct {
                    hnsw: Hnsw::new(DotProduct),
                    searcher: Searcher::new(),
                }
            },
        };
        
        Self { 
            index_internal,
            dim,
            metric,
            id_to_index: HashMap::new(),
            index_to_id: HashMap::new(),
            metadata: HashMap::new(),
            vector_values: HashMap::new(),
        }
    }
    
    /// Get the metric this index was built for
    pub fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    /// Get the maximum ID from the stored vectors
    pub fn max_id(&self) -> Option<u64> {
        self.metadata.keys().max().copied()
    }
}

impl<'de> Deserialize<'de> for HNSWIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> 
    where D: Deserializer<'de> 
    {
        // Use an anonymous struct to match the JSON structure
        #[derive(Deserialize)]
        struct Temp {
            dim: usize,
            metric: SimilarityMetric,
            metadata: HashMap<u64, VectorMetadata>,
            vector_values: HashMap<u64, Vec<f64>>,
        }
        
        let data = Temp::deserialize(deserializer)?;
        
        if data.dim == 0 {
            return Err(serde::de::Error::custom("Invalid dimension: cannot be 0"));
        }
        
        // Create the appropriate HNSW index based on the metric
        let mut index_internal = match data.metric {
            SimilarityMetric::Euclidean => {
                HNSWIndexInternal::Euclidean {
                    hnsw: Hnsw::new(Euclidean),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::Cosine => {
                HNSWIndexInternal::Cosine {
                    hnsw: Hnsw::new(Cosine),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::Manhattan => {
                HNSWIndexInternal::Manhattan {
                    hnsw: Hnsw::new(Manhattan),
                    searcher: Searcher::new(),
                }
            },
            SimilarityMetric::DotProduct => {
                HNSWIndexInternal::DotProduct {
                    hnsw: Hnsw::new(DotProduct),
                    searcher: Searcher::new(),
                }
            },
        };
        
        let mut new_id_to_index = HashMap::new();
        let mut new_index_to_id = HashMap::new();
        
        // Insert all vectors into the appropriate HNSW index
        for (id, values) in &data.vector_values {
            if values.len() != data.dim {
                return Err(serde::de::Error::custom(format!(
                    "Vector dimension mismatch: expected {}, got {}", 
                    data.dim, values.len()
                )));
            }
            
            let internal_index = match &mut index_internal {
                HNSWIndexInternal::Euclidean { hnsw, searcher } => {
                    hnsw.insert(values.clone(), searcher)
                },
                HNSWIndexInternal::Cosine { hnsw, searcher } => {
                    hnsw.insert(values.clone(), searcher)
                },
                HNSWIndexInternal::Manhattan { hnsw, searcher } => {
                    hnsw.insert(values.clone(), searcher)
                },
                HNSWIndexInternal::DotProduct { hnsw, searcher } => {
                    hnsw.insert(values.clone(), searcher)
                },
            };
            
            new_id_to_index.insert(*id, internal_index);
            new_index_to_id.insert(internal_index, *id);
        }
        
        Ok(HNSWIndex {
            index_internal,
            dim: data.dim,
            metric: data.metric,
            id_to_index: new_id_to_index,
            index_to_id: new_index_to_id,
            metadata: data.metadata,
            vector_values: data.vector_values,
        })
    }
}

impl VectorIndex for HNSWIndex {
    fn add(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dim {
            return Err(format!("Vector dimension mismatch: expected {}, got {}", self.dim, vector.values.len()));
        }
        
        if self.id_to_index.contains_key(&vector.id) {
            return Err(format!("Vector ID {} already exists", vector.id));
        }
        
        let internal_index = match &mut self.index_internal {
            HNSWIndexInternal::Euclidean { hnsw, searcher } => {
                hnsw.insert(vector.values.clone(), searcher)
            },
            HNSWIndexInternal::Cosine { hnsw, searcher } => {
                hnsw.insert(vector.values.clone(), searcher)
            },
            HNSWIndexInternal::Manhattan { hnsw, searcher } => {
                hnsw.insert(vector.values.clone(), searcher)
            },
            HNSWIndexInternal::DotProduct { hnsw, searcher } => {
                hnsw.insert(vector.values.clone(), searcher)
            },
        };
        
        // Store metadata and values separately
        let vector_metadata = VectorMetadata {
            text: vector.text,
            metadata: vector.metadata,
        };
        
        self.id_to_index.insert(vector.id, internal_index);
        self.index_to_id.insert(internal_index, vector.id);
        self.metadata.insert(vector.id, vector_metadata);
        self.vector_values.insert(vector.id, vector.values);
        
        Ok(())
    }
    fn delete(&mut self, id: u64) -> Result<(), String> {
        if !self.id_to_index.contains_key(&id) {
            return Err(format!("Vector ID {} does not exist", id));
        }
        
        let internal_index = self.id_to_index[&id];
        
        //  Since HNSW doesn't support deletion, we just remove the reference to the node in the mapping
        self.id_to_index.remove(&id);
        self.index_to_id.remove(&internal_index);
        self.metadata.remove(&id);
        self.vector_values.remove(&id);
        
        Ok(())
    }
    fn search(&self, query: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, crate::errors::VectorLiteError> {
        if query.len() != self.dim {
            return Err(crate::errors::VectorLiteError::DimensionMismatch { 
                expected: self.dim, 
                actual: query.len() 
            });
        }
        
        // Reject searches that don't match the metric the index was built for
        // HNSW's graph structure is optimized for a specific distance metric
        if similarity_metric != self.metric {
            return Err(crate::errors::VectorLiteError::MetricMismatch { 
                requested: similarity_metric,
                index: self.metric 
            });
        }
        
        if self.metadata.is_empty() {
            return Ok(Vec::new());
        }
        
        let query_vec = query.to_vec();
        let max_candidates = std::cmp::min(k, self.metadata.len());
        if max_candidates == 0 {
            return Ok(Vec::new());
        }
        
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: !0,
            };
            max_candidates
        ];
        
        // Use the appropriate HNSW index based on the metric
        let results = match &self.index_internal {
            HNSWIndexInternal::Euclidean { hnsw, .. } => {
                let mut searcher: Searcher<u64> = Searcher::new();
                hnsw.nearest(&query_vec, max_candidates, &mut searcher, &mut neighbors)
            },
            HNSWIndexInternal::Cosine { hnsw, .. } => {
                let mut searcher: Searcher<u64> = Searcher::new();
                hnsw.nearest(&query_vec, max_candidates, &mut searcher, &mut neighbors)
            },
            HNSWIndexInternal::Manhattan { hnsw, .. } => {
                let mut searcher: Searcher<u64> = Searcher::new();
                hnsw.nearest(&query_vec, max_candidates, &mut searcher, &mut neighbors)
            },
            HNSWIndexInternal::DotProduct { hnsw, .. } => {
                let mut searcher: Searcher<u64> = Searcher::new();
                hnsw.nearest(&query_vec, max_candidates, &mut searcher, &mut neighbors)
            },
        };
        
        // Convert HNSW distances to similarity scores
        // The HNSW returns distances in its native Unit (u64 scaled by 1000)
        let mut search_results: Vec<SearchResult> = results.iter()
            .filter(|n| n.index != !0) // Filter out invalid results
            .filter_map(|n| {
                self.index_to_id.get(&n.index).and_then(|&custom_id| {
                    self.metadata.get(&custom_id).map(|meta| {
                        // Convert u64 distance back to f64, then to similarity
                        let distance = n.distance as f64 / 1000.0;
                        let score = convert_distance_to_similarity(distance, similarity_metric);
                        
                        SearchResult { 
                            id: custom_id, 
                            score,
                            text: meta.text.clone(),
                            metadata: meta.metadata.clone()
                        }
                    })
                })
            })
            .collect();
        
        // Sort by similarity score and take top k
        search_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        search_results.truncate(k);
        Ok(search_results)
    }
    fn len(&self) -> usize {
        self.metadata.len()
    }
    fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }
    fn get_vector(&self, id: u64) -> Option<Vector> {
        self.metadata.get(&id).and_then(|meta| {
            self.vector_values.get(&id).map(|values| {
                Vector {
                    id,
                    values: values.clone(),
                    text: meta.text.clone(),
                    metadata: meta.metadata.clone(),
                }
            })
        })
    }
    fn dimension(&self) -> usize {
        self.dim
    }
}

impl Debug for HNSWIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HNSWIndex")
            .field("len", &self.metadata.len())
            .field("is_empty", &self.metadata.is_empty())
            .field("dimension", &self.dim)
            .finish()
    }
}
#[test]
fn test_create_hnswindex() {
    let hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    assert!(hnsw.is_empty());
    assert_eq!(hnsw.dimension(), 3);
}

#[test]
fn test_add_vector() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    let vector = Vector {
        id: 1,
        values: vec![1.0, 2.0, 3.0],
        text: "test".to_string(),
        metadata: None,
    };
    
    assert!(hnsw.add(vector).is_ok());
    assert_eq!(hnsw.len(), 1);
    assert!(!hnsw.is_empty());
}

#[test]
fn test_add_vector_dimension_mismatch() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    let vector = Vector {
        id: 1,
        values: vec![1.0, 2.0], // Wrong dimension
        text: "test".to_string(),
        metadata: None,
    };
    
    assert!(hnsw.add(vector).is_err());
    assert_eq!(hnsw.len(), 0);
}

#[test]
fn test_search_basic() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "test".to_string(), metadata: None },
        Vector { id: 4, values: vec![1.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    assert_eq!(hnsw.len(), 4);
    
    // Search for vector similar to [1.0, 0.0, 0.0]
    let query = vec![1.1, 0.1, 0.1];
    let results = hnsw.search(&query, 2, SimilarityMetric::Euclidean).unwrap();
    
    assert!(!results.is_empty());
    assert!(results.len() <= 2);
    
    // Results should be sorted by score (highest first)
    for i in 1..results.len() {
        assert!(results[i-1].score >= results[i].score);
    }
}

    #[test]
    fn test_search_empty_index() {
        let hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
        let query = vec![1.0, 2.0, 3.0];
        let results = hnsw.search(&query, 5, SimilarityMetric::Euclidean).unwrap();
        
        assert!(results.is_empty());
    }

#[test]
fn test_id_mapping() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    // Add vectors with custom IDs
    let vectors = vec![
        Vector { id: 100, values: vec![1.0, 0.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 200, values: vec![0.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 300, values: vec![0.0, 0.0, 1.0], text: "test".to_string(), metadata: None },
        Vector { id: 400, values: vec![1.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    // Test that we can retrieve vectors by their custom IDs
    assert!(hnsw.get_vector(100).is_some());
    assert!(hnsw.get_vector(200).is_some());
    assert!(hnsw.get_vector(300).is_some());
    assert!(hnsw.get_vector(400).is_some());
    assert!(hnsw.get_vector(999).is_none());
    
    // Test that search returns the correct custom IDs
    let query = vec![1.1, 0.1, 0.1];
    let results = hnsw.search(&query, 2, SimilarityMetric::Euclidean).unwrap();
    
    assert!(!results.is_empty());
    // The first result should be the vector with ID 100 (most similar to [1.0, 0.0, 0.0])
    assert_eq!(results[0].id, 100);
}

#[test]
fn test_duplicate_id_error() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    let vector1 = Vector { id: 1, values: vec![1.0, 2.0, 3.0], text: "test".to_string(), metadata: None };
    let vector2 = Vector { id: 1, values: vec![4.0, 5.0, 6.0], text: "test".to_string(), metadata: None }; // Same ID
    
    assert!(hnsw.add(vector1).is_ok());
    assert!(hnsw.add(vector2).is_err()); // Should fail with duplicate ID
}

#[test]
fn test_delete_vector() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    let vector = Vector { id: 42, values: vec![1.0, 2.0, 3.0], text: "test".to_string(), metadata: None };
    assert!(hnsw.add(vector).is_ok());
    assert_eq!(hnsw.len(), 1);
    
    // Delete the vector
    assert!(hnsw.delete(42).is_ok());
    assert_eq!(hnsw.len(), 0);
    assert!(hnsw.get_vector(42).is_none());
    
    // Try to delete non-existent vector
    assert!(hnsw.delete(999).is_err());
}

#[test]
fn test_feature_flags() {
    // Test that the constants are properly set based on features
    // This test will only pass if the correct feature is enabled
    let hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    // Verify the HNSW was created successfully
    assert!(hnsw.is_empty());
    assert_eq!(hnsw.dimension(), 3);
    
    // The actual connection values are tested at compile time
    // through the type system, so we just verify the struct works
}

#[test]
fn test_serialization_deserialization() {
    use serde_json;
    
    // Create an HNSW index with some data
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "test".to_string(), metadata: None },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    let serialized = serde_json::to_string(&hnsw).expect("Serialization should work");
    let mut deserialized: HNSWIndex = serde_json::from_str(&serialized).expect("Deserialization should work");
    
    // Verify the deserialized index has the same properties
    assert_eq!(deserialized.len(), 3);
    assert_eq!(deserialized.dimension(), 3);
    assert!(!deserialized.is_empty());
    
    // Verify we can retrieve vectors by ID
    assert!(deserialized.get_vector(1).is_some());
    assert!(deserialized.get_vector(2).is_some());
    assert!(deserialized.get_vector(3).is_some());
    
    // Test that we can retrieve vectors by ID (this should work)
    let vector1 = deserialized.get_vector(1).unwrap();
    let vector2 = deserialized.get_vector(2).unwrap();
    let vector3 = deserialized.get_vector(3).unwrap();
    
    // Verify the vectors have the correct values
    assert_eq!(vector1.values, vec![1.0, 0.0, 0.0]);
    assert_eq!(vector2.values, vec![0.0, 1.0, 0.0]);
    assert_eq!(vector3.values, vec![0.0, 0.0, 1.0]);
    
    // Test that we can add a new vector to the deserialized index
    let new_vector = Vector { id: 4, values: vec![1.0, 1.0, 1.0], text: "test".to_string(), metadata: None };
    assert!(deserialized.add(new_vector).is_ok());
    assert_eq!(deserialized.len(), 4);
    
    let query = vec![1.1, 0.1, 0.1];
    
    let results = deserialized.search(&query, 2, SimilarityMetric::Euclidean).unwrap();
    assert!(!results.is_empty(), "Search should return some results");
    assert!(results.len() <= 2, "Should return at most 2 results as requested");
    
    // Verify results are sorted by score (highest first)
    for i in 1..results.len() {
        assert!(results[i-1].score >= results[i].score, 
            "Results should be sorted by score (highest first)");
    }
    
    // Verify that all returned IDs are valid
    for result in &results {
        assert!(deserialized.get_vector(result.id).is_some(), 
            "All returned IDs should correspond to existing vectors");
    }
    
    // The first result should be the most similar to [1.1, 0.1, 0.1]
    // which should be the vector [1.0, 0.0, 0.0] (ID 1)
    assert_eq!(results[0].id, 1, "First result should be the most similar vector");
    assert!(results[0].score > 0.0, "Similarity score should be positive");
    
    // Test that we can still use the index for basic operations
    assert_eq!(deserialized.len(), 4); // Should still have 4 vectors after adding one
    assert_eq!(deserialized.dimension(), 3);
    assert!(!deserialized.is_empty());
}

#[test]
fn test_empty_hnsw_serialization_deserialization() {
    use serde_json;
    
    // Create an empty HNSW index
    let empty_hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    assert!(empty_hnsw.is_empty());
    assert_eq!(empty_hnsw.dimension(), 3);
    
    let serialized = serde_json::to_string(&empty_hnsw).expect("Serialization should work");
    let mut deserialized: HNSWIndex = serde_json::from_str(&serialized).expect("Deserialization should work");
    
    // Verify the deserialized empty index has the same properties
    assert_eq!(deserialized.len(), 0);
    assert_eq!(deserialized.dimension(), 3);
    assert!(deserialized.is_empty());
    
    // Test that we can add vectors to the deserialized empty index
    let vector = Vector { id: 1, values: vec![1.0, 2.0, 3.0], text: "test".to_string(), metadata: None };
    assert!(deserialized.add(vector).is_ok());
    assert_eq!(deserialized.len(), 1);
    assert!(!deserialized.is_empty());
}

#[test]
fn test_search_with_limited_vectors() {
    let mut hnsw = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    
    // Add only 3 vectors
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "test".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "test".to_string(), metadata: None },
    ];
    
    for vector in vectors {
        assert!(hnsw.add(vector).is_ok());
    }
    
    assert_eq!(hnsw.len(), 3);
    
    // Test searching for k=4 (more than we have vectors)
    // This should not panic and should return at most 3 results
    let query = vec![1.1, 0.1, 0.1];
    let results = hnsw.search(&query, 4, SimilarityMetric::Euclidean).unwrap();
    
    // Should return at most 3 results (all available vectors)
    assert!(results.len() <= 3);
    assert!(!results.is_empty());
    
    // Results should be sorted by score (highest first)
    for i in 1..results.len() {
        assert!(results[i-1].score >= results[i].score);
    }
}

/// Tests for distance to similarity conversion functions
#[cfg(test)]
mod conversion_tests {
    use super::{convert_distance_to_similarity, SimilarityMetric};

    #[test]
    fn test_euclidean_distance_conversion() {
        // Test zero distance (identical vectors)
        let distance = 0.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Euclidean);
        assert_eq!(similarity, 1.0, "Zero distance should give similarity of 1.0");
        
        // Test small distance
        let distance = 0.5;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Euclidean);
        let expected = 1.0 / (1.0 + 0.5);
        assert!((similarity - expected).abs() < 1e-10, "Small distance conversion should be correct");
        
        // Test medium distance
        let distance = 1.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Euclidean);
        let expected = 1.0 / (1.0 + 1.0);
        assert!((similarity - expected).abs() < 1e-10, "Medium distance conversion should be correct");
        
        // Test large distance
        let distance = 10.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Euclidean);
        let expected = 1.0 / (1.0 + 10.0);
        assert!((similarity - expected).abs() < 1e-10, "Large distance conversion should be correct");
        
        // Test very large distance
        let distance = 100.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Euclidean);
        assert!(similarity > 0.0 && similarity < 0.01, "Very large distance should give very low similarity");
    }

    #[test]
    fn test_cosine_distance_conversion() {
        // Test zero distance (identical vectors)
        let distance = 0.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Cosine);
        assert_eq!(similarity, 1.0, "Zero cosine distance should give similarity of 1.0");
        
        // Test small distance (similar vectors)
        let distance = 100.0; // cosine distance of 0.1 in scaled units
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Cosine);
        let expected = 1.0 - (100.0 / 1000.0);
        assert!((similarity - expected).abs() < 1e-10, "Small cosine distance conversion should be correct");
        
        // Test medium distance
        let distance = 500.0; // cosine distance of 0.5 in scaled units
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Cosine);
        let expected = 1.0 - (500.0 / 1000.0);
        assert!((similarity - expected).abs() < 1e-10, "Medium cosine distance conversion should be correct");
        
        // Test maximum distance (opposite vectors)
        let distance = 2000.0; // cosine distance of 2.0 in scaled units
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Cosine);
        let expected = 1.0 - (2000.0 / 1000.0);
        assert!((similarity - expected).abs() < 1e-10, "Maximum cosine distance conversion should be correct");
        
        // Verify similarity is bounded
        assert!((-1.0..=1.0).contains(&similarity), "Cosine similarity should be bounded");
    }

    #[test]
    fn test_manhattan_distance_conversion() {
        // Test zero distance (identical vectors)
        let distance = 0.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Manhattan);
        assert_eq!(similarity, 1.0, "Zero Manhattan distance should give similarity of 1.0");
        
        // Test small distance
        let distance = 1.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Manhattan);
        let expected = 1.0 / (1.0 + 1.0);
        assert!((similarity - expected).abs() < 1e-10, "Small Manhattan distance conversion should be correct");
        
        // Test medium distance
        let distance = 5.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Manhattan);
        let expected = 1.0 / (1.0 + 5.0);
        assert!((similarity - expected).abs() < 1e-10, "Medium Manhattan distance conversion should be correct");
        
        // Test large distance
        let distance = 20.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::Manhattan);
        let expected = 1.0 / (1.0 + 20.0);
        assert!((similarity - expected).abs() < 1e-10, "Large Manhattan distance conversion should be correct");
    }

    #[test]
    fn test_dotproduct_distance_conversion() {
        // Test zero distance (maximum dot product)
        let distance = 0.0;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::DotProduct);
        assert_eq!(similarity, 1.0, "Zero dot product distance should give similarity of 1.0");
        
        // Test small distance
        let distance = 100.0_f64;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::DotProduct);
        let ratio: f64 = (1000.0 - 100.0) / 1000.0;
        let expected: f64 = ratio.clamp(0.0, 1.0);
        assert!((similarity - expected).abs() < 1e-10, "Small dot product distance conversion should be correct");
        
        // Test medium distance
        let distance = 500.0_f64;
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::DotProduct);
        let ratio: f64 = (1000.0 - 500.0) / 1000.0;
        let expected: f64 = ratio.clamp(0.0, 1.0);
        assert!((similarity - expected).abs() < 1e-10, "Medium dot product distance conversion should be correct");
        
        // Test maximum distance
        let distance = 2000.0_f64; // negative dot product clamped
        let similarity = convert_distance_to_similarity(distance, SimilarityMetric::DotProduct);
        assert_eq!(similarity, 0.0, "Maximum dot product distance should give similarity of 0.0");
        
        // Test that similarity is bounded [0, 1]
        let distances = vec![0.0_f64, 100.0, 500.0, 1000.0, 1500.0, 2000.0];
        for dist in distances {
            let sim = convert_distance_to_similarity(dist, SimilarityMetric::DotProduct);
            assert!((0.0..=1.0).contains(&sim), "DotProduct similarity should be in [0, 1]");
        }
    }

    #[test]
    fn test_conversion_monotonicity() {
        // Test that similarity decreases monotonically with increasing distance
        // for all metrics
        
        let distances = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        
        for metric in &[
            SimilarityMetric::Euclidean,
            SimilarityMetric::Cosine,
            SimilarityMetric::Manhattan,
        ] {
            let mut prev_sim = 1.0;
            for &dist in &distances {
                let sim = convert_distance_to_similarity(dist, *metric);
                assert!(sim <= prev_sim, 
                    "Similarity should decrease as distance increases for {:?}", 
                    metric);
                prev_sim = sim;
            }
        }
    }

    #[test]
    fn test_conversion_edge_cases() {
        // Test with extremely small distances
        for metric in &[
            SimilarityMetric::Euclidean,
            SimilarityMetric::Cosine,
            SimilarityMetric::Manhattan,
            SimilarityMetric::DotProduct,
        ] {
            let sim = convert_distance_to_similarity(0.0001, *metric);
            assert!(sim > 0.9, "Extremely small distance should give high similarity");
            assert!(sim <= 1.0, "Similarity should not exceed 1.0");
        }
        
        // Test with extremely large distances
        for metric in &[
            SimilarityMetric::Euclidean,
            SimilarityMetric::Manhattan,
        ] {
            let sim = convert_distance_to_similarity(100000.0, *metric);
            assert!(sim > 0.0, "Even very large distance should give non-zero similarity");
            assert!(sim < 0.01, "Very large distance should give very low similarity");
        }
    }

    #[test]
    fn test_conversion_known_vectors() {
        // Test with actual vector calculations
        
        // Two identical vectors should have high similarity
        let identical_distance_euclidean = 0.0;
        let identical_distance_cosine = 0.0;
        
        let euclidean_sim = convert_distance_to_similarity(identical_distance_euclidean, SimilarityMetric::Euclidean);
        let cosine_sim = convert_distance_to_similarity(identical_distance_cosine, SimilarityMetric::Cosine);
        
        assert_eq!(euclidean_sim, 1.0);
        assert_eq!(cosine_sim, 1.0);
        
        // Opposite vectors (for cosine): [1,0,0] and [-1,0,0]
        // Cosine distance = 2 (in raw form) = 2000 (scaled by 1000)
        let opposite_distance = 2000.0;
        let cosine_sim = convert_distance_to_similarity(opposite_distance, SimilarityMetric::Cosine);
        assert!((cosine_sim - (-1.0)).abs() < 0.01, "Opposite vectors should have negative cosine similarity");
        
        // Perpendicular vectors: [1,0] and [0,1]
        // Cosine distance ≈ 1 = 1000 (scaled)
        let perpendicular_distance = 1000.0;
        let cosine_sim = convert_distance_to_similarity(perpendicular_distance, SimilarityMetric::Cosine);
        assert!((cosine_sim - 0.0).abs() < 0.01, "Perpendicular vectors should have cosine similarity ≈ 0");
    }

    #[test]
    fn test_scaling_factor_documentation() {
        // Verify the scaling factors used in the conversions
        // This helps document the behavior for future reference
        
        // Cosine: scaled by 1000.0
        // Maximum cosine distance is 2.0 (raw) = 2000 (scaled)
        let max_cosine_distance = 2000.0;
        let min_cosine_sim = convert_distance_to_similarity(max_cosine_distance, SimilarityMetric::Cosine);
        assert_eq!(min_cosine_sim, -1.0, "Maximum cosine distance should yield similarity of -1.0");
        
        // DotProduct: scaled by 1000.0
        // Maximum distance is when dot product is at minimum (negative)
        let max_dotproduct_distance = 2000.0;
        let min_dotproduct_sim = convert_distance_to_similarity(max_dotproduct_distance, SimilarityMetric::DotProduct);
        assert_eq!(min_dotproduct_sim, 0.0, "Maximum dot product distance should yield similarity of 0.0");
        
        // Euclidean and Manhattan use 1/(1+distance) formula
        // They don't have a hard upper bound but should approach 0
        let large_distance = 1000.0;
        let large_euclidean_sim = convert_distance_to_similarity(large_distance, SimilarityMetric::Euclidean);
        let large_manhattan_sim = convert_distance_to_similarity(large_distance, SimilarityMetric::Manhattan);
        assert!(large_euclidean_sim < 0.01 && large_euclidean_sim > 0.0);
        assert!(large_manhattan_sim < 0.01 && large_manhattan_sim > 0.0);
    }
}

