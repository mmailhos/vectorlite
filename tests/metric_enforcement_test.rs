use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric, HNSWIndex, VectorIndex, Vector};

#[test]
fn test_hnsw_metric_enforcement() {
    // Test that HNSW index enforces the metric it was built with
    let mut hnsw_euclidean = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    let mut hnsw_cosine = HNSWIndex::new(3, SimilarityMetric::Cosine);
    let mut hnsw_manhattan = HNSWIndex::new(3, SimilarityMetric::Manhattan);
    let mut hnsw_dotproduct = HNSWIndex::new(3, SimilarityMetric::DotProduct);
    
    // Add the same vectors to all indices
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "first".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "second".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "third".to_string(), metadata: None },
    ];
    
    for vector in &vectors {
        hnsw_euclidean.add(vector.clone()).unwrap();
        hnsw_cosine.add(vector.clone()).unwrap();
        hnsw_manhattan.add(vector.clone()).unwrap();
        hnsw_dotproduct.add(vector.clone()).unwrap();
    }
    
    let query = vec![1.1, 0.1, 0.1];
    
    // Test that each index only accepts its own metric
    
    // Euclidean index should only work with Euclidean metric
    let results = hnsw_euclidean.search(&query, 2, SimilarityMetric::Euclidean);
    assert!(!results.is_empty(), "Euclidean index should work with Euclidean metric");
    
    let results = hnsw_euclidean.search(&query, 2, SimilarityMetric::Cosine);
    assert!(results.is_empty(), "Euclidean index should reject Cosine metric");
    
    let results = hnsw_euclidean.search(&query, 2, SimilarityMetric::Manhattan);
    assert!(results.is_empty(), "Euclidean index should reject Manhattan metric");
    
    let results = hnsw_euclidean.search(&query, 2, SimilarityMetric::DotProduct);
    assert!(results.is_empty(), "Euclidean index should reject DotProduct metric");
    
    // Cosine index should only work with Cosine metric
    let results = hnsw_cosine.search(&query, 2, SimilarityMetric::Cosine);
    assert!(!results.is_empty(), "Cosine index should work with Cosine metric");
    
    let results = hnsw_cosine.search(&query, 2, SimilarityMetric::Euclidean);
    assert!(results.is_empty(), "Cosine index should reject Euclidean metric");
    
    let results = hnsw_cosine.search(&query, 2, SimilarityMetric::Manhattan);
    assert!(results.is_empty(), "Cosine index should reject Manhattan metric");
    
    let results = hnsw_cosine.search(&query, 2, SimilarityMetric::DotProduct);
    assert!(results.is_empty(), "Cosine index should reject DotProduct metric");
    
    // Manhattan index should only work with Manhattan metric
    let results = hnsw_manhattan.search(&query, 2, SimilarityMetric::Manhattan);
    assert!(!results.is_empty(), "Manhattan index should work with Manhattan metric");
    
    let results = hnsw_manhattan.search(&query, 2, SimilarityMetric::Euclidean);
    assert!(results.is_empty(), "Manhattan index should reject Euclidean metric");
    
    let results = hnsw_manhattan.search(&query, 2, SimilarityMetric::Cosine);
    assert!(results.is_empty(), "Manhattan index should reject Cosine metric");
    
    let results = hnsw_manhattan.search(&query, 2, SimilarityMetric::DotProduct);
    assert!(results.is_empty(), "Manhattan index should reject DotProduct metric");
    
    // DotProduct index should only work with DotProduct metric
    let results = hnsw_dotproduct.search(&query, 2, SimilarityMetric::DotProduct);
    assert!(!results.is_empty(), "DotProduct index should work with DotProduct metric");
    
    let results = hnsw_dotproduct.search(&query, 2, SimilarityMetric::Euclidean);
    assert!(results.is_empty(), "DotProduct index should reject Euclidean metric");
    
    let results = hnsw_dotproduct.search(&query, 2, SimilarityMetric::Cosine);
    assert!(results.is_empty(), "DotProduct index should reject Cosine metric");
    
    let results = hnsw_dotproduct.search(&query, 2, SimilarityMetric::Manhattan);
    assert!(results.is_empty(), "DotProduct index should reject Manhattan metric");
}

#[test]
fn test_client_metric_enforcement() {
    // Test that collections created with specific metrics enforce them
    let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new().unwrap()));
    
    // Create collections with different metrics
    client.create_collection("euclidean_collection", IndexType::HNSW, SimilarityMetric::Euclidean).unwrap();
    client.create_collection("cosine_collection", IndexType::HNSW, SimilarityMetric::Cosine).unwrap();
    client.create_collection("manhattan_collection", IndexType::HNSW, SimilarityMetric::Manhattan).unwrap();
    client.create_collection("dotproduct_collection", IndexType::HNSW, SimilarityMetric::DotProduct).unwrap();
    
    // Add the same text to all collections
    client.add_text_to_collection("euclidean_collection", "Hello world", None).unwrap();
    client.add_text_to_collection("cosine_collection", "Hello world", None).unwrap();
    client.add_text_to_collection("manhattan_collection", "Hello world", None).unwrap();
    client.add_text_to_collection("dotproduct_collection", "Hello world", None).unwrap();
    
    // Test that searching with the correct metric works
    let results = client.search_text_in_collection("euclidean_collection", "hello", 1, SimilarityMetric::Euclidean).unwrap();
    assert!(!results.is_empty(), "Euclidean collection should work with Euclidean metric");
    
    let results = client.search_text_in_collection("cosine_collection", "hello", 1, SimilarityMetric::Cosine).unwrap();
    assert!(!results.is_empty(), "Cosine collection should work with Cosine metric");
    
    let results = client.search_text_in_collection("manhattan_collection", "hello", 1, SimilarityMetric::Manhattan).unwrap();
    assert!(!results.is_empty(), "Manhattan collection should work with Manhattan metric");
    
    let results = client.search_text_in_collection("dotproduct_collection", "hello", 1, SimilarityMetric::DotProduct).unwrap();
    assert!(!results.is_empty(), "DotProduct collection should work with DotProduct metric");
    
    // Test that searching with wrong metrics returns empty results (due to HNSW rejection)
    let results = client.search_text_in_collection("euclidean_collection", "hello", 1, SimilarityMetric::Cosine).unwrap();
    assert!(results.is_empty(), "Euclidean collection should reject Cosine metric");
    
    let results = client.search_text_in_collection("cosine_collection", "hello", 1, SimilarityMetric::Euclidean).unwrap();
    assert!(results.is_empty(), "Cosine collection should reject Euclidean metric");
    
    let results = client.search_text_in_collection("manhattan_collection", "hello", 1, SimilarityMetric::Euclidean).unwrap();
    assert!(results.is_empty(), "Manhattan collection should reject Euclidean metric");
    
    let results = client.search_text_in_collection("dotproduct_collection", "hello", 1, SimilarityMetric::Euclidean).unwrap();
    assert!(results.is_empty(), "DotProduct collection should reject Euclidean metric");
}

#[test]
fn test_hnsw_metric_accuracy() {
    // Test that each metric type returns correct results for its specific distance calculation
    
    let mut hnsw_euclidean = HNSWIndex::new(3, SimilarityMetric::Euclidean);
    let mut hnsw_cosine = HNSWIndex::new(3, SimilarityMetric::Cosine);
    
    // Add orthogonal vectors (for cosine, these have 0 similarity)
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "x-axis".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "y-axis".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "z-axis".to_string(), metadata: None },
        Vector { id: 4, values: vec![1.0, 1.0, 1.0], text: "diagonal".to_string(), metadata: None },
    ];
    
    for vector in &vectors {
        hnsw_euclidean.add(vector.clone()).unwrap();
        hnsw_cosine.add(vector.clone()).unwrap();
    }
    
    // Query close to x-axis
    let query = vec![0.9, 0.1, 0.1];
    
    // For Euclidean, closest should be x-axis (id=1)
    let euclidean_results = hnsw_euclidean.search(&query, 1, SimilarityMetric::Euclidean);
    assert_eq!(euclidean_results.len(), 1);
    assert_eq!(euclidean_results[0].id, 1, "Euclidean should find x-axis as closest");
    
    // For Cosine, the angle matters more than distance
    // The normalized query direction is very close to x-axis
    let cosine_results = hnsw_cosine.search(&query, 1, SimilarityMetric::Cosine);
    assert_eq!(cosine_results.len(), 1);
    assert_eq!(cosine_results[0].id, 1, "Cosine should also find x-axis as most similar");
}

#[test]
fn test_flat_index_accepts_any_metric() {
    // Flat index should accept any metric at search time since it's a brute-force search
    use vectorlite::FlatIndex;
    
    let mut flat = FlatIndex::new(3, Vec::new());
    
    let vectors = vec![
        Vector { id: 1, values: vec![1.0, 0.0, 0.0], text: "first".to_string(), metadata: None },
        Vector { id: 2, values: vec![0.0, 1.0, 0.0], text: "second".to_string(), metadata: None },
        Vector { id: 3, values: vec![0.0, 0.0, 1.0], text: "third".to_string(), metadata: None },
    ];
    
    for vector in &vectors {
        flat.add(vector.clone()).unwrap();
    }
    
    let query = vec![1.1, 0.1, 0.1];
    
    // Flat index should work with all metrics
    let results = flat.search(&query, 2, SimilarityMetric::Euclidean);
    assert!(!results.is_empty(), "Flat index should work with Euclidean");
    
    let results = flat.search(&query, 2, SimilarityMetric::Cosine);
    assert!(!results.is_empty(), "Flat index should work with Cosine");
    
    let results = flat.search(&query, 2, SimilarityMetric::Manhattan);
    assert!(!results.is_empty(), "Flat index should work with Manhattan");
    
    let results = flat.search(&query, 2, SimilarityMetric::DotProduct);
    assert!(!results.is_empty(), "Flat index should work with DotProduct");
}
