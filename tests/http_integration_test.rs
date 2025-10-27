use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;
use vectorlite::{VectorLiteClient, create_app};

// Mock embedding function for testing
struct MockEmbeddingFunction {
    dimension: usize,
}

impl MockEmbeddingFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl vectorlite::EmbeddingFunction for MockEmbeddingFunction {
    fn generate_embedding(&self, _text: &str) -> vectorlite::embeddings::Result<Vec<f64>> {
        // Return a simple mock embedding
        Ok(vec![1.0, 2.0, 3.0])
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

fn create_test_client() -> VectorLiteClient {
    let embedding_fn = MockEmbeddingFunction::new(3);
    VectorLiteClient::new(Box::new(embedding_fn))
}

#[tokio::test]
async fn test_health_check() {
    let client = create_test_client();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "healthy");
    assert_eq!(json["service"], "vectorlite");
}

#[tokio::test]
async fn test_list_collections_empty() {
    let client = create_test_client();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/collections")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["collections"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_create_collection() {
    let client = create_test_client();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let payload = json!({
        "name": "test_collection",
        "index_type": "flat"
    });

    let request = Request::builder()
        .uri("/collections")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "test_collection");
}

#[tokio::test]
async fn test_create_duplicate_collection() {
    let client = create_test_client();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let payload = json!({
        "name": "test_collection",
        "index_type": "flat"
    });

    // Create first collection
    let request = Request::builder()
        .uri("/collections")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Try to create duplicate
    let request = Request::builder()
        .uri("/collections")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_get_collection_info() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/collections/test_collection")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["info"]["name"], "test_collection");
    assert_eq!(json["info"]["count"], 0);
    assert!(json["info"]["is_empty"].as_bool().unwrap());
}

#[tokio::test]
async fn test_add_text_to_collection() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let payload = json!({
        "text": "Hello world"
    });

    let request = Request::builder()
        .uri("/collections/test_collection/text")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], 0);
}


#[tokio::test]
async fn test_search_text() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let payload = json!({
        "query": "Hello",
        "k": 5,
        "similarity_metric": "cosine"
    });

    let request = Request::builder()
        .uri("/collections/test_collection/search/text")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["results"].is_array());
    assert_eq!(json["results"].as_array().unwrap().len(), 1);
    assert_eq!(json["results"][0]["id"], 0);
}


#[tokio::test]
async fn test_get_vector() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/collections/test_collection/vectors/0")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["vector"]["id"], 0);
    assert_eq!(json["vector"]["values"], json!([1.0, 2.0, 3.0]));
}

#[tokio::test]
async fn test_delete_vector() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    client.add_text_to_collection("test_collection", "Hello world", None).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/collections/test_collection/vectors/0")
        .method("DELETE")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json.is_object());
}

#[tokio::test]
async fn test_delete_collection() {
    let mut client = create_test_client();
    client.create_collection("test_collection", vectorlite::IndexType::Flat, vectorlite::SimilarityMetric::Cosine).unwrap();
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let request = Request::builder()
        .uri("/collections/test_collection")
        .method("DELETE")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["name"], "test_collection");
}
