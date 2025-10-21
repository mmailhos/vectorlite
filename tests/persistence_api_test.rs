//! # Persistence API Integration Tests
//!
//! Tests the HTTP API endpoints for collection persistence functionality.

use vectorlite::{VectorLiteClient, create_app};
use axum::http::{Method, StatusCode};
use axum::body::Body;
use tower::ServiceExt;
use serde_json::json;

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
        Ok(vec![1.0; self.dimension])
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[tokio::test]
async fn test_save_collection_api() {
    // Create a client with mock embedding function
    let client = VectorLiteClient::new(Box::new(MockEmbeddingFunction::new(3)));
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    // Create a collection
    let create_request = json!({
        "name": "test_collection",
        "index_type": "flat"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(create_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Add some text to the collection
    let add_text_request = json!({
        "text": "Hello world"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/test_collection/text")
                .header("content-type", "application/json")
                .body(Body::from(add_text_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Save the collection
    let save_request = json!({
        "file_path": "./test_collection.vlc"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/test_collection/save")
                .header("content-type", "application/json")
                .body(Body::from(save_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify the file was created
    assert!(std::path::Path::new("./test_collection.vlc").exists());

    // Clean up
    let _ = std::fs::remove_file("./test_collection.vlc");
}

#[tokio::test]
async fn test_load_collection_api() {
    // First, create and save a collection
    let client = VectorLiteClient::new(Box::new(MockEmbeddingFunction::new(3)));
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    // Create a collection
    let create_request = json!({
        "name": "source_collection",
        "index_type": "flat"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections")
                .header("content-type", "application/json")
                .body(Body::from(create_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Add some text
    let add_text_request = json!({
        "text": "Test document"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/source_collection/text")
                .header("content-type", "application/json")
                .body(Body::from(add_text_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Save the collection
    let save_request = json!({
        "file_path": "./test_load_collection.vlc"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/source_collection/save")
                .header("content-type", "application/json")
                .body(Body::from(save_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Now test loading the collection
    let load_request = json!({
        "file_path": "./test_load_collection.vlc",
        "collection_name": "loaded_collection"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/load")
                .header("content-type", "application/json")
                .body(Body::from(load_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify the loaded collection exists
    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::GET)
                .uri("/collections/loaded_collection")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Clean up
    let _ = std::fs::remove_file("./test_load_collection.vlc");
}

#[tokio::test]
async fn test_save_nonexistent_collection() {
    let client = VectorLiteClient::new(Box::new(MockEmbeddingFunction::new(3)));
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let save_request = json!({
        "file_path": "./nonexistent.vlc"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/nonexistent/save")
                .header("content-type", "application/json")
                .body(Body::from(save_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // The response should indicate failure
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["success"], false);
    assert!(response_json["message"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_load_nonexistent_file() {
    let client = VectorLiteClient::new(Box::new(MockEmbeddingFunction::new(3)));
    let app = create_app(std::sync::Arc::new(std::sync::RwLock::new(client)));

    let load_request = json!({
        "file_path": "./nonexistent.vlc",
        "collection_name": "test"
    });

    let response = app
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method(Method::POST)
                .uri("/collections/load")
                .header("content-type", "application/json")
                .body(Body::from(load_request.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // The response should indicate failure
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response_json["success"], false);
    assert!(response_json["message"].as_str().unwrap().contains("Failed to load"));
}
