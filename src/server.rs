//! # HTTP Server Module
//!
//! This module provides HTTP API endpoints for VectorLite, enabling easy integration
//! with AI agents and other services. The server uses Axum for high-performance
//! async request handling.
//!
//! # API Endpoints
//!
//! ## Health Check
//! - `GET /health` - Server health status
//!
//! ## Collection Management
//! - `GET /collections` - List all collections
//! - `POST /collections` - Create a new collection
//! - `DELETE /collections/{name}` - Delete a collection
//!
//! ## Vector Operations
//! - `POST /collections/{name}/text` - Add text (auto-generates embedding, optional metadata)
//! - `POST /collections/{name}/search/text` - Search by text
//! - `GET /collections/{name}/vectors/{id}` - Get vector by ID
//! - `DELETE /collections/{name}/vectors/{id}` - Delete vector by ID
//!
//! ## Persistence Operations
//! - `POST /collections/{name}/save` - Save collection to file
//! - `POST /collections/load` - Load collection from file
//!
//! ### Save Collection
//! ```bash
//! curl -X POST http://localhost:3001/collections/my_docs/save \
//!      -H 'Content-Type: application/json' \
//!      -d '{"file_path": "./my_docs.vlc"}'
//! ```
//!
//! ### Load Collection
//! ```bash
//! curl -X POST http://localhost:3001/collections/load \
//!      -H 'Content-Type: application/json' \
//!      -d '{"file_path": "./my_docs.vlc", "collection_name": "restored_docs"}'
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use vectorlite::{VectorLiteClient, EmbeddingGenerator, start_server};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
//!     start_server(client, "127.0.0.1", 3002).await?;
//!     Ok(())
//! }
//! ```

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, error};

use crate::{VectorLiteClient, SearchResult, SimilarityMetric, IndexType};
use crate::errors::{VectorLiteError, VectorLiteResult};

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub index_type: String, // "flat" or "hnsw"
    #[serde(default)]
    pub metric: String, // "cosine", "euclidean", "manhattan", "dotproduct"
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AddTextRequest {
    pub text: String,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct AddTextResponse {
    pub id: Option<u64>,
}


#[derive(Debug, Deserialize)]
pub struct SearchTextRequest {
    pub query: String,
    pub k: Option<usize>,
    pub similarity_metric: Option<String>,
}


#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Option<Vec<SearchResult>>,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfoResponse {
    pub info: Option<crate::client::CollectionInfo>,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct SaveCollectionRequest {
    pub file_path: String,
}

#[derive(Debug, Serialize)]
pub struct SaveCollectionResponse {
    pub file_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoadCollectionRequest {
    pub file_path: String,
    pub collection_name: Option<String>, // Optional: if not provided, uses name from file
}

#[derive(Debug, Serialize)]
pub struct LoadCollectionResponse {
    pub collection_name: Option<String>,
}

// App state
pub type AppState = Arc<RwLock<VectorLiteClient>>;

// Helper functions
fn parse_index_type(index_type: &str) -> VectorLiteResult<IndexType> {
    match index_type.to_lowercase().as_str() {
        "flat" => Ok(IndexType::Flat),
        "hnsw" => Ok(IndexType::HNSW),
        _ => Err(VectorLiteError::InvalidIndexType { index_type: index_type.to_string() }),
    }
}

fn parse_similarity_metric(metric: &str) -> VectorLiteResult<SimilarityMetric> {
    match metric.to_lowercase().as_str() {
        "cosine" => Ok(SimilarityMetric::Cosine),
        "euclidean" => Ok(SimilarityMetric::Euclidean),
        "manhattan" => Ok(SimilarityMetric::Manhattan),
        "dotproduct" => Ok(SimilarityMetric::DotProduct),
        _ => Err(VectorLiteError::InvalidSimilarityMetric { metric: metric.to_string() }),
    }
}

// Handlers
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "vectorlite"
    }))
}

async fn list_collections(
    State(state): State<AppState>,
) -> Result<Json<ListCollectionsResponse>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let collections = client.list_collections();
    Ok(Json(ListCollectionsResponse {
        collections,
    }))
}

async fn create_collection(
    State(state): State<AppState>,
    Json(payload): Json<CreateCollectionRequest>,
) -> Result<Json<CreateCollectionResponse>, StatusCode> {
    let index_type = match parse_index_type(&payload.index_type) {
        Ok(t) => t,
        Err(e) => {
            return Err(e.status_code());
        }
    };

    // Parse metric, default to cosine if not provided
    let metric = if payload.metric.is_empty() {
        SimilarityMetric::Cosine  // Default to cosine if not provided
    } else {
        match parse_similarity_metric(&payload.metric) {
            Ok(m) => m,
            Err(e) => {
                return Err(e.status_code());
            }
        }
    };

    let mut client = state.write().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.create_collection(&payload.name, index_type, metric) {
        Ok(_) => {
            info!("Created collection: {}", payload.name);
            Ok(Json(CreateCollectionResponse {
                name: payload.name,
            }))
        }
        Err(e) => {
            error!("Failed to create collection '{}': {}", payload.name, e);
            Err(e.status_code())
        }
    }
}

async fn get_collection_info(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CollectionInfoResponse>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.get_collection_info(&collection_name) {
        Ok(info) => Ok(Json(CollectionInfoResponse {
            info: Some(info),
        })),
        Err(e) => Err(e.status_code()),
    }
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> Result<Json<CreateCollectionResponse>, StatusCode> {
    let mut client = state.write().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.delete_collection(&collection_name) {
        Ok(_) => {
            info!("Deleted collection: {}", collection_name);
            Ok(Json(CreateCollectionResponse {
                name: collection_name,
            }))
        }
        Err(e) => Err(e.status_code()),
    }
}

async fn add_text(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<AddTextRequest>,
) -> Result<Json<AddTextResponse>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.add_text_to_collection(&collection_name, &payload.text, payload.metadata) {
        Ok(id) => {
            info!("Added text to collection '{}' with ID: {}", collection_name, id);
            Ok(Json(AddTextResponse {
                id: Some(id),
            }))
        }
        Err(e) => {
            Err(e.status_code())
        }
    }
}



async fn search_text(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<SearchTextRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let k = payload.k.unwrap_or(10);
    let similarity_metric = match payload.similarity_metric {
        Some(metric) => match parse_similarity_metric(&metric) {
            Ok(m) => m,
            Err(e) => {
                return Err(e.status_code());
            }
        },
        None => SimilarityMetric::Cosine,
    };

    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.search_text_in_collection(&collection_name, &payload.query, k, similarity_metric) {
        Ok(results) => {
            info!("Search completed for collection '{}' with {} results", collection_name, results.len());
            Ok(Json(SearchResponse {
                results: Some(results),
            }))
        }
        Err(e) => Err(e.status_code()),
    }
}


async fn get_vector(
    State(state): State<AppState>,
    Path((collection_name, vector_id)): Path<(String, u64)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.get_vector_from_collection(&collection_name, vector_id) {
        Ok(Some(vector)) => {
            Ok(Json(serde_json::json!({
                "vector": vector
            })))
        }
        Ok(None) => {
            Err(StatusCode::NOT_FOUND)
        }
        Err(e) => {
            Err(e.status_code())
        }
    }
}

async fn delete_vector(
    State(state): State<AppState>,
    Path((collection_name, vector_id)): Path<(String, u64)>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.delete_from_collection(&collection_name, vector_id) {
        Ok(_) => {
            info!("Deleted vector {} from collection '{}'", vector_id, collection_name);
            Ok(Json(serde_json::json!({})))
        }
        Err(e) => {
            Err(e.status_code())
        }
    }
}

async fn save_collection(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<SaveCollectionRequest>,
) -> Result<Json<SaveCollectionResponse>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Get the collection
    let collection = match client.get_collection(&collection_name) {
        Some(collection) => collection,
        None => {
            return Err(StatusCode::NOT_FOUND);
        }
    };

    // Convert file path to PathBuf
    let file_path = PathBuf::from(&payload.file_path);
    
    // Save the collection
    match collection.save_to_file(&file_path) {
        Ok(_) => {
            info!("Saved collection '{}' to file: {}", collection_name, payload.file_path);
            Ok(Json(SaveCollectionResponse {
                file_path: Some(payload.file_path),
            }))
        }
        Err(_) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn load_collection(
    State(state): State<AppState>,
    Json(payload): Json<LoadCollectionRequest>,
) -> Result<Json<LoadCollectionResponse>, StatusCode> {
    // Convert file path to PathBuf
    let file_path = PathBuf::from(&payload.file_path);
    
    // Load the collection from file
    let collection = match crate::Collection::load_from_file(&file_path) {
        Ok(collection) => collection,
        Err(e) => {
            // Check if it's a file not found error
            if let crate::persistence::PersistenceError::Io(io_err) = &e
                && io_err.kind() == std::io::ErrorKind::NotFound {
                return Err(VectorLiteError::FileNotFound(format!("File not found: {}", payload.file_path)).status_code());
            }
            return Err(VectorLiteError::from(e).status_code());
        }
    };

    // Determine the collection name to use
    let collection_name = payload.collection_name.unwrap_or_else(|| collection.name().to_string());
    
    // Add the collection to the client
    let mut client = state.write().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Check if collection already exists
    if client.has_collection(&collection_name) {
        return Err(StatusCode::CONFLICT);
    }

    // Extract the index from the loaded collection
    let index = {
        let index_guard = collection.index_read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        (*index_guard).clone()
    };
    
    // Create a new collection with the loaded data
    let new_collection = crate::Collection::new(collection_name.clone(), index);
    
    // Add the collection to the client
    if client.add_collection(new_collection).is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    
    info!("Loaded collection '{}' from file: {}", collection_name, payload.file_path);
    Ok(Json(LoadCollectionResponse {
        collection_name: Some(collection_name),
    }))
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/collections", get(list_collections))
        .route("/collections", post(create_collection))
        .route("/collections/:name", get(get_collection_info))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/text", post(add_text))
        .route("/collections/:name/search/text", post(search_text))
        .route("/collections/:name/vectors/:id", get(get_vector))
        .route("/collections/:name/vectors/:id", delete(delete_vector))
        .route("/collections/:name/save", post(save_collection))
        .route("/collections/load", post(load_collection))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn start_server(
    client: VectorLiteClient,
    host: &str,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app(Arc::new(RwLock::new(client)));
    
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
    info!("VectorLite server starting on {}:{}", host, port);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}
