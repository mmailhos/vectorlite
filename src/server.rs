use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, error};

use crate::{VectorLiteClient, Vector, SearchResult, SimilarityMetric, IndexType};

// Request/Response types
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub index_type: String, // "flat" or "hnsw"
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct AddTextRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AddTextResponse {
    pub success: bool,
    pub id: Option<u64>,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct AddVectorRequest {
    pub id: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct AddVectorResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchTextRequest {
    pub query: String,
    pub k: Option<usize>,
    pub similarity_metric: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchVectorRequest {
    pub query: Vec<f64>,
    pub k: Option<usize>,
    pub similarity_metric: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub success: bool,
    pub results: Option<Vec<SearchResult>>,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfoResponse {
    pub success: bool,
    pub info: Option<crate::client::CollectionInfo>,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub success: bool,
    pub collections: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub success: bool,
    pub message: String,
}

// App state
pub type AppState = Arc<RwLock<VectorLiteClient>>;

// Helper functions
fn parse_index_type(index_type: &str) -> Result<IndexType, String> {
    match index_type.to_lowercase().as_str() {
        "flat" => Ok(IndexType::Flat),
        "hnsw" => Ok(IndexType::HNSW),
        _ => Err(format!("Invalid index type: {}. Must be 'flat' or 'hnsw'", index_type)),
    }
}

fn parse_similarity_metric(metric: &str) -> Result<SimilarityMetric, String> {
    match metric.to_lowercase().as_str() {
        "cosine" => Ok(SimilarityMetric::Cosine),
        "euclidean" => Ok(SimilarityMetric::Euclidean),
        "manhattan" => Ok(SimilarityMetric::Manhattan),
        "dotproduct" => Ok(SimilarityMetric::DotProduct),
        _ => Err(format!("Invalid similarity metric: {}. Must be 'cosine', 'euclidean', 'manhattan', or 'dotproduct'", metric)),
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
        success: true,
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
            return Ok(Json(CreateCollectionResponse {
                success: false,
                message: e,
            }));
        }
    };

    let mut client = state.write().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.create_collection(&payload.name, index_type) {
        Ok(_) => {
            info!("Created collection: {}", payload.name);
            Ok(Json(CreateCollectionResponse {
                success: true,
                message: format!("Collection '{}' created successfully", payload.name),
            }))
        }
        Err(e) => {
            error!("Failed to create collection '{}': {}", payload.name, e);
            Ok(Json(CreateCollectionResponse {
                success: false,
                message: e,
            }))
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
            success: true,
            info: Some(info),
            message: "Collection info retrieved successfully".to_string(),
        })),
        Err(e) => Ok(Json(CollectionInfoResponse {
            success: false,
            info: None,
            message: e,
        })),
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
                success: true,
                message: format!("Collection '{}' deleted successfully", collection_name),
            }))
        }
        Err(e) => {
            error!("Failed to delete collection '{}': {}", collection_name, e);
            Ok(Json(CreateCollectionResponse {
                success: false,
                message: e,
            }))
        }
    }
}

async fn add_text(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<AddTextRequest>,
) -> Result<Json<AddTextResponse>, StatusCode> {
    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.add_text_to_collection(&collection_name, &payload.text) {
        Ok(id) => {
            info!("Added text to collection '{}' with ID: {}", collection_name, id);
            Ok(Json(AddTextResponse {
                success: true,
                id: Some(id),
                message: "Text added successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to add text to collection '{}': {}", collection_name, e);
            Ok(Json(AddTextResponse {
                success: false,
                id: None,
                message: e,
            }))
        }
    }
}

async fn add_vector(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<AddVectorRequest>,
) -> Result<Json<AddVectorResponse>, StatusCode> {
    let vector = Vector {
        id: payload.id,
        values: payload.values,
    };

    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.add_vector_to_collection(&collection_name, vector) {
        Ok(_) => {
            info!("Added vector to collection '{}' with ID: {}", collection_name, payload.id);
            Ok(Json(AddVectorResponse {
                success: true,
                message: "Vector added successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to add vector to collection '{}': {}", collection_name, e);
            Ok(Json(AddVectorResponse {
                success: false,
                message: e,
            }))
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
                return Ok(Json(SearchResponse {
                    success: false,
                    results: None,
                    message: e,
                }));
            }
        },
        None => SimilarityMetric::Cosine,
    };

    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.search_text_in_collection(&collection_name, &payload.query, k, similarity_metric) {
        Ok(results) => {
            info!("Search completed for collection '{}' with {} results", collection_name, results.len());
            Ok(Json(SearchResponse {
                success: true,
                results: Some(results),
                message: "Search completed successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Search failed for collection '{}': {}", collection_name, e);
            Ok(Json(SearchResponse {
                success: false,
                results: None,
                message: e,
            }))
        }
    }
}

async fn search_vector(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(payload): Json<SearchVectorRequest>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let k = payload.k.unwrap_or(10);
    let similarity_metric = match payload.similarity_metric {
        Some(metric) => match parse_similarity_metric(&metric) {
            Ok(m) => m,
            Err(e) => {
                return Ok(Json(SearchResponse {
                    success: false,
                    results: None,
                    message: e,
                }));
            }
        },
        None => SimilarityMetric::Cosine,
    };

    let client = state.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    match client.search_vector_in_collection(&collection_name, &payload.query, k, similarity_metric) {
        Ok(results) => {
            info!("Vector search completed for collection '{}' with {} results", collection_name, results.len());
            Ok(Json(SearchResponse {
                success: true,
                results: Some(results),
                message: "Vector search completed successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Vector search failed for collection '{}': {}", collection_name, e);
            Ok(Json(SearchResponse {
                success: false,
                results: None,
                message: e,
            }))
        }
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
                "success": true,
                "vector": vector,
                "message": "Vector retrieved successfully"
            })))
        }
        Ok(None) => {
            Ok(Json(serde_json::json!({
                "success": false,
                "message": "Vector not found"
            })))
        }
        Err(e) => {
            error!("Failed to get vector {} from collection '{}': {}", vector_id, collection_name, e);
            Ok(Json(serde_json::json!({
                "success": false,
                "message": e
            })))
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
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Vector deleted successfully"
            })))
        }
        Err(e) => {
            error!("Failed to delete vector {} from collection '{}': {}", vector_id, collection_name, e);
            Ok(Json(serde_json::json!({
                "success": false,
                "message": e
            })))
        }
    }
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/collections", get(list_collections))
        .route("/collections", post(create_collection))
        .route("/collections/:name", get(get_collection_info))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/text", post(add_text))
        .route("/collections/:name/vector", post(add_vector))
        .route("/collections/:name/search/text", post(search_text))
        .route("/collections/:name/search/vector", post(search_vector))
        .route("/collections/:name/vectors/:id", get(get_vector))
        .route("/collections/:name/vectors/:id", delete(delete_vector))
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
