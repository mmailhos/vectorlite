# VectorLite

A high-performance, in-memory vector database optimized for AI agent workloads with HTTP API and thread-safe concurrency.

## Overview

VectorLite is designed for **single-instance, low-latency vector operations** in AI agent environments. It prioritizes **sub-millisecond search performance** over distributed scalability, making it ideal for:

- **AI Agent Sessions**: Session-scoped vector storage with fast retrieval
- **Real-time Search**: Sub-millisecond response requirements  
- **Prototype Development**: Rapid iteration without infrastructure complexity
- **Single-tenant Applications**: No multi-tenancy isolation requirements

### Key Features
- **In-memory storage** for zero-latency access patterns
- **Native Rust ML models** using Candle framework with pluggable architecture
- **Thread-safe concurrency** with RwLock per collection and atomic ID generation
- **HNSW indexing** for approximate nearest neighbor search with configurable accuracy

## HTTP API

RESTful interface optimized for AI agent integration:

```bash
# Health check
GET /health

# Collection management
GET /collections
POST /collections {"name": "docs", "index_type": "hnsw"}
DELETE /collections/{name}

# Vector operations
POST /collections/{name}/text {"text": "Hello world"}
POST /collections/{name}/vector {"id": 1, "values": [0.1, 0.2, ...]}
POST /collections/{name}/search/text {"query": "hello", "k": 10}
POST /collections/{name}/search/vector {"query": [0.1, 0.2, ...], "k": 10}
GET /collections/{name}/vectors/{id}
DELETE /collections/{name}/vectors/{id}
```

## Index Types

### FlatIndex
- **Complexity**: O(n) search, O(1) insert
- **Memory**: Linear with dataset size
- **Use Case**: Small datasets (< 10K vectors) or exact search requirements

### HNSWIndex
- **Complexity**: O(log n) search, O(log n) insert
- **Memory**: ~2-3x vector size due to graph structure
- **Use Case**: Large datasets with approximate search tolerance

## ML Model Integration

### Built-in Embedding Models
- **all-MiniLM-L6-v2**: Default 384-dimensional model for general-purpose text
- **Candle Framework**: Native Rust ML inference with CPU/GPU acceleration
- **Pluggable Architecture**: Easy integration of custom embedding models
- **Memory Efficient**: Models loaded once and shared across requests

### Similarity Metrics
- **Cosine**: Default for normalized embeddings, scale-invariant
- **Euclidean**: Geometric distance, sensitive to vector magnitude
- **Manhattan**: L1 norm, robust to outliers
- **Dot Product**: Raw similarity, requires consistent vector scaling

## Configuration Profiles

```bash
# Balanced (default)
cargo build

# Memory-constrained environments
cargo build --features memory-optimized

# High-precision search
cargo build --features high-accuracy
```


## Getting Started

```rust
use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};

// Create client with embedding function
let client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));

// Create collection
client.create_collection("documents", IndexType::HNSW)?;

// Add text (auto-generates embedding and ID)
let id = client.add_text_to_collection("documents", "Hello world")?;

// Search
let results = client.search_text_in_collection(
    "documents", 
    "hello", 
    5, 
    SimilarityMetric::Cosine
)?;
```

## HTTP Server Example

```rust
use vectorlite::{VectorLiteClient, EmbeddingGenerator, start_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
    start_server(client, "127.0.0.1", 3000).await?;
    Ok(())
}
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.