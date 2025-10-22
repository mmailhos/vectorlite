# VectorLite

[![Crates.io](https://img.shields.io/crates/v/vectorlite.svg)](https://crates.io/crates/vectorlite)
[![docs.rs](https://docs.rs/vectorlite/badge.svg)](https://docs.rs/vectorlite)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Tests](https://github.com/mmailhos/vectorlite/workflows/Rust/badge.svg?branch=main)](https://github.com/mmailhos/vectorlite/actions)

A high-performance, in-memory vector database optimized for AI agent workloads with HTTP API and thread-safe concurrency.

## Overview

VectorLite is designed for **single-instance, low-latency vector operations** in AI agent environments. It prioritizes **sub-millisecond search performance** over distributed scalability, making it ideal for:

- **AI Agent Sessions**: Session-scoped vector storage with fast retrieval
- **Real-time Search**: Sub-millisecond response requirements for pre-computed embeddings
- **Prototype Development**: Rapid iteration without infrastructure complexity
- **Single-tenant Applications**: No multi-tenancy isolation requirements

### Key Features
- **In-memory storage** for zero-latency access patterns
- **Native Rust ML models** using [Candle](https://github.com/huggingface/candle) framework with pluggable architecture. Bring your own embedding model (default to [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
- **Thread-safe concurrency** with RwLock per collection and atomic ID generation
- **HNSW indexing** for approximate nearest neighbor search with configurable accuracy
- **Collection persistence** with vector lite collection (VLC) file format for saving/loading collections

## HTTP API

Presentation of the RESTful interface. 

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

# Persistence operations
POST /collections/{name}/save {"file_path": "./collection.vlc"}
POST /collections/load {"file_path": "./collection.vlc", "collection_name": "restored"}
```

## Index Types

### Flat
- **Complexity**: O(n) search, O(1) insert
- **Memory**: Linear with dataset size
- **Use Case**: Small datasets (< 10K vectors) or exact search requirements

### HNSW
- **Complexity**: O(log n) search, O(log n) insert
- **Memory**: ~2-3x vector size due to graph structure
- **Use Case**: Large datasets with approximate search tolerance

See [Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320) paper for details.

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

### CLI Usage

Start the server with optional collection loading:

```bash
# Start empty server
cargo run --bin vectorlite -- --port 3002

# Start with pre-loaded collection
cargo run --bin vectorlite -- --filepath ./my_collection.vlc --port 3002
```

## Testing

Run tests with mock embeddings (CI-friendly, no model files required):
```bash
cargo test --features mock-embeddings
```

Run tests with real ML models (requires downloaded models):
```bash
cargo test
```

### Download ML Model

This downloads the BERT-based embedding model files needed for real embedding generation:
```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
