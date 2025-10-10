# VectorLite

A high-performance, in-memory vector database optimized for AI agent workloads with HTTP API and thread-safe concurrency.

## Architecture Philosophy

VectorLite is designed for **single-instance, low-latency vector operations** in AI agent environments. It prioritizes **sub-millisecond search performance** over distributed scalability, making it ideal for session-based AI applications where data locality and response time are critical.

## Design Decisions

### In-Memory Architecture
- **Pure in-memory storage** for zero-latency access patterns
- **Session-scoped data** optimized for AI agent workflows
- **Predictable memory bounds** with no persistence overhead

### Thread-Safe Concurrency Model
- **RwLock per collection** enabling concurrent searches
- **Atomic ID generation** for lock-free operations
- **No distributed coordination** overhead for single-instance performance

### Built-in ML Integration
- **Native Rust ML models** using Candle framework
- **Interchangeable embedding models** with pluggable architecture
- **Pre-computed embeddings** to minimize lock contention

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

## Concurrency Model

### Lock Hierarchy
1. **Collection Level**: Each collection has independent RwLock
2. **Index Level**: HNSW/Flat index wrapped in Arc<RwLock<VectorIndexWrapper>>
3. **ID Generation**: AtomicU64 with Relaxed ordering for maximum performance

### Memory Ordering Strategy
- **ID Generation**: `Ordering::Relaxed` - sufficient for uniqueness guarantees
- **Index Operations**: RwLock provides sequential consistency
- **Embedding Generation**: Outside critical sections to minimize contention

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

## Use Cases

### Optimal Scenarios
- **AI Agent Sessions**: Session-scoped vector storage with fast retrieval
- **Real-time Search**: Sub-millisecond response requirements
- **Prototype Development**: Rapid iteration without infrastructure complexity
- **Single-tenant Applications**: No multi-tenancy isolation requirements

### Anti-Patterns
- **Distributed Systems**: No built-in clustering or replication
- **Persistent Storage**: Data loss on restart
- **Multi-tenant SaaS**: No tenant isolation or resource limits
- **Large-scale Analytics**: Not optimized for batch processing

## Key Advantages

| Aspect | Benefit |
|--------|---------|
| **Storage** | Zero-latency access with predictable memory usage |
| **Concurrency** | Multiple concurrent searches with simple lock model |
| **ML Integration** | Native Rust models with pluggable architecture |
| **Indexing** | HNSW provides O(log n) search with configurable accuracy |
| **API** | Simple synchronous HTTP interface for easy integration |
| **Performance** | Single-instance optimization for maximum throughput |

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