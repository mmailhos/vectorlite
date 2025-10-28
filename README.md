# VectorLite

[![Crates.io](https://img.shields.io/crates/v/vectorlite.svg)](https://crates.io/crates/vectorlite)
[![docs.rs](https://img.shields.io/docsrs/vectorlite/latest)](https://docs.rs/vectorlite)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Tests](https://github.com/mmailhos/vectorlite/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/mmailhos/vectorlite/actions)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0.3-green.svg)](docs/openapi.yaml)

**A tiny, in-process Rust vector store with built-in embeddings for sub-millisecond semantic search.**

VectorLite is a high-performance, **in-memory vector database** optimized for **AI agent** and **edge** workloads.  
It co-locates model inference (via [Candle](https://github.com/huggingface/candle)) with a low-latency vector index, making it ideal for **session-scoped**, **single-instance**, or **privacy-sensitive** environments.

## Why VectorLite?
| Feature | Description |
|----------|-------------|
| **Sub-millisecond search** | In-memory HNSW or flat search tuned for real-time agent loops. |
| **Built-in embeddings** | Runs [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) locally using Candle, or any other model of your choice. No external API calls. |
| **Single-binary simplicity** | No dependencies, no servers to orchestrate. Start instantly via CLI or Docker. |
| **Session-scoped collections** | Perfect for ephemeral agent sessions or sidecars |
| **Thread-safe concurrency** | RwLock-based access and atomic ID generation for multi-threaded workloads. |
| **Instant persistence** | Save or restore collections snapshots in one call. |

VectorLite trades distributed scalability for deterministic performance, perfect for use cases where latency mattters more than millions of vectors.

## When to Use It

| Scenario | Why VectorLite fits |
|-----------|--------------------|
| **AI agent sessions** | Keep short-lived embeddings per conversation. No network latency. |
| **Edge or embedded AI** | Run fully offline with model + index in one binary. |
| **Realtime search / personalization** | Sub-ms search for pre-computed embeddings. |
| **Local prototyping & CI** | Rust-native, no external services. |
| **Single-tenant microservices** | Lightweight sidecar for semantic capabilities. |

## Quick Start

### Run from Source
```bash
cargo run --bin vectorlite -- --port 3001

# Start with preloaded collection
cargo run --bin vectorlite -- --filepath ./my_collection.vlc --port 3001
```

### Run with Docker

With default settings:
```bash
docker build -t vectorlite .
docker run -p 3001:3001 vectorlite
```


With a different embeddings model and memory-optimized HNSW:
```bash
docker build \
  --build-arg MODEL_NAME="sentence-transformers/paraphrase-MiniLM-L3-v2" \
  --build-arg FEATURES="memory-optimized" \
  -t vectorlite-small .
```

## HTTP API Overview
| Operation             | Method & Endpoint                         | Body                                                               |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------ |
| **Health**            | `GET /health`                             | –                                                                  |
| **List collections**  | `GET /collections`                        | –                                                                  |
| **Create collection** | `POST /collections`                       | `{"name": "docs", "index_type": "hnsw", "metric": "cosine"}`|
| **Delete collection** | `DELETE /collections/{name}`              | –                                                                  |
| **Add text**          | `POST /collections/{name}/text`           | `{"text": "Hello world", "metadata": {...}}`|
| **Search (text)**     | `POST /collections/{name}/search/text`    | `{"query": "hello", "k": 5}`     |
| **Get vector**        | `GET /collections/{name}/vectors/{id}`    | –                                                                  |
| **Delete vector**     | `DELETE /collections/{name}/vectors/{id}` | –                                                                  |
| **Save collection**   | `POST /collections/{name}/save`           | `{"file_path": "./collection.vlc"}`                                |
| **Load collection**   | `POST /collections/load`                  | `{"file_path": "./collection.vlc", "collection_name": "restored"}` |


## Index Types

VectorLite supports 2 indexes: **Flat** and **HNSW**.

| Index    | Search Complexity | Insert   | Use Case                              |
| -------- | ----------------- | -------- | ------------------------------------- |
| **Flat** | O(n)              | O(1)     | Small datasets (<10K) or exact search |
| **HNSW** | O(log n)          | O(log n) | Larger datasets or approximate search |

See [Hierarchical Navigable Small World](https://arxiv.org/abs/1603.09320).

Note: Flat indices support all metrics dynamically. HNSW index must be created with a default distance metric (`cosine`, `euclidean`, `manhattan` or `dotproduct`). 

### Configuration profiles for HNSW

| Profile              | Features                         | Use Case                       |
| -------------------- | -------------------------------- | ------------------------------ |
| **default**          | balanced                         | general workloads              |
| **memory-optimized** | reduced precision, smaller graph | constrained devices            |
| **high-accuracy**    | higher recall, more memory       | offline re-ranking or research |

```bash
cargo build --features memory-optimized
```


### Similarity Metrics
- **Cosine**: Default for normalized embeddings, scale-invariant
- **Euclidean**: Geometric distance, sensitive to vector magnitude
- **Manhattan**: L1 norm, robust to outliers
- **Dot Product**: Raw similarity, requires consistent vector scaling

## Rust SDK Example

```rust,no_run
use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));

    client.create_collection("quotes", IndexType::HNSW, Some(SimilarityMetric::Cosine))?;
    
    let id = client.add_text_to_collection(
        "quotes", 
        "I just want to lie on the beach and eat hot dogs",
        Some(json!({
            "author": "Kevin Malone",
            "tags": ["the-office", "s3:e23"],
            "year": 2005,
        }))
    )?;

    // Metric optional - auto-detected from HNSW index
    let results = client.search_text_in_collection(
        "quotes",
        "beach games",
        3,
        None,
    )?;

    for result in &results {
        println!("ID: {}, Score: {:.4}", result.id, result.score);
    }

    Ok(())
}
```

## Testing

Run tests with mock embeddings (CI-friendly, no model files required):
```bash
cargo test --features mock-embeddings
```

Run tests with local models:
```bash
cargo test
```

### Download ML Model

This downloads the BERT-based embedding model files needed for real embedding generation:
```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
```

The model files must be present in the `./models/{model-name}/` directory with the required files:
- `config.json`
- `pytorch_model.bin` 
- `tokenizer.json`


### Using a different model

You can override the default embedding model at compile time using the `custom-model` feature:

```bash
DEFAULT_EMBEDDING_MODEL="sentence-transformers/paraphrase-MiniLM-L3-v2" cargo build --features custom-model

DEFAULT_EMBEDDING_MODEL="sentence-transformers/paraphrase-MiniLM-L3-v2" cargo run --features custom-model
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
