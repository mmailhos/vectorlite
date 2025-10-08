# VectorLite

A lightweight vector database with built-in embeddings generation, search and retrieval capabilities for AI agents

## Features

- **Two Index Types**: Flat and HNSW indexing
- **ChromaDB-like API**: Simple client interface with collections
- **Multiple Similarity Metrics**: Cosine, Euclidean, Manhattan, Dot Product
- **Built-in Embeddings**: Uses all-MiniLM-L6-v2 model
- **Memory-only**: No persistence, all data in RAM

## Quick Start

```rust
use vectorlite::{VectorLiteClient, EmbeddingGenerator, IndexType, SimilarityMetric};

// Create client with embedding function
let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));

// Create collection
client.create_collection("documents", IndexType::Flat)?;

// Add text (auto-generates embedding and ID)
let id = client.add_text_to_collection("documents", "Hello world")?;

// Search
let collection = client.get_collection("documents").unwrap();
let results = collection.search_text("hello", 5, SimilarityMetric::Cosine)?;
```

## Index Types

- **FlatIndex**: O(n) search, good for small datasets
- **HNSWIndex**: O(log n) search, good for large datasets

## Similarity Metrics

- `SimilarityMetric::Cosine` - Cosine similarity (default)
- `SimilarityMetric::Euclidean` - Euclidean distance
- `SimilarityMetric::Manhattan` - Manhattan distance  
- `SimilarityMetric::DotProduct` - Dot product

## HNSW Configuration

Build with different performance profiles:

```bash
cargo build                           # Default (balanced)
cargo build --features memory-optimized  # Lower memory
cargo build --features high-accuracy     # Higher accuracy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.