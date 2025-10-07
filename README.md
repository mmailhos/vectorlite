# VectorLite

A lightweight vector database implementation in Rust for learning and experimentation.

## Overview

VectorLite provides multiple indexing strategies for vector similarity search, including flat indexing and [HNSW (Hierarchical Navigable Small World)](https://arxiv.org/abs/1603.09320) graphs. It's designed as an educational project to explore vector database concepts and Rust implementation patterns.

## Features

- **Two Index Types**: Simple flat indexing and advanced HNSW indexing
- **Configurable HNSW**: Three performance profiles for different memory/accuracy tradeoffs
- **Cosine Similarity**: Built-in cosine similarity calculation
- **Type Safety**: Leverages Rust's type system for safe vector operations
- **Comprehensive Testing**: Full test coverage with integration tests

## Index Types

### FlatIndex
- **Best for**: Small to medium datasets, memory-constrained environments
- **Performance**: O(n) search, O(1) memory access patterns
- **Use case**: Datasets with < 100K vectors where simplicity is preferred

### HNSWIndex
- **Best for**: Large-scale approximate nearest neighbor search
- **Performance**: O(log n) search complexity
- **Use case**: Production vector search with configurable accuracy/memory tradeoffs

## HNSW Configuration

VectorLite supports three HNSW performance profiles via feature flags:

| Feature | Max Connections | Layer 0 Connections | Use Case |
|---------|----------------|---------------------|----------|
| `fast` (default) | 16 | 32 | Balanced performance and memory |
| `memory-optimized` | 8 | 16 | Lower memory usage, slightly slower |
| `high-accuracy` | 32 | 64 | Higher accuracy, more memory usage |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vectorlite = "0.1.0"
```

Basic usage:

```rust
use vectorlite::{HNSWIndex, Vector, VectorIndex, EmbeddingGenerator};

// Create an embedding generator (uses all-MiniLM-L6-v2 by default)
let embedder = EmbeddingGenerator::new()?;

// Create an index for 384-dimensional vectors
let mut index = HNSWIndex::new(embedder.dimension());

// Generate embeddings and add to index
let text = "Rust is a systems programming language";
let embedding = embedder.generate_embedding(text)?;
let vector = Vector {
    id: 1,
    values: embedding,
};
index.add(vector)?;

// Search for similar vectors
let query_text = "programming in Rust";
let query_embedding = embedder.generate_embedding(query_text)?;
let results = index.search(&query_embedding, 10);
```

## Loading Models

VectorLite uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by default:

```rust
let embedder = EmbeddingGenerator::new()?; 
```

For custom models, place the required files (`config.json`, `tokenizer.json`, `pytorch_model.bin`) in a directory and specify the path:

```rust
let embedder = EmbeddingGenerator::new_from_path("./models/my-custom-model")?;
```

## Building with Different Configurations

```bash
# Default (fast) configuration
cargo build

# Memory-optimized configuration
cargo build --features memory-optimized

# High-accuracy configuration  
cargo build --features high-accuracy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.