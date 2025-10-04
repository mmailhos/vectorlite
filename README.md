# VectorLite

A lightweight vector database implementation in Rust for learning and experimentation.

## Overview

VectorLite provides multiple indexing strategies for vector similarity search, including flat indexing, hash-based indexing, and [HNSW (Hierarchical Navigable Small World)](https://arxiv.org/abs/1603.09320) graphs. It's designed as an educational project to explore vector database concepts and Rust implementation patterns.

## Features

- **Multiple Index Types**: Choose the right indexing strategy for your use case
- **Configurable HNSW**: Three performance profiles for different memory/accuracy tradeoffs
- **Cosine Similarity**: Built-in cosine similarity calculation
- **Type Safety**: Leverages Rust's type system for safe vector operations
- **Comprehensive Testing**: Full test coverage with integration tests

## Index Types

### FlatIndex
- **Best for**: Search-heavy workloads, memory-constrained environments
- **Performance**: O(n) search, O(1) memory access patterns
- **Use case**: Static datasets where search is the primary operation

### HashIndex  
- **Best for**: Dynamic datasets with frequent ID-based operations
- **Performance**: O(1) ID lookups, O(n) search
- **Use case**: Applications requiring frequent vector updates or deletions

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
use vectorlite::{HNSWIndex, Vector, VectorIndex};

// Create an index for 768-dimensional vectors
let mut index = HNSWIndex::new(768);

// Add vectors
let vector = Vector {
    id: 1,
    values: vec![0.1, 0.2, 0.3, /* ... 768 dimensions */],
};
index.add(vector)?;

// Search for similar vectors
let query = vec![0.1, 0.2, 0.3, /* ... 768 dimensions */];
let results = index.search(&query, 10); // Find 10 most similar vectors
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