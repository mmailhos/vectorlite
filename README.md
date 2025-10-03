# VectorLite

Implementation of a simple vector database to learn Rust.

## Features

VectorLite supports different HNSW configurations through feature flags:

- **`fast`** (default): Balanced performance and memory usage
  - Max connections: 16
  - Max connections (layer 0): 32

- **`memory-optimized`**: Lower memory usage, slightly slower search
  - Max connections: 8
  - Max connections (layer 0): 16

- **`high-accuracy`**: Higher accuracy, more memory usage
  - Max connections: 32
  - Max connections (layer 0): 64

## Usage

```bash
# Use default (fast) configuration
cargo build

# Use memory-optimized configuration
cargo build --features memory-optimized

# Use high-accuracy configuration
cargo build --features high-accuracy
```
