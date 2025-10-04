# VectorLite Python Bindings

High-performance vector similarity search library with Python bindings.

## Features

- **Multiple Index Types**: Flat search and HNSW (Hierarchical Navigable Small World) indexes
- **High Performance**: Rust-powered with Python bindings via PyO3
- **Easy to Use**: Simple Python API for vector operations
- **Flexible**: Support for custom vector dimensions and similarity metrics

## Installation

### Development Installation

To install from source for development:

```bash
# Install maturin (if not already installed)
pip install maturin

# Build and install in development mode
cd bindings
maturin develop

# Or build in release mode for better performance
maturin develop --release
```

### Production Installation

```bash
# Install from PyPI (when published)
pip install vectorlite
```

## Quick Start

```python
import vectorlite

# Create some vectors
vectors = [
    vectorlite.Vector(id=0, values=[1.0, 0.0, 0.0]),
    vectorlite.Vector(id=1, values=[0.0, 1.0, 0.0]),
    vectorlite.Vector(id=2, values=[0.0, 0.0, 1.0]),
]

# Create a flat index
index = vectorlite.FlatIndexWrapper(dimension=3, vectors=vectors)

# Search for similar vectors
query = [0.5, 0.5, 0.0]
results = index.search(query, k=2)
for result in results:
    print(f"ID: {result.id}, Score: {result.score}")

# Create an HNSW index for better performance on large datasets
hnsw_index = vectorlite.HNSWIndexWrapper(dimension=3, vectors=vectors)
results = hnsw_index.search(query, k=2)
```

## API Reference

### Classes

#### `Vector`
Represents a vector with an ID and values.

```python
vector = vectorlite.Vector(id=1, values=[1.0, 2.0, 3.0])
print(vector.id)      # 1
print(vector.values)  # [1.0, 2.0, 3.0]
```

#### `SearchResult`
Represents a search result with an ID and similarity score.

```python
result = vectorlite.SearchResult(id=1, score=0.95)
print(result.id)    # 1
print(result.score) # 0.95
```

#### `FlatIndexWrapper`
A flat (brute force) vector index. Good for small datasets.

```python
index = vectorlite.FlatIndexWrapper(dimension=3, vectors=None)
index.add(vectorlite.Vector(id=1, values=[1.0, 2.0, 3.0]))
results = index.search([1.0, 2.0, 3.0], k=5)
```

#### `HNSWIndexWrapper`
A Hierarchical Navigable Small World index. Good for large datasets.

```python
index = vectorlite.HNSWIndexWrapper(dimension=3, vectors=None)
index.add(vectorlite.Vector(id=1, values=[1.0, 2.0, 3.0]))
results = index.search([1.0, 2.0, 3.0], k=5)
```

### Functions

#### `cosine_similarity_py(a, b)`
Calculate cosine similarity between two vectors.

```python
similarity = vectorlite.cosine_similarity_py([1.0, 0.0], [0.0, 1.0])
print(similarity)  # 0.0 (orthogonal vectors)
```

#### `get_default_vector_dimension()`
Get the default vector dimension used by the library.

```python
dim = vectorlite.get_default_vector_dimension()
print(dim)  # 768
```

## Examples

See `example.py` for a comprehensive example of using the library.

## Testing

Run the test suite:

```bash
python test_bindings.py
```

## Performance

The library is built with Rust for high performance. For best results:

- Use `maturin develop --release` for development
- Use HNSW indexes for large datasets (>1000 vectors)
- Use flat indexes for small datasets or when you need exact results

## License

MIT License - see the main project README for details.
