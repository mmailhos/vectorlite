#!/usr/bin/env python3
"""
Example usage of the VectorLite Python bindings.
"""

import vectorlite
import random

def main():
    print("VectorLite Python Bindings Example")
    print("=" * 40)
    
    # Create some sample vectors
    dimension = 3
    vectors = []
    for i in range(5):
        values = [random.random() for _ in range(dimension)]
        vectors.append(vectorlite.Vector(id=i, values=values))
        print(f"Created vector {i}: {vectors[-1]}")
    
    print("\n" + "=" * 40)
    print("Testing FlatIndex")
    print("=" * 40)
    
    # Create a flat index
    flat_index = vectorlite.FlatIndexWrapper(dimension=dimension, vectors=vectors)
    print(f"Created flat index: {flat_index}")
    print(f"Index length: {len(flat_index)}")
    print(f"Index dimension: {flat_index.dimension()}")
    
    # Search for similar vectors
    query = [0.5, 0.5, 0.5]
    print(f"\nSearching for vectors similar to {query}")
    results = flat_index.search(query, k=3)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result}")
    
    print("\n" + "=" * 40)
    print("Testing HNSWIndex")
    print("=" * 40)
    
    # Create an HNSW index
    hnsw_index = vectorlite.HNSWIndexWrapper(dimension=dimension, vectors=vectors)
    print(f"Created HNSW index: {hnsw_index}")
    print(f"Index length: {len(hnsw_index)}")
    print(f"Index dimension: {hnsw_index.dimension()}")
    
    # Search for similar vectors
    print(f"\nSearching for vectors similar to {query}")
    results = hnsw_index.search(query, k=3)
    for i, result in enumerate(results):
        print(f"  {i+1}. {result}")
    
    print("\n" + "=" * 40)
    print("Testing utility functions")
    print("=" * 40)
    
    # Test cosine similarity
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    similarity = vectorlite.cosine_similarity_py(a, b)
    print(f"Cosine similarity between {a} and {b}: {similarity:.6f}")
    
    # Test identical vectors
    c = [1.0, 2.0, 3.0]
    d = [1.0, 2.0, 3.0]
    similarity = vectorlite.cosine_similarity_py(c, d)
    print(f"Cosine similarity between {c} and {d}: {similarity:.6f}")
    
    # Test default dimension
    default_dim = vectorlite.get_default_vector_dimension()
    print(f"Default vector dimension: {default_dim}")
    
    print("\n" + "=" * 40)
    print("Testing vector operations")
    print("=" * 40)
    
    # Test adding a new vector
    new_vector = vectorlite.Vector(id=10, values=[0.1, 0.2, 0.3])
    print(f"Adding new vector: {new_vector}")
    flat_index.add(new_vector)
    print(f"Index length after adding: {len(flat_index)}")
    
    # Test getting a vector by ID
    retrieved = flat_index.get_vector(10)
    if retrieved:
        print(f"Retrieved vector: {retrieved}")
    
    # Test deleting a vector
    print("Deleting vector with ID 0")
    flat_index.delete(0)
    print(f"Index length after deletion: {len(flat_index)}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
