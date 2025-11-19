#!/usr/bin/env python3
"""
Test script for VectorLite Python bindings.
"""

import vectorlite
import math

def test_vector_creation():
    """Test Vector creation and properties."""
    print("Testing Vector creation...")
    v = vectorlite.Vector(id=1, values=[1.0, 2.0, 3.0])
    assert v.id == 1
    assert v.values == [1.0, 2.0, 3.0]
    print(f"✓ Vector creation: {v}")

def test_search_result():
    """Test SearchResult creation and properties."""
    print("Testing SearchResult creation...")
    sr = vectorlite.SearchResult(id=1, score=0.95)
    assert sr.id == 1
    assert sr.score == 0.95
    print(f"✓ SearchResult creation: {sr}")

def test_flat_index():
    """Test FlatIndex functionality."""
    print("Testing FlatIndex...")
    
    # Create vectors
    vectors = [
        vectorlite.Vector(id=0, values=[1.0, 0.0, 0.0]),
        vectorlite.Vector(id=1, values=[0.0, 1.0, 0.0]),
        vectorlite.Vector(id=2, values=[0.0, 0.0, 1.0]),
    ]
    
    # Create index
    index = vectorlite.FlatIndexWrapper(dimension=3, vectors=vectors)
    assert len(index) == 3
    assert index.dimension() == 3
    assert not index.is_empty()
    print(f"✓ FlatIndex creation: {index}")
    
    # Test search
    query = [1.0, 0.0, 0.0]
    results = index.search(query, k=2)
    assert len(results) == 2
    assert results[0].id == 0  # Should match the first vector
    assert abs(results[0].score - 1.0) < 1e-10  # Perfect match
    print(f"✓ FlatIndex search: {results}")
    
    # Test add
    new_vector = vectorlite.Vector(id=3, values=[0.5, 0.5, 0.0])
    index.add(new_vector)
    assert len(index) == 4
    print("✓ FlatIndex add operation")
    
    # Test get_vector
    retrieved = index.get_vector(3)
    assert retrieved is not None
    assert retrieved.id == 3
    print(f"✓ FlatIndex get_vector: {retrieved}")
    
    # Test delete
    index.delete(3)
    assert len(index) == 3
    print("✓ FlatIndex delete operation")

def test_hnsw_index():
    """Test HNSWIndex functionality."""
    print("Testing HNSWIndex...")
    
    # Create vectors
    vectors = [
        vectorlite.Vector(id=0, values=[1.0, 0.0, 0.0]),
        vectorlite.Vector(id=1, values=[0.0, 1.0, 0.0]),
        vectorlite.Vector(id=2, values=[0.0, 0.0, 1.0]),
    ]
    
    # Create index
    index = vectorlite.HNSWIndexWrapper(dimension=3, vectors=vectors)
    assert len(index) == 3
    assert index.dimension() == 3
    assert not index.is_empty()
    print(f"✓ HNSWIndex creation: {index}")
    
    # Test search
    query = [1.0, 0.0, 0.0]
    results = index.search(query, k=2)
    assert len(results) == 2
    print(f"✓ HNSWIndex search: {results}")

def test_cosine_similarity():
    """Test cosine similarity function."""
    print("Testing cosine similarity...")
    
    # Test identical vectors
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    similarity = vectorlite.cosine_similarity_py(a, b)
    assert abs(similarity - 1.0) < 1e-10
    print(f"✓ Identical vectors similarity: {similarity}")
    
    # Test orthogonal vectors
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    similarity = vectorlite.cosine_similarity_py(a, b)
    assert abs(similarity - 0.0) < 1e-10
    print(f"✓ Orthogonal vectors similarity: {similarity}")
    
    # Test opposite vectors
    a = [1.0, 2.0, 3.0]
    b = [-1.0, -2.0, -3.0]
    similarity = vectorlite.cosine_similarity_py(a, b)
    assert abs(similarity - (-1.0)) < 1e-10
    print(f"✓ Opposite vectors similarity: {similarity}")

def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")
    
    default_dim = vectorlite.get_default_vector_dimension()
    assert isinstance(default_dim, int)
    assert default_dim > 0
    print(f"✓ Default vector dimension: {default_dim}")

def main():
    """Run all tests."""
    print("VectorLite Python Bindings Test Suite")
    print("=" * 50)
    
    try:
        test_vector_creation()
        test_search_result()
        test_flat_index()
        test_hnsw_index()
        test_cosine_similarity()
        test_utility_functions()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
