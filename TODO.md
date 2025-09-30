# TODO

## 🎯 **Immediate Next Steps (High Impact, Low Effort)**

### **1. Make It Actually Functional**
- **Add a search demo** - Currently you only load data but never search it
- **Add CLI interface** - Let users input queries and see results
- **Display search results** - Show the actual GitHub issues that match

### **2. Add Basic Testing**
- **Unit tests** for `cosine_similarity` function
- **Integration test** with a small sample dataset
- **Benchmark** search performance

## 🚀 **Medium-term Enhancements (Good Learning Value)**

### **3. Performance Optimization**
- **Parallel search** using `rayon` crate for multi-threaded similarity calculations
- **Memory optimization** - your current approach loads everything into memory
- **SIMD optimizations** for vector operations

### **4. Better Data Structures**
- **Indexing strategies** - Currently O(n) linear search, consider:
  - **Inverted index** for text-based filtering
  - **LSH (Locality Sensitive Hashing)** for approximate search
  - **KD-trees** or **Ball trees** for spatial indexing

### **5. Enhanced Search Features**
- **Multiple similarity metrics** (Euclidean, Manhattan, etc.)
- **Filtering capabilities** (by date, repository, etc.)
- **Search result ranking** improvements

## 🏗️ **Architecture Improvements (Advanced Learning)**

### **6. Modular Design**
- **Split into modules** (`lib.rs`, separate files for different components)
- **Trait-based design** for different similarity metrics
- **Configuration system** for different embedding models

### **7. Persistence & Caching**
- **Serialize/deserialize** the vector store to disk
- **Incremental updates** instead of reloading everything
- **Memory-mapped files** for large datasets

### **8. API Development**
- **REST API** using `axum` or `warp`
- **gRPC interface** for high-performance clients
- **WebSocket** for real-time search

## 🔬 **Advanced Features (Expert Level)**

### **9. Machine Learning Integration**
- **On-the-fly embedding generation** using `candle` or `ort`
- **Fine-tuning** similarity functions
- **Clustering** and **dimensionality reduction**

### **10. Production Readiness**
- **Logging** with `tracing`
- **Metrics** and **monitoring**
- **Docker containerization**
- **Database integration** (PostgreSQL with pgvector)

## 📚 **Learning-Focused Recommendations**

**If you want to learn Rust better:**
- Focus on **#1-3** (functionality + testing + performance)
- This teaches error handling, testing, async programming, and optimization

**If you want to learn vector search:**
- Focus on **#4-5** (data structures + search features)
- This teaches algorithms, data structures, and search theory

**If you want to learn systems programming:**
- Focus on **#6-8** (architecture + persistence + APIs)
- This teaches modular design, I/O, and network programming

## 🎯 **My Top 3 Recommendations for You:**

1. **Start with #1** - Make it actually searchable and show results
2. **Then #2** - Add tests to ensure your code works correctly  
3. **Then #3** - Add parallel search to learn about Rust's concurrency

This gives you a working, tested, and performant vector search engine while learning core Rust concepts!

What aspect interests you most? Vector search algorithms, Rust performance optimization, or building a complete application?
