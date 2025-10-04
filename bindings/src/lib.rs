use pyo3::prelude::*;
use ::vectorlite::{Vector as RustVector, VectorIndex, FlatIndex, HNSWIndex, cosine_similarity, DEFAULT_VECTOR_DIMENSION};

/// Python wrapper for Vector
#[pyclass]
#[derive(Clone)]
pub struct Vector {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pymethods]
impl Vector {
    #[new]
    fn new(id: u64, values: Vec<f64>) -> Self {
        Self { id, values }
    }

    fn __repr__(&self) -> String {
        format!("Vector(id={}, values=[{}])", self.id, 
                self.values.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>().join(", "))
    }
}

/// Python wrapper for SearchResult
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub id: u64,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl SearchResult {
    #[new]
    fn new(id: u64, score: f64) -> Self {
        Self { id, score }
    }

    fn __repr__(&self) -> String {
        format!("SearchResult(id={}, score={:.6})", self.id, self.score)
    }
}

/// Python wrapper for FlatIndex
#[pyclass]
pub struct FlatIndexWrapper {
    inner: FlatIndex,
}

#[pymethods]
impl FlatIndexWrapper {
    #[new]
    fn new(dimension: usize, vectors: Option<Vec<Vector>>) -> PyResult<Self> {
        let rust_vectors: Vec<RustVector> = vectors
            .unwrap_or_default()
            .into_iter()
            .map(|v| RustVector { id: v.id, values: v.values })
            .collect();
        
        Ok(Self {
            inner: FlatIndex::new(dimension, rust_vectors),
        })
    }

    fn add(&mut self, vector: Vector) -> PyResult<()> {
        let rust_vector = RustVector { id: vector.id, values: vector.values };
        self.inner.add(rust_vector).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(())
    }

    fn delete(&mut self, id: u64) -> PyResult<()> {
        self.inner.delete(id).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(())
    }

    fn search(&self, query: Vec<f64>, k: usize) -> Vec<SearchResult> {
        self.inner.search(&query, k)
            .into_iter()
            .map(|r| SearchResult { id: r.id, score: r.score })
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn get_vector(&self, id: u64) -> Option<Vector> {
        self.inner.get_vector(id).map(|v| Vector { id: v.id, values: v.values.clone() })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("FlatIndex(dimension={}, len={})", self.inner.dimension(), self.inner.len())
    }
}

/// Python wrapper for HNSWIndex
#[pyclass]
pub struct HNSWIndexWrapper {
    inner: HNSWIndex,
}

#[pymethods]
impl HNSWIndexWrapper {
    #[new]
    fn new(dimension: usize, vectors: Option<Vec<Vector>>) -> PyResult<Self> {
        let rust_vectors: Vec<RustVector> = vectors
            .unwrap_or_default()
            .into_iter()
            .map(|v| RustVector { id: v.id, values: v.values })
            .collect();
        
        let mut index = HNSWIndex::new(dimension);
        for vector in rust_vectors {
            index.add(vector).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        }
        Ok(Self { inner: index })
    }

    fn add(&mut self, vector: Vector) -> PyResult<()> {
        let rust_vector = RustVector { id: vector.id, values: vector.values };
        self.inner.add(rust_vector).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(())
    }

    fn delete(&mut self, id: u64) -> PyResult<()> {
        self.inner.delete(id).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(())
    }

    fn search(&self, query: Vec<f64>, k: usize) -> Vec<SearchResult> {
        self.inner.search(&query, k)
            .into_iter()
            .map(|r| SearchResult { id: r.id, score: r.score })
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn get_vector(&self, id: u64) -> Option<Vector> {
        self.inner.get_vector(id).map(|v| Vector { id: v.id, values: v.values.clone() })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("HNSWIndex(dimension={}, len={})", self.inner.dimension(), self.inner.len())
    }
}

/// Calculate cosine similarity between two vectors
#[pyfunction]
fn cosine_similarity_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    Ok(cosine_similarity(&a, &b))
}

/// Get the default vector dimension
#[pyfunction]
fn get_default_vector_dimension() -> usize {
    DEFAULT_VECTOR_DIMENSION
}

/// A Python module implemented in Rust.
#[pymodule]
fn vectorlite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<FlatIndexWrapper>()?;
    m.add_class::<HNSWIndexWrapper>()?;
    m.add_function(wrap_pyfunction!(cosine_similarity_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_vector_dimension, m)?)?;
    Ok(())
}
