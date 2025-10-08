use std::collections::HashMap;
use std::sync::Arc;
use crate::{VectorIndexWrapper, VectorIndex, Vector, SearchResult, SimilarityMetric, EmbeddingFunction};

pub struct VectorLiteClient {
    collections: HashMap<String, Collection>,
    settings: Settings,
    embedding_function: Arc<dyn EmbeddingFunction>,
}

pub struct Settings {}

impl VectorLiteClient {
    pub fn new(embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            settings: Settings {},
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn new_with_settings(settings: Settings, embedding_function: Box<dyn EmbeddingFunction>) -> Self {
        Self {
            collections: HashMap::new(),
            settings,
            embedding_function: Arc::from(embedding_function),
        }
    }

    pub fn create_collection(&mut self, name: &str, index_type: IndexType) -> Result<&mut Collection, String> {
        if self.collections.contains_key(name) {
            return Err(format!("Collection '{}' already exists", name));
        }

        let dimension = self.embedding_function.dimension();
        let index = match index_type {
            IndexType::Flat => VectorIndexWrapper::Flat(crate::FlatIndex::new(dimension, Vec::new())),
            IndexType::HNSW => VectorIndexWrapper::HNSW(crate::HNSWIndex::new(dimension)),
        };

        let collection = Collection {
            name: name.to_string(),
            index,
            embedding_function: Arc::clone(&self.embedding_function),
            next_id: 1,
        };

        self.collections.insert(name.to_string(), collection);
        Ok(self.collections.get_mut(name).unwrap())
    }

    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn get_collection_mut(&mut self, name: &str) -> Option<&mut Collection> {
        self.collections.get_mut(name)
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<(), String> {
        if self.collections.remove(name).is_some() {
            Ok(())
        } else {
            Err(format!("Collection '{}' not found", name))
        }
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    pub fn add_text_to_collection(&mut self, collection_name: &str, text: &str) -> Result<u64, String> {
        if let Some(collection) = self.collections.get_mut(collection_name) {
            collection.add_text(text)
        } else {
            Err(format!("Collection '{}' not found", collection_name))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IndexType {
    Flat,
    HNSW,
}

pub struct Collection {
    name: String,
    index: VectorIndexWrapper,
    embedding_function: Arc<dyn EmbeddingFunction>,
    next_id: u64,
}

impl Collection {
    pub fn add_text(&mut self, text: &str) -> Result<u64, String> {
        let id = self.next_id;
        self.next_id += 1;
        
        let embedding = self.embedding_function.generate_embedding(text)
            .map_err(|e| e.to_string())?;
        let vector = Vector { id, values: embedding };
        self.index.add(vector)?;
        Ok(id)
    }

    pub fn add_vector(&mut self, vector: Vector) -> Result<(), String> {
        self.index.add(vector)
    }

    pub fn delete(&mut self, id: u64) -> Result<(), String> {
        self.index.delete(id)
    }

    pub fn search_text(&self, query_text: &str, k: usize, similarity_metric: SimilarityMetric) -> Result<Vec<SearchResult>, String> {
        let query_embedding = self.embedding_function.generate_embedding(query_text)
            .map_err(|e| e.to_string())?;
        Ok(self.index.search(&query_embedding, k, similarity_metric))
    }

    pub fn search_vector(&self, query_vector: &[f64], k: usize, similarity_metric: SimilarityMetric) -> Vec<SearchResult> {
        self.index.search(query_vector, k, similarity_metric)
    }

    pub fn get_vector(&self, id: u64) -> Option<&Vector> {
        self.index.get_vector(id)
    }

    pub fn count(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}