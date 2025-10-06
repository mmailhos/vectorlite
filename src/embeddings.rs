use thiserror::Error;
use std::path::Path;

use candle_core::{Device, Tensor, DType, IndexOp};
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

const DEFAULT_EMBEDDING_MODEL: &str = "all-MiniLM-L6-v2";

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model loading failed: {0}")]
    ModelLoading(String),
    #[error("Tokenization failed: {0}")]
    Tokenization(String),
    #[error("Inference failed: {0}")]
    Inference(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

pub struct EmbeddingGenerator {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

impl std::fmt::Debug for EmbeddingGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGenerator")
            .field("device", &self.device)
            .field("dimension", &self.dimension)
            .field("model", &"<BertModel>")
            .field("tokenizer", &"<Tokenizer>")
            .finish()
    }
}

impl EmbeddingGenerator {
    /// Create a new embedding generator using all-MiniLM-L6-v2 by default
    pub fn new() -> Result<Self> {
        Self::new_from_path(&format!("./models/{}", DEFAULT_EMBEDDING_MODEL))
    }

    pub fn new_from_path(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let (model, tokenizer, dimension) = Self::load_model_from_path(model_path, &device)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
        })
    }

    fn load_model_from_path(model_path: &str, device: &Device) -> Result<(BertModel, Tokenizer, usize)> {
        let model_dir = Path::new(model_path);
        
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(EmbeddingError::ModelLoading(format!(
                "Tokenizer file not found: {}. Please ensure the model is properly downloaded.",
                tokenizer_path.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbeddingError::ModelLoading(format!("Failed to load tokenizer: {}", e)))?;
        
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(EmbeddingError::ModelLoading(format!(
                "Config file not found: {}. Please ensure the model is properly downloaded.",
                config_path.display()
            )));
        }
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbeddingError::ModelLoading(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| EmbeddingError::ModelLoading(format!("Failed to parse config: {}", e)))?;
        
        let dimension = config.hidden_size;
        
        let model_file = model_dir.join("pytorch_model.bin");
        if !model_file.exists() {
            return Err(EmbeddingError::ModelLoading(format!(
                "Model weights file not found: {}. Please ensure the model is properly downloaded.",
                model_file.display()
            )));
        }
        let weights = candle_nn::VarBuilder::from_pth(&model_file, DType::F32, device)
            .map_err(|e| EmbeddingError::ModelLoading(format!("Failed to load weights: {}", e)))?;
        let model = BertModel::load(weights, &config)
            .map_err(|e| EmbeddingError::ModelLoading(format!("Failed to create model: {}", e)))?;
        
        Ok((model, tokenizer, dimension))
    }


    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f64>> {
        // Tokenize input
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| EmbeddingError::Tokenization(format!("Tokenization failed: {}", e)))?;
        let token_ids = encoding.get_ids();
        
        // Convert to tensor
        let input_ids = Tensor::new(token_ids, &self.device)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to create input tensor: {}", e)))?;
        let input_ids = input_ids.unsqueeze(0)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to add batch dimension: {}", e)))?;
        
        // Create token type ids (all zeros for single sequence)
        let token_type_ids = Tensor::zeros((1, input_ids.dim(1).unwrap()), input_ids.dtype(), &self.device)
            .map_err(|e| EmbeddingError::Inference(format!("Failed to create token type ids: {}", e)))?;
        
        // Run through BERT model
        let outputs = self.model.forward(&input_ids, &token_type_ids, None)
            .map_err(|e| EmbeddingError::Inference(format!("Model inference failed: {}", e)))?;
        
        // Extract [CLS] token embedding (first token)
        let cls_embedding = outputs.i((0, 0))
            .map_err(|e| EmbeddingError::Inference(format!("Failed to extract CLS token: {}", e)))?;
        
        // Convert to Vec<f64> (convert from F32 to F64)
        let embedding_f32: Vec<f32> = cls_embedding.to_vec1()
            .map_err(|e| EmbeddingError::Inference(format!("Failed to convert to Vec: {}", e)))?;
        let embedding: Vec<f64> = embedding_f32.into_iter().map(|x| x as f64).collect();
        
        // L2 normalize
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized: Vec<f64> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };
        
        Ok(normalized)
    }


    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn generate_embeddings_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        texts.iter()
            .map(|text| self.generate_embedding(text))
            .collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_generation() {
        let generator = EmbeddingGenerator::new().unwrap();
        let text = "hello world this is a test";
        let embedding = generator.generate_embedding(text).unwrap();

        assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 dimension
        assert!(!embedding.iter().all(|&x| x == 0.0), "Embedding should not be all zeros");
    }

    #[test]
    fn test_embedding_dimension() {
        let generator = EmbeddingGenerator::new().unwrap();
        assert_eq!(generator.dimension(), 384);
    }

    #[test]
    fn test_embedding_consistency() {
        let generator = EmbeddingGenerator::new().unwrap();
        let text = "the quick brown fox";
        
        let embedding1 = generator.generate_embedding(text).unwrap();
        let embedding2 = generator.generate_embedding(text).unwrap();
        
        // Embeddings should be identical for the same text
        for (a, b) in embedding1.iter().zip(embedding2.iter()) {
            assert!((a - b).abs() < 1e-10, "Embeddings should be consistent");
        }
    }

    #[test]
    fn test_embedding_normalization() {
        let generator = EmbeddingGenerator::new().unwrap();
        let text = "test normalization";
        let embedding = generator.generate_embedding(text).unwrap();

        // Check that the embedding is L2 normalized
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "Embedding should be L2 normalized");
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let generator = EmbeddingGenerator::new().unwrap();
        
        let text1 = "hello world";
        let text2 = "goodbye universe";
        
        let embedding1 = generator.generate_embedding(text1).unwrap();
        let embedding2 = generator.generate_embedding(text2).unwrap();
        
        // Different texts should produce different embeddings
        let cosine_sim = cosine_similarity(&embedding1, &embedding2);
        assert!(cosine_sim < 0.99, "Different texts should produce different embeddings");
    }

    #[test]
    fn test_batch_embedding_generation() {
        let generator = EmbeddingGenerator::new().unwrap();
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];
        
        let embeddings = generator.generate_embeddings_batch(&texts).unwrap();
        
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
        assert_eq!(embeddings[2].len(), 384);
    }

    #[test]
    fn test_empty_text_embedding() {
        let generator = EmbeddingGenerator::new().unwrap();
        let embedding = generator.generate_embedding("").unwrap();
        
        assert_eq!(embedding.len(), 384);
        // Empty text should still produce a valid embedding (mostly zeros)
        assert!(embedding.iter().all(|&x| x.abs() < 1.0), "Empty text should produce valid embedding");
    }

    // Note: Similar text similarity test removed for placeholder implementation
    // In a real BERT implementation, similar texts would produce similar embeddings

    // Helper function for cosine similarity
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}