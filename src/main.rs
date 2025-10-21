use clap::Parser;
use vectorlite::{VectorLiteClient, EmbeddingGenerator, start_server};
use std::path::Path;
use tracing::info;

#[derive(Parser, Debug)]
#[command(
    version, 
    about = "VectorLite - A high-performance, in-memory vector database optimized for AI agent workloads",
    long_about = None
)]
struct Args {
    #[arg(short, long)]
    filepath: Option<String>,

    #[arg(short, long, default_value_t = 3001)]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    info!("Starting VectorLite server...");
    info!("Host: {}", args.host);
    info!("Port: {}", args.port);
    
    // Create a new client with embedding generator
    let mut client = VectorLiteClient::new(Box::new(EmbeddingGenerator::new()?));
    
    // Load VLC file if provided
    if let Some(filepath) = &args.filepath {
        info!("Loading collection from VLC file: {}", filepath);
        let collection = vectorlite::load_collection_from_file(Path::new(filepath))?;
        let collection_name = collection.name().to_string();
        
        client.add_collection(collection)?;
        info!("Successfully loaded collection '{}' from {}", collection_name, filepath);
    } else {
        info!("Starting with empty server - no collections loaded");
    }
    
    start_server(client, &args.host, args.port).await?;
    
    Ok(())
}