use clap::Parser;
use vectorlite::{load_test_dataset, DEFAULT_EMBEDDING_DIMENSION};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    filepath: String,

    #[arg(short, long, default_value_t = DEFAULT_EMBEDDING_DIMENSION)]
    dimension: usize,
}

fn main() {
    let args = Args::parse();
    let vector_store = load_test_dataset(args.filepath.as_str(), args.dimension).expect("Failed to load dataset");
    println!("Loaded {} embeddings", vector_store.data.len());
}