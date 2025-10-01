use clap::Parser;
use vectorlite::DEFAULT_VECTOR_DIMENSION;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    filepath: String,

    #[arg(short, long, default_value_t = DEFAULT_VECTOR_DIMENSION)]
    dimension: usize,
}

fn main() {
    let _args = Args::parse();
}