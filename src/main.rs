use std::fs::File;
use std::io::{ BufReader, BufRead, Error};

#[derive(serde::Deserialize)]
struct Issue {
    html_url: String,
    title: String,
    comments: String,
    body: String,
    comment_length: u32,
    text: String,
    embeddings: Vec<f64>,
}

fn load_test_dataset(path: String) -> Result<Vec<Vec<f64>>, Error> {
    let mut db: Vec<Vec<f64>> = Vec::new();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let issue = serde_json::from_str::<Issue>(&line?)?;
        db.push(issue.embeddings);
    }
    return Ok(db);
}

fn main() {
    let db = load_test_dataset("dataset/github-embeddings-doy/issues-datasets-embedded.jsonl".to_string()).unwrap();
    println!("{}", db.len());
}
