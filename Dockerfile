FROM rust:1.81-slim AS builder

# Install nightly toolchain for edition 2024 support
RUN rustup toolchain install nightly && rustup default nightly

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ARG FEATURES=""

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies first (this layer will be cached if Cargo.toml doesn't change)
RUN cargo build --release --features "${FEATURES}" && rm -rf src

COPY src ./src

RUN cargo build --release --features "${FEATURES}"

FROM python:3.11-slim AS model-downloader

ARG MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

RUN pip install --no-cache-dir huggingface_hub[cli]

WORKDIR /app

RUN mkdir -p models

RUN huggingface-cli download ${MODEL_NAME} \
    --local-dir models/$(basename ${MODEL_NAME}) \
    --local-dir-use-symlinks False

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r vectorlite && useradd -r -g vectorlite vectorlite

WORKDIR /app

COPY --from=builder /app/target/release/vectorlite /usr/local/bin/vectorlite
COPY --from=model-downloader /app/models /app/models

RUN chown -R vectorlite:vectorlite /app
USER vectorlite

EXPOSE 3001

ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

CMD ["/usr/local/bin/vectorlite", "--host", "0.0.0.0", "--port", "3001"]