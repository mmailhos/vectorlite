#!/bin/bash


set -e

echo "🚀 Installing VectorLite Python bindings..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if maturin is available
if ! command -v maturin &> /dev/null; then
    echo "📦 Installing maturin..."
    python3 -m pip install maturin
fi

# Build and install the package
echo "🔨 Building and installing vectorlite..."
maturin develop --release

echo "✅ VectorLite Python bindings installed successfully!"
echo ""
echo "You can now use vectorlite in Python:"
echo "  import vectorlite"
echo "  index = vectorlite.FlatIndexWrapper(dimension=3)"
echo ""
echo "Run 'python example.py' to see a demonstration."
