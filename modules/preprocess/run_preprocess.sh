#!/bin/bash

# Data preprocessing script
# Usage: ./run_preprocess.sh <input_dir> <output_path> [seed]

set -e  # Exit on error

# Default values
DEFAULT_SEED=42

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_path> [seed]"
    echo ""
    echo "Arguments:"
    echo "  input_dir   : Directory containing raw data files (json/jsonl)"
    echo "  output_path : Output file path for processed data (should end with .jsonl)"
    echo "  seed        : Random seed for reproducibility (default: $DEFAULT_SEED)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/raw/data /path/to/output/processed_data.jsonl 42"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_PATH="$2"
SEED="${3:-$DEFAULT_SEED}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "========================================="
echo "Data Preprocessing Script"
echo "========================================="
echo "Input directory: $INPUT_DIR"
echo "Output path: $OUTPUT_PATH"
echo "Random seed: $SEED"
echo "Project root: $PROJECT_ROOT"
echo "========================================="

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the preprocessing
echo "Starting data preprocessing..."
python -m modules.data_fetch.data_preprocess \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_PATH" \
    --seed "$SEED"

echo "========================================="
echo "Data preprocessing completed!"
echo "Output saved to: $OUTPUT_PATH"
echo "=========================================" 