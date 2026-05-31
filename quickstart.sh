#!/bin/bash
# Quick Start Script for Gen_KD Local Training
# Run this script to quickly test and train the models

set -e

PROJECT_ROOT="/home/ror-technologies/Hikigai/Gen_KD"
cd "$PROJECT_ROOT"

echo "==========================================="
echo "  GEN_KD - LOCAL TRAINING QUICK START"
echo "==========================================="
echo ""

# Function to print usage
usage() {
    echo "Usage: ./quickstart.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test      - Run test suite (recommended first)"
    echo "  dry-run   - Run training with 1 batch per generation"
    echo "  train     - Run full training with defaults"
    echo "  download  - Pre-download models"
    echo "  setup     - Run setup validation"
    echo "  help      - Show this message"
    echo ""
    echo "Examples:"
    echo "  ./quickstart.sh test"
    echo "  ./quickstart.sh dry-run"
    echo "  ./quickstart.sh train"
}

# Get command
COMMAND=${1:-help}

case "$COMMAND" in
    test)
        echo "Running test suite..."
        echo ""
        python3 test_local.py
        ;;
    
    dry-run)
        echo "Running dry-run (1 batch per generation)..."
        echo ""
        python -m Gen_KD.train --dry-run
        ;;
    
    train)
        echo "Starting full training..."
        echo "This will train with default parameters:"
        echo "  - Models: 3x tiny-gpt2"
        echo "  - Epochs: 3"
        echo "  - Batch size: 8"
        echo "  - Learning rate: 5e-5"
        echo ""
        python -m Gen_KD.train
        ;;
    
    setup)
        echo "Running setup validation..."
        echo ""
        python3 setup_local.py
        ;;
    
    download)
        echo "Downloading models for offline use..."
        echo ""
        python download_models.py --category tiny
        echo ""
        echo "Downloaded tiny models. Use --help for more options:"
        python download_models.py --help
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        usage
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "Done!"
echo "==========================================="
