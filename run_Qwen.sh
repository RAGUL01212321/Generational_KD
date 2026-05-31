#!/bin/bash
# Qwen Model Testing Quick Start

cd "$(dirname "$0")"

echo "=================================="
echo "  QWEN MODEL TESTING SUITE"
echo "=================================="
echo ""

case "${1:-help}" in
    test-0.5B)
        echo "Testing Qwen 0.5B model..."
        python3 apollo_testing/test_models.py --model qwen_0.5B
        ;;
    
    test-1.8B)
        echo "Testing Qwen 1.8B model..."
        python3 apollo_testing/test_models.py --model qwen_1.8B
        ;;
    
    test-all)
        echo "Testing all Qwen models..."
        python3 apollo_testing/test_models.py --all
        ;;
    
    infer-0.5B)
        echo "Starting interactive inference with Qwen 0.5B..."
        python3 apollo_testing/inference.py --model qwen_0.5B
        ;;
    
    infer-1.8B)
        echo "Starting interactive inference with Qwen 1.8B..."
        python3 apollo_testing/inference.py --model qwen_1.8B
        ;;
    
    list)
        echo "Available Apollo models:"
        python3 apollo_testing/test_models.py --list
        ;;
    
    help|--help|-h)
        cat << 'EOF'
Qwen Model Testing Suite - Quick Start

Usage: ./run_apollo.sh [COMMAND]

Commands:
  test-0.5B     Test Qwen 0.5B model
  test-1.8B     Test Qwen 1.8B model
  test-all      Test all available models
  infer-0.5B    Interactive inference with Qwen 0.5B
  infer-1.8B    Interactive inference with Qwen 1.8B
  list          List available models
  help          Show this help message

Examples:
  # Test model loading and info
  ./run_apollo.sh test-0.5B
  ./run_apollo.sh test-1.8B
  
  # Start interactive session
  ./run_apollo.sh infer-0.5B
  ./run_apollo.sh infer-1.8B

Documentation:
  See apollo_testing/README.md for detailed information

EOF
        ;;
    
    *)
        echo "Unknown command: $1"
        echo "Use './run_apollo.sh help' for usage information"
        exit 1
        ;;
esac
