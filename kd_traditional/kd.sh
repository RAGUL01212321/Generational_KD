#!/usr/bin/bash
# Quick script to run KD training

if [ "$1" = "train" ]; then
    echo "Starting KD training..."
    python3 train_kd.py "${@:2}"
elif [ "$1" = "train-quick" ]; then
    echo "Starting quick KD training (1000 samples, 1 epoch)..."
    python3 train_kd.py --epochs 1 --max-samples 1000 "${@:2}"
elif [ "$1" = "train-resume" ]; then
    if [ -z "$2" ]; then
        echo "Usage: ./kd.sh train-resume <checkpoint_path>"
        exit 1
    fi
    echo "Resuming training from checkpoint: $2"
    python3 train_kd.py --resume "$2" "${@:3}"
elif [ "$1" = "help" ] || [ -z "$1" ]; then
    echo "Knowledge Distillation Training Script"
    echo ""
    echo "Usage: ./kd.sh {train|train-quick|train-resume|help}"
    echo ""
    echo "Commands:"
    echo "  train              - Start full KD training"
    echo "  train-quick        - Quick test (1000 samples, 1 epoch)"
    echo "  train-resume PATH  - Resume from checkpoint"
    echo "  help               - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./kd.sh train"
    echo "  ./kd.sh train --epochs 5 --batch-size 8"
    echo "  ./kd.sh train-quick --common-dim 512"
    echo "  ./kd.sh train-resume kd_checkpoints/step_100.pt --epochs 5"
else
    echo "Unknown command: $1"
    echo "Run './kd.sh help' for usage"
    exit 1
fi
