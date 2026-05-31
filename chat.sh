#!/bin/bash
# Simple script to chat with Apollo models

if [ "$1" = "0.5B" ]; then
    python3 Gen_KD/model_testings/apollo_0.5B/test_apollo_0.5B.py
elif [ "$1" = "1.8B" ]; then
    python3 Gen_KD/model_testings/apollo_1.8B/test_apollo_1.8B.py
else
    echo "Usage: ./chat.sh {0.5B|1.8B}"
    echo ""
    echo "Examples:"
    echo "  ./chat.sh 0.5B   - Chat with Apollo 0.5B"
    echo "  ./chat.sh 1.8B   - Chat with Apollo 1.8B"
fi
