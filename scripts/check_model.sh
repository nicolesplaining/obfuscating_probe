#!/bin/bash
# Check model compatibility for activation extraction
# Run from project root: bash scripts/check_model.sh

python -m look_ahead_probe.check_model \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --max_new_tokens 10
