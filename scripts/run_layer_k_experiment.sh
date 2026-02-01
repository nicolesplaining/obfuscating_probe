#!/bin/bash
# Quick test of layer-k experiment pipeline
# Run from project root: bash scripts/run_layer_k_experiment.sh

set -e

echo "Running layer-k experiment (test mode with small parameters)..."
echo ""

python scripts/layer_k_experiment.py \
    --model_name meta-llama/Llama-3.2-1B \
    --train_dataset_path data/example_dataset.jsonl \
    --val_dataset_path data/example_val.jsonl \
    --max_k 1 \
    --max_prompts 10 \
    --max_new_tokens 20 \
    --probe_type linear \
    --num_epochs 1 \
    --batch_size 32 \
    --output_dir experiment_results

echo ""
echo "✓ Pipeline test complete! Check experiment_results/ for outputs"
