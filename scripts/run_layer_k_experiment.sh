#!/bin/bash
# Experiment: Probe all layers of Llama 3.2 1B for k=1,2,3

set -e

# Configuration
MODEL="meta-llama/Llama-3.2-1B"
DATASET="data/example_dataset.jsonl"
MAX_K=3
PROBE_TYPE="linear"
NUM_EPOCHS=1
BATCH_SIZE=32
LEARNING_RATE=1e-3

# Output directories
EXPERIMENT_DIR="./experiment_results"
DATASET_DIR="${EXPERIMENT_DIR}/datasets"
PROBES_DIR="${EXPERIMENT_DIR}/probes"
RESULTS_FILE="${EXPERIMENT_DIR}/results_table.txt"

echo "========================================="
echo "Layer-K Probing Experiment"
echo "========================================="
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "K values: 1, 2, 3"
echo "Probe type: ${PROBE_TYPE}"
echo "========================================="
echo ""

# Clean up and create directories
rm -rf ${EXPERIMENT_DIR}
mkdir -p ${DATASET_DIR} ${PROBES_DIR}

# Step 1: Build activation dataset
echo "Step 1: Building activation dataset (all layers, k=1,2,3)..."
python -m look_ahead_probe.build_look_ahead_activation_dataset \
    --model_name ${MODEL} \
    --dataset_path ${DATASET} \
    --split_value train \
    --max_k ${MAX_K} \
    --max_new_tokens 30 \
    --output_path ${DATASET_DIR}/activations.pt

echo ""
echo "Step 2: Training probes for all layer-k combinations..."
echo ""

# Get number of layers from the dataset metadata
LAYERS=$(python -c "
import torch
data = torch.load('${DATASET_DIR}/activations.pt')
print(len(data['metadata']['layers']))
")

echo "Detected ${LAYERS} layers in model"
echo ""

# Train probes for each (layer, k) combination
for k in 1 2 3; do
    for layer in $(seq 0 $((LAYERS - 1))); do
        echo "Training: Layer ${layer}, k=${k}..."

        python -m look_ahead_probe.main \
            --train_dataset ${DATASET_DIR}/activations.pt \
            --layer ${layer} \
            --k ${k} \
            --probe_type ${PROBE_TYPE} \
            --num_epochs ${NUM_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --save_path ${PROBES_DIR}/layer${layer}_k${k}.pt \
            2>&1 | grep -E "Epoch|Final|Accuracy" || true

        echo ""
    done
done

echo ""
echo "Step 3: Aggregating results..."
echo ""

# Create results table
python << 'PYTHON_SCRIPT'
import torch
import sys
from pathlib import Path

results_dir = Path("./experiment_results/probes")
output_file = "./experiment_results/results_table.txt"

# Collect all results
results = {}
for probe_file in sorted(results_dir.glob("layer*_k*.pt")):
    # Parse filename: layer{N}_k{K}.pt
    parts = probe_file.stem.split('_')
    layer = int(parts[0].replace('layer', ''))
    k = int(parts[1].replace('k', ''))

    # Load results
    data = torch.load(probe_file)
    if 'results' in data and data['results']:
        acc = data['results']['accuracy']
        top5 = data['results']['top5_accuracy']
        loss = data['results']['loss']
        results[(layer, k)] = {'acc': acc, 'top5': top5, 'loss': loss}

# Get dimensions
if not results:
    print("No results found!")
    sys.exit(1)

layers = sorted(set(layer for layer, k in results.keys()))
ks = sorted(set(k for layer, k in results.keys()))

# Print table
with open(output_file, 'w') as f:
    # Header
    line = "=" * 80
    f.write(line + "\n")
    f.write("Layer-K Probing Results: Llama 3.2 1B (MLP Probe)\n")
    f.write(line + "\n\n")

    # Accuracy table
    f.write("Top-1 Accuracy:\n")
    f.write("-" * 80 + "\n")
    header = f"{'Layer':<8}"
    for k in ks:
        header += f"k={k:<12}"
    f.write(header + "\n")
    f.write("-" * 80 + "\n")

    for layer in layers:
        row = f"{layer:<8}"
        for k in ks:
            if (layer, k) in results:
                acc = results[(layer, k)]['acc']
                row += f"{acc:>6.2%}      "
            else:
                row += "N/A         "
        f.write(row + "\n")

    f.write("\n")

    # Top-5 Accuracy table
    f.write("Top-5 Accuracy:\n")
    f.write("-" * 80 + "\n")
    f.write(header + "\n")
    f.write("-" * 80 + "\n")

    for layer in layers:
        row = f"{layer:<8}"
        for k in ks:
            if (layer, k) in results:
                top5 = results[(layer, k)]['top5']
                row += f"{top5:>6.2%}      "
            else:
                row += "N/A         "
        f.write(row + "\n")

    f.write("\n")

    # Loss table
    f.write("Final Loss:\n")
    f.write("-" * 80 + "\n")
    f.write(header + "\n")
    f.write("-" * 80 + "\n")

    for layer in layers:
        row = f"{layer:<8}"
        for k in ks:
            if (layer, k) in results:
                loss = results[(layer, k)]['loss']
                row += f"{loss:>6.4f}      "
            else:
                row += "N/A         "
        f.write(row + "\n")

    f.write("\n" + line + "\n")

    # Summary statistics
    f.write("\nSummary Statistics:\n")
    f.write("-" * 80 + "\n")

    for k in ks:
        k_results = [results[(l, k)]['acc'] for l in layers if (l, k) in results]
        if k_results:
            best_layer = max(layers, key=lambda l: results[(l, k)]['acc'] if (l, k) in results else 0)
            best_acc = results[(best_layer, k)]['acc']
            avg_acc = sum(k_results) / len(k_results)
            f.write(f"k={k}: Best layer={best_layer} ({best_acc:.2%}), Average={avg_acc:.2%}\n")

    f.write(line + "\n")

# Also print to console
with open(output_file, 'r') as f:
    print(f.read())

print(f"\nResults saved to: {output_file}")
PYTHON_SCRIPT

echo ""
echo "========================================="
echo "Experiment complete!"
echo "Results saved to: ${RESULTS_FILE}"
echo "========================================="
