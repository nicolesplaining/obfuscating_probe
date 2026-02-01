# Scripts

Helper scripts for experiments and testing.

## Available Scripts

### `layer_k_experiment.py` ⭐
**Purpose:** End-to-end layer-k probing experiment pipeline

**What it does:**
Orchestrates the 3-step pipeline:
1. **Check model** - Verify compatibility (`check_model.py`)
2. **Build datasets** - Extract train (and optional val) activations
3. **Train & evaluate** - For all layers and k values (`train_all_layers.py`)
   - Automatically evaluates on validation set if provided
   - Saves results with train/val metrics to JSON

**Usage:**
```bash
# With validation set
python scripts/layer_k_experiment.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --train_dataset_path data/train.jsonl \
    --val_dataset_path data/val.jsonl \
    --max_k 5 \
    --probe_type mlp \
    --num_epochs 10

# Training only (no validation)
python scripts/layer_k_experiment.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --train_dataset_path data/example_dataset.jsonl \
    --max_k 3 \
    --skip_check
```

**JSONL format** (no split field needed):
```jsonl
{"text": "Your prompt text here"}
{"text": "Another prompt"}
```

**Output:**
```
experiment_results/
├── activations.pt              # Training activations
├── val_activations.pt          # Validation activations (if provided)
├── probes/                     # Trained probes
│   ├── k1/
│   ├── k2/
│   └── k3/
└── experiment_results.json     # Train & val results ⭐
```

**Key arguments:**
- `--model_name` - Model to probe
- `--train_dataset_path` - Training data (JSONL)
- `--val_dataset_path` - Validation data (JSONL, optional)
- `--max_k` - Maximum lookahead distance
- `--probe_type` - "linear" or "mlp"
- `--skip_check` - Skip model compatibility check

### `run_layer_k_experiment.sh`
**Purpose:** Quick pipeline test with small parameters

**What it does:**
Runs `layer_k_experiment.py` with small test parameters to verify the pipeline works

**Usage:**
```bash
bash scripts/run_layer_k_experiment.sh
```

**Test parameters:**
- 10 prompts, 20 tokens, k=1,2,3
- Linear probe, 2 epochs
- Fast execution (~5-10 minutes)

---

### `test_run.sh`
**Purpose:** Quick smoke test of the full pipeline

**What it does:**
1. Checks model integrity
2. Builds small test dataset
3. Trains a single probe to verify everything works

**Usage:**
```bash
cd look_ahead_probe
./test_run.sh
```

---

### `check_model.sh`
**Purpose:** Verify model compatibility for activation extraction

**Usage:**
```bash
./check_model.sh
```

## Running from Root

All scripts should be run from their respective directories. If you want to run from project root:

```bash
# For experiments
bash scripts/run_layer_k_experiment.sh

# For testing
bash look_ahead_probe/test_run.sh
```
