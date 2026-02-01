# Scripts

Helper scripts for experiments and testing.

## Available Scripts

### `run_layer_k_experiment.sh`
**Purpose:** Comprehensive layer-k probing experiment

**What it does:**
1. Builds activation dataset for all layers with k=1,2,3
2. Trains MLP probes for each (layer, k) combination
3. Generates results table showing accuracy across all configurations

**Usage:**
```bash
cd scripts
./run_layer_k_experiment.sh
```

**Output:**
- `experiment_results/datasets/` - Extracted activations
- `experiment_results/probes/` - Trained probe weights
- `experiment_results/results_table.txt` - Results summary table

**Customize:** Edit variables at top of script:
- `MODEL` - Model to probe
- `MAX_K` - Maximum lookahead distance
- `NUM_EPOCHS` - Training epochs
- `PROBE_TYPE` - "linear" or "mlp"

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
