# look_ahead_probe

A modular Python package for training probes to predict future tokens from language model activations.

## Package Structure

```
look_ahead_probe/
├── __init__.py                  # Package initialization and exports
├── data_loading.py              # JSONL data loading and PyTorch datasets
├── activation_extraction.py     # Efficient activation extraction during generation
├── probe.py                     # Probe architecture definitions
├── train_probe.py              # Training script and CLI
└── README.md                    # This file
```

## Module Descriptions

### `data_loading.py`
Handles loading prompts from JSONL files and creating PyTorch datasets.

**Key components:**
- `load_jsonl_prompts()`: Load prompts from JSONL with split support
- `ActivationDataset`: PyTorch Dataset for activation-target pairs

### `activation_extraction.py`
Extracts activations from language models during generation using an efficient single-pass approach.

**Key components:**
- `generate_and_extract_activations()`: Main extraction function
- `verify_activation_equivalence()`: Test if efficient extraction is valid for your model

**Key insight:** Uses the causal masking property to extract activations in a single forward pass after generation, making it 50× faster than token-by-token extraction.

### `probe.py`
Defines probe architectures for predicting future tokens.

**Key components:**
- `FutureTokenProbe`: Configurable probe (linear or MLP)

**Architectures:**
- `linear`: Single linear layer (fast, interpretable)
- `mlp`: Two-layer MLP with ReLU and dropout (more expressive)

### `train_probe.py`
Main training script with full pipeline and CLI.

**Key components:**
- `train_probe()`: Training loop with validation
- `evaluate_probe()`: Evaluation with top-1 and top-5 accuracy
- `main()`: CLI entry point

## Usage

### As a CLI tool

```bash
python -m look_ahead_probe.train_probe \
    --model_name gpt2-small \
    --layer 6 \
    --k 5 \
    --dataset_path ../example_dataset.jsonl \
    --num_epochs 10 \
    --batch_size 64
```

### As a Python package

```python
from look_ahead_probe import (
    load_jsonl_prompts,
    generate_and_extract_activations,
    FutureTokenProbe,
    ActivationDataset,
)
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Load prompts
prompts = load_jsonl_prompts("data.jsonl", split_value="train")

# Extract activations
activations, targets, texts = generate_and_extract_activations(
    model=model,
    prompts=prompts,
    layer_idx=6,
    k=5,  # Predict 5 tokens ahead
    max_new_tokens=50
)

# Create dataset and dataloader
dataset = ActivationDataset(activations, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize probe
probe = FutureTokenProbe(
    input_dim=model.cfg.d_model,
    vocab_size=model.cfg.d_vocab,
    probe_type="linear"
)

# Train probe (see train_probe.py for full training loop)
```

## Verification

Before running large experiments, verify that the efficient extraction is valid:

```bash
python -m look_ahead_probe.train_probe \
    --model_name gpt2-small \
    --layer 6 \
    --k 5 \
    --dataset_path ../example_dataset.jsonl \
    --verify_equivalence
```

This tests whether activations extracted during generation match those from a single forward pass.

## Command-Line Arguments

See `train_probe.py --help` for full documentation of command-line arguments.

**Key arguments:**
- `--model_name`: Model to probe (e.g., gpt2-small)
- `--layer`: Which layer to extract activations from
- `--k`: How many tokens ahead to predict
- `--probe_type`: Architecture (linear or mlp)
- `--dataset_path`: Path to JSONL dataset
- `--verify_equivalence`: Run verification before training
- `--save_path`: Where to save trained probe

## Dataset Format

JSONL files with `text` field and optional `split` field:

```json
{"text": "The cat sat on the mat.", "split": "train"}
{"text": "Machine learning is fascinating.", "split": "train"}
{"text": "The weather is nice today.", "split": "val"}
```

## Dependencies

```
torch
transformers
transformer-lens
tqdm
numpy
```

## See Also

- [ACTIVATION_EXTRACTION_APPROACH.md](../ACTIVATION_EXTRACTION_APPROACH.md) - Detailed explanation of the efficient extraction method
- [README.md](../README.md) - Project-level documentation
