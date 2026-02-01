"""
Data loading utilities for activation extraction.

This module handles loading prompts from JSONL files and creating
PyTorch datasets for training probes.
"""

import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset


def load_jsonl_prompts(
    path: str,
    *,
    text_field: str = "text",
    split_field: Optional[str] = "split",
    split_value: Optional[str] = None,
    default_split: str = "train",
    max_examples: Optional[int] = None,
) -> List[str]:
    """
    Load prompts from a JSONL file.

    Expected schema (per line): {"text": "...", "split": "train"|"val"|"test", ...}

    Args:
        path: Path to a .jsonl file.
        text_field: JSON key to read prompt text from.
        split_field: JSON key to read split name from. If None, do not use splits.
        split_value: If provided, only keep examples whose split equals this.
        default_split: Used when split_field exists but is missing on a row.
        max_examples: If provided, stop after loading this many prompts.

    Returns:
        List of prompt strings

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file is not .jsonl or contains invalid data
    """
    prompts: List[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if p.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a .jsonl dataset file, got: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_idx} of {path}, got {type(obj).__name__}")

            if split_field is not None:
                row_split = obj.get(split_field, default_split)
                if split_value is not None and row_split != split_value:
                    continue

            text = obj.get(text_field, None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing/empty '{text_field}' string on line {line_idx} of {path}. "
                    f"Got: {repr(text)}"
                )
            prompts.append(text)

            if max_examples is not None and len(prompts) >= max_examples:
                break

    return prompts


class ActivationDataset(Dataset):
    """Dataset of activations and future token targets."""

    def __init__(self, activations: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            activations: [n_samples, d_model] - activations at current position
            targets: [n_samples] - token IDs that appear k steps in the future
        """
        assert len(activations) == len(targets), "Activations and targets must have same length"
        self.activations = activations
        self.targets = targets

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]
