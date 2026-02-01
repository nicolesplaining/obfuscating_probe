"""
look_ahead_probe: Train probes to predict future tokens from LM activations.

This package provides tools for extracting activations from language models
during generation and training probes to predict future tokens.
"""

from .activation_extraction import (
    generate_and_extract_activations,
    verify_activation_equivalence,
)
from .data_loading import ActivationDataset, load_jsonl_prompts
from .probe import FutureTokenProbe

__all__ = [
    "FutureTokenProbe",
    "ActivationDataset",
    "load_jsonl_prompts",
    "generate_and_extract_activations",
    "verify_activation_equivalence",
]
