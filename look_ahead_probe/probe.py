"""
Probe architectures for predicting future tokens from activations.

This module defines neural network architectures that learn to predict
what token a language model will output k steps in the future, given
the model's internal activation at the current position.
"""

import torch.nn as nn


class FutureTokenProbe(nn.Module):
    """Probe that predicts what token the model will output in k steps."""

    def __init__(self, input_dim: int, vocab_size: int, probe_type: str = "linear"):
        """
        Initialize a future token prediction probe.

        Args:
            input_dim: Dimension of input activations (typically model's d_model)
            vocab_size: Size of the model's vocabulary
            probe_type: Type of probe architecture:
                - "linear": Single linear layer (fast, interpretable)
                - "mlp": Two-layer MLP with ReLU and dropout (more expressive)

        Raises:
            ValueError: If probe_type is not recognized
        """
        super().__init__()
        self.probe_type = probe_type

        if probe_type == "linear":
            # Simple linear probe: directly map activations to logits
            self.probe = nn.Linear(input_dim, vocab_size)
        elif probe_type == "mlp":
            # MLP probe: add nonlinearity and capacity
            self.probe = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim * 2, vocab_size)
            )
        else:
            raise ValueError(f"Unknown probe type: {probe_type}. Choose 'linear' or 'mlp'.")

    def forward(self, x):
        """
        Forward pass through the probe.

        Args:
            x: Input activations [batch_size, input_dim]

        Returns:
            logits: Token prediction logits [batch_size, vocab_size]
        """
        return self.probe(x)
