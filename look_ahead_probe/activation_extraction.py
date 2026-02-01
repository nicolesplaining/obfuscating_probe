"""
Activation extraction from language models during generation.

This module provides efficient extraction of internal activations from
transformer language models, leveraging the causal masking property to
extract activations in a single forward pass after generation.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm


def verify_activation_equivalence(
    model: HookedTransformer,
    prompt: str,
    layer_idx: int,
    max_new_tokens: int,
    device: str = "cuda"
) -> bool:
    """
    Verify that activations extracted during token-by-token generation
    match those extracted from a single forward pass on the complete sequence.

    This tests the core assumption that enables efficient activation extraction:
    Due to causal masking in transformers, the activation at position i only
    depends on tokens[:i+1]. Therefore, it will be identical whether computed:
    (a) during autoregressive generation (position i is part of a sequence of length i+1)
    (b) in a single forward pass on the complete sequence (position i sees the same tokens[:i+1])

    If this returns True, you can safely use the efficient single-pass approach.
    If False, there may be numerical instability, non-deterministic operations,
    or incorrect causal masking in the model.

    Args:
        model: The language model to test
        prompt: A single text prompt to generate from
        layer_idx: Which layer to extract activations from
        max_new_tokens: Number of tokens to generate for the test
        device: Device to use

    Returns:
        True if max absolute difference < 1e-5 at all positions, False otherwise
    """
    model.eval()
    eos_token_id = model.tokenizer.eos_token_id

    print(f"\n{'='*60}")
    print(f"Verifying Activation Equivalence")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Layer: {layer_idx}, Max new tokens: {max_new_tokens}\n")

    with torch.no_grad():
        # METHOD 1: Token-by-token generation (storing activations as we go)
        current_tokens = model.to_tokens(prompt).to(device)
        method1_activations = {}

        for step in range(max_new_tokens):
            logits, cache = model.run_with_cache(current_tokens)
            layer_acts = cache["resid_post", layer_idx][0]  # [seq_len, d_model]

            # Store activations for positions we haven't stored yet
            for pos in range(current_tokens.shape[1]):
                if pos not in method1_activations:
                    method1_activations[pos] = layer_acts[pos].cpu()

            # Greedy generation
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        final_tokens_method1 = current_tokens

        # METHOD 2: Generate first, then single forward pass
        # (Re-generate to ensure same tokens)
        current_tokens = model.to_tokens(prompt).to(device)

        for step in range(max_new_tokens):
            logits = model(current_tokens)
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        # Single forward pass on complete sequence
        _, cache = model.run_with_cache(current_tokens)
        method2_activations = cache["resid_post", layer_idx][0]  # [seq_len, d_model]

        final_tokens_method2 = current_tokens

        # Verify tokens are identical
        if not torch.equal(final_tokens_method1, final_tokens_method2):
            print("ERROR: Generated tokens differ between methods!")
            return False

        # Compare activations at each position
        print(f"Comparing activations at {len(method1_activations)} positions:\n")
        print(f"{'Position':<10} {'Max Difference':<20} {'Status':<10}")
        print(f"{'-'*40}")

        all_valid = True
        max_diff_overall = 0.0

        for pos in sorted(method1_activations.keys()):
            act1 = method1_activations[pos]
            act2 = method2_activations[pos].cpu()

            max_diff = (act1 - act2).abs().max().item()
            max_diff_overall = max(max_diff_overall, max_diff)

            is_valid = max_diff < 1e-5
            status = "✓ PASS" if is_valid else "✗ FAIL"
            all_valid = all_valid and is_valid

            print(f"{pos:<10} {max_diff:<20.2e} {status:<10}")

        print(f"{'-'*40}")
        print(f"Overall max difference: {max_diff_overall:.2e}")
        print(f"Threshold: 1e-5")
        print(f"\nResult: {'✓ EQUIVALENCE VERIFIED' if all_valid else '✗ EQUIVALENCE FAILED'}")
        print(f"{'='*60}\n")

        return all_valid


def generate_and_extract_activations(
    model: HookedTransformer,
    prompts: List[str],
    layer_idx: int,
    k: int,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate text and extract activations using an efficient single-pass approach.

    EFFICIENT APPROACH EXPLANATION:
    Instead of extracting activations during token-by-token generation (which requires
    storing activations at each generation step), we use a two-step process:

    1. Generate the complete sequence first using greedy decoding (deterministic)
    2. Run a SINGLE forward pass on the entire generated sequence
    3. Extract all activations at once from the activation cache

    WHY THIS WORKS - The Causal Masking Property:
    In transformer models with causal (autoregressive) masking, the activation at
    position i only depends on tokens at positions [0, 1, ..., i]. It does NOT depend
    on future tokens at positions [i+1, i+2, ...] because the attention mask prevents
    information flow from future positions.

    This means:
    - Activation at position i during generation (when sequence length is i+1)
    - Activation at position i in a batch pass (when sequence length is N, with i < N)

    These are IDENTICAL because position i sees the same tokens[:i+1] in both cases.

    BENEFITS:
    - Much faster: Single forward pass vs. N forward passes for N tokens
    - Lower memory: Don't need to store intermediate activations during generation
    - Simpler code: No complex state tracking during generation

    CRITICAL ASSUMPTION:
    This relies on the model being deterministic (eval mode, no dropout) and using
    proper causal masking. Use verify_activation_equivalence() to test this assumption
    holds for your specific model.

    Args:
        model: The language model
        prompts: List of text prompts to start generation from
        layer_idx: Which layer to extract activations from
        k: How many tokens ahead to predict (activation[i] -> token[i+k])
        max_new_tokens: Maximum number of tokens to generate per prompt (default: 50)
        temperature: Kept for API compatibility (currently uses greedy decoding)
        device: Device to use

    Returns:
        activations: [n_samples, d_model] - activations at position i
        targets: [n_samples] - token IDs at position i+k
        generated_texts: List of generated text strings for inspection
    """
    all_activations = []
    all_targets = []
    generated_texts = []

    model.eval()
    eos_token_id = model.tokenizer.eos_token_id

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating and extracting activations"):
            # STEP 1: Generate complete sequence using greedy decoding
            # We use greedy (deterministic) generation to ensure reproducibility
            # and to match the verification assumptions
            current_tokens = model.to_tokens(prompt).to(device)  # [1, prompt_len]

            for step in range(max_new_tokens):
                # Forward pass to get logits (don't need cache here)
                logits = model(current_tokens)

                # Greedy decoding: always pick most likely token
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

                # Append to sequence
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

                # Stop if EOS token generated
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

            # STEP 2: Single forward pass on complete generated sequence
            # This is the key optimization - one pass instead of N passes
            # Thanks to causal masking, activations will be identical to those
            # computed during generation
            _, cache = model.run_with_cache(current_tokens)

            # STEP 3: Extract all activations at once
            # Shape: [seq_len, d_model] - all positions in one tensor
            layer_acts = cache["resid_post", layer_idx][0]

            # Decode generated text for inspection
            generated_text = model.tokenizer.decode(current_tokens[0])
            generated_texts.append(generated_text)

            # STEP 4: Create training pairs by slicing the activation tensor
            # For each valid position i, pair activation[i] with token[i+k]
            total_len = current_tokens.shape[1]
            for i in range(total_len - k):
                # Slice activation at position i directly from the tensor
                act = layer_acts[i, :]  # [d_model]

                # Target token at position i+k
                target = current_tokens[0, i + k]

                all_activations.append(act.cpu())
                all_targets.append(target.cpu())

    # Stack all samples into tensors
    activations = torch.stack(all_activations)  # [n_samples, d_model]
    targets = torch.stack(all_targets)  # [n_samples]

    return activations, targets, generated_texts
