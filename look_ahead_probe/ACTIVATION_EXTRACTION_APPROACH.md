# Activation Extraction Approach for Future Token Prediction Probes

## Overview

This codebase extracts activations from language models to train probes that predict future tokens. Specifically, we:

1. **Generate text sequences** from prompts using a language model
2. **Extract internal activations** (hidden states) at each position in the sequence
3. **Create training pairs**: activation at position `i` → token at position `i+k`
4. **Train a probe** (linear classifier or small MLP) to predict token `i+k` given activation `i`

The goal is to understand if and how the model internally represents information about future tokens - a form of "planning" or "look-ahead" in the model's computations.

## Why the Efficient Approach Works

### The Causal Masking Property

Transformer language models use **causal (autoregressive) attention masking**. This means:

- Token at position `i` can only attend to positions `[0, 1, 2, ..., i]`
- Token at position `i` **cannot** see future tokens at positions `[i+1, i+2, ...]`
- The activation at position `i` is computed using **only** the tokens it can see: `tokens[:i+1]`

### Two Ways to Compute the Same Activation

Because of causal masking, we can compute activations in two equivalent ways:

**Method 1: During Token-by-Token Generation (Naive)**
```python
# Start with prompt
tokens = [t0, t1, t2]  # prompt

# Generate token 3
run_model([t0, t1, t2]) → get activation at position 2
# Activation at pos 2 sees: [t0, t1, t2]

# Generate token 4
run_model([t0, t1, t2, t3]) → get activation at position 2
# Activation at pos 2 sees: [t0, t1, t2] (same as before!)
```

**Method 2: After Complete Generation (Efficient)**
```python
# Generate complete sequence first
tokens = [t0, t1, t2, t3, t4, t5]

# Single forward pass
run_model([t0, t1, t2, t3, t4, t5])
# Activation at pos 2 sees: [t0, t1, t2] (due to causal masking)
# Activation at pos 3 sees: [t0, t1, t2, t3]
# etc.
```

**Key Insight**: Due to causal masking, the activation at position `i` in Method 2 is **identical** to the activation computed in Method 1, because both see exactly the same input tokens: `tokens[:i+1]`.

### Efficiency Gains

| Approach | Forward Passes | Memory | Complexity |
|----------|---------------|---------|------------|
| Token-by-token | N (one per generated token) | Store activations incrementally | O(N) passes |
| Single-pass | 1 (on complete sequence) | Store activations once | O(1) passes |

For generating 50 tokens, the efficient approach is **50× faster** in terms of forward passes!

## Key Assumptions

The efficient single-pass approach relies on these assumptions:

### 1. **Proper Causal Masking**
The model must implement correct causal attention masking where position `i` cannot attend to positions `> i`.

✅ **True for**: Standard transformer LMs (GPT, LLaMA, etc.)
❌ **False for**: Encoder models (BERT), prefix-LM models with bidirectional prefix attention

### 2. **Deterministic Forward Pass**
The model's forward pass must be deterministic (same inputs → same outputs).

✅ **Ensured by**:
- `model.eval()` mode (disables dropout)
- No stochastic layers during inference
- Same input tokens

❌ **Could fail if**:
- Dropout is active during inference
- Random number generation without fixed seed
- Non-deterministic operations (e.g., non-deterministic CUDA ops)

### 3. **Identical Token Sequences**
We must use the **exact same generated tokens** for the single forward pass.

✅ **Ensured by**:
- Using greedy decoding (deterministic)
- Storing the generated tokens and reusing them

❌ **Could fail if**:
- Using sampling with different random seeds
- Re-generating tokens instead of reusing them

### 4. **Numerical Stability**
Floating-point operations should be numerically stable enough that the same computation produces the same result.

✅ **Generally true for**:
- Standard attention and feedforward layers
- Modern deep learning frameworks

❌ **Could fail if**:
- Extreme numerical instability in the model
- Different computation orders (rare but possible)

## When This Might Fail

### Edge Cases to Watch For:

1. **Sampling-based Generation**
   - If you use temperature sampling, nucleus sampling, or other stochastic methods, re-running generation will produce different tokens
   - **Solution**: Store and reuse the generated tokens, or use greedy decoding

2. **Different Attention Implementations**
   - Some optimized attention kernels (Flash Attention, xFormers) might have slight numerical differences
   - **Solution**: Verify equivalence (see below)

3. **Prefix-LM or Encoder-Decoder Models**
   - Models with bidirectional attention in the prefix won't satisfy causal masking
   - **Solution**: Only use this approach with pure autoregressive decoder models

4. **Adaptive Computation**
   - Models with dynamic depth or early exiting might behave differently
   - **Solution**: Not applicable for such models; use token-by-token approach

5. **External State**
   - Models that maintain hidden state between calls (RNNs, Mamba, etc.)
   - **Solution**: Not applicable; use token-by-token approach

## Verification: Testing the Assumption

We provide a verification function to test if the efficient approach is valid for your model:

```python
from probe import verify_activation_equivalence

# Test on your model
is_valid = verify_activation_equivalence(
    model=model,
    prompt="The quick brown fox jumps over the",
    layer_idx=6,  # Test at layer 6
    max_new_tokens=20,
    device="cuda"
)

if is_valid:
    print("✓ Safe to use efficient single-pass approach")
else:
    print("✗ Use token-by-token approach instead")
```

### What the Verification Does:

1. **Method A**: Generate tokens one-by-one, storing activation at each step
2. **Method B**: Generate complete sequence, then run single forward pass
3. **Compare**: Check if activations differ by more than `1e-5`

If the maximum difference is below `1e-5` at all positions, the approaches are equivalent.

### Example Output:

```
============================================================
Verifying Activation Equivalence
============================================================
Prompt: The quick brown fox jumps over the...
Layer: 6, Max new tokens: 20

Comparing activations at 25 positions:

Position   Max Difference       Status
----------------------------------------
0          3.45e-08            ✓ PASS
1          2.11e-08            ✓ PASS
2          5.67e-08            ✓ PASS
...
24         1.23e-08            ✓ PASS
----------------------------------------
Overall max difference: 5.67e-08
Threshold: 1e-5

Result: ✓ EQUIVALENCE VERIFIED
============================================================
```

## Trade-offs: Efficient vs. Token-by-Token

### Efficient Single-Pass Approach

**Pros:**
- ⚡ **Fast**: Single forward pass instead of N passes
- 💾 **Memory efficient**: Don't store intermediate states during generation
- 🎯 **Simple**: Clean, straightforward implementation
- 🔍 **Easy to verify**: Can test equivalence explicitly

**Cons:**
- 🎲 **Greedy only**: Currently limited to deterministic greedy decoding
- 🧪 **Requires verification**: Must verify assumption holds for your model
- 🚫 **Not suitable for all models**: Requires causal masking

**Best for:**
- Standard transformer decoder models (GPT-2, LLaMA, etc.)
- Large-scale activation extraction (many prompts, long sequences)
- When generation quality isn't critical (greedy is acceptable)

### Token-by-Token Approach

**Pros:**
- 🎲 **Flexible sampling**: Can use temperature, top-k, top-p, etc.
- 🔒 **Always correct**: Directly captures what happened during generation
- 🌍 **Universal**: Works for any model architecture
- 🎯 **True generation activations**: Exactly what the model computed

**Cons:**
- 🐌 **Slow**: N forward passes for N generated tokens
- 💾 **Memory intensive**: Must store activations during generation
- 🔧 **Complex**: Careful state management required

**Best for:**
- Models without causal masking (RNNs, state-space models, etc.)
- When sampling diversity is important
- Smaller-scale experiments
- When you need exact generation-time activations

## Practical Recommendations

### Use the Efficient Approach When:
- ✅ Using a standard transformer decoder (GPT-2, LLaMA, OPT, etc.)
- ✅ Greedy decoding is acceptable for your use case
- ✅ You've verified equivalence on your specific model
- ✅ You're extracting activations from many prompts/long sequences

### Use Token-by-Token When:
- ✅ You need sampling (temperature, nucleus, etc.)
- ✅ Working with non-standard architectures
- ✅ Verification fails for your model
- ✅ You need activations from the exact generation process for interpretability

### Verification Checklist:

Before using the efficient approach in production:

- [ ] Run `verify_activation_equivalence()` on your model
- [ ] Test with multiple layers (early, middle, late)
- [ ] Test with different sequence lengths
- [ ] Verify on representative prompts from your dataset
- [ ] Check that `model.eval()` is called (no dropout)

## Code Example

### Using the Efficient Approach:

```python
from probe import generate_and_extract_activations, verify_activation_equivalence

# First, verify the assumption holds
is_valid = verify_activation_equivalence(
    model=model,
    prompt="Test prompt",
    layer_idx=6,
    max_new_tokens=50
)

if not is_valid:
    raise RuntimeError("Efficient approach not valid for this model!")

# Extract activations efficiently
activations, targets, generated_texts = generate_and_extract_activations(
    model=model,
    prompts=train_prompts,
    layer_idx=6,
    k=5,  # Predict 5 tokens ahead
    max_new_tokens=50,
    device="cuda"
)

# Train probe
# ... (rest of training code)
```

## Additional Notes

### On Temperature Sampling:

The current implementation uses greedy decoding for efficiency and reproducibility. If you need temperature sampling:

1. Generate with sampling to get diverse tokens
2. **Store the generated tokens**
3. Run the single forward pass on those stored tokens
4. The efficient approach still works because you're using the same tokens

### On Batch Processing:

The function processes prompts one at a time. For further speedup, you could:
- Batch multiple prompts together
- Pad to same length
- Run single forward pass on entire batch

However, this adds complexity with padding/masking, so we keep it simple for clarity.

### On Other Activation Types:

The current code extracts `resid_post` (residual stream after layer). You can extract:
- `attn_out`: Attention layer output
- `mlp_out`: MLP layer output
- `resid_mid`: Residual stream mid-layer
- etc.

The same efficient approach works for any activation type, as long as it's computed causally.

---

**Questions or issues?** The verification function is your friend - run it to test assumptions on your specific model and use case!
