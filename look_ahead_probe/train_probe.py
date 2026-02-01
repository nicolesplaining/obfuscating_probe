"""
Training script for future token prediction probes.

This script handles the full pipeline:
1. Load a language model
2. Load prompts from JSONL dataset
3. Generate text and extract activations
4. Train a probe to predict future tokens
5. Evaluate and save the trained probe
"""

import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm

from activation_extraction import generate_and_extract_activations, verify_activation_equivalence
from data_loading import ActivationDataset, load_jsonl_prompts
from probe import FutureTokenProbe


def train_probe(
    probe: FutureTokenProbe,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: str = "cuda"
) -> dict:
    """
    Train the probe to predict future tokens.

    Args:
        probe: The probe model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on

    Returns:
        Dictionary containing training history:
            - train_loss: List of training losses per epoch
            - train_acc: List of training accuracies per epoch
            - val_loss: List of validation losses per epoch (if val_loader provided)
            - val_acc: List of validation accuracies per epoch (if val_loader provided)
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for activations, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            activations = activations.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = probe(activations)
            loss = F.cross_entropy(logits, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += len(targets)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        if val_loader is not None:
            probe.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for activations, targets in val_loader:
                    activations = activations.to(device)
                    targets = targets.to(device)

                    logits = probe(activations)
                    loss = F.cross_entropy(logits, targets)

                    val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == targets).sum().item()
                    val_total += len(targets)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

    return history


def evaluate_probe(
    probe: FutureTokenProbe,
    test_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """
    Evaluate probe on test set.

    Args:
        probe: The trained probe model
        test_loader: DataLoader for test data
        device: Device to evaluate on

    Returns:
        Dictionary containing evaluation metrics:
            - loss: Average cross-entropy loss
            - accuracy: Top-1 accuracy
            - top5_accuracy: Top-5 accuracy
            - predictions: List of predicted token IDs
            - targets: List of ground truth token IDs
    """
    probe.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for activations, targets in tqdm(test_loader, desc="Evaluating"):
            activations = activations.to(device)
            targets = targets.to(device)

            logits = probe(activations)
            loss = F.cross_entropy(logits, targets)

            test_loss += loss.item()
            preds = logits.argmax(dim=-1)
            test_correct += (preds == targets).sum().item()
            test_total += len(targets)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total

    # Top-k accuracy
    top5_correct = 0
    with torch.no_grad():
        for activations, targets in test_loader:
            activations = activations.to(device)
            targets = targets.to(device)

            logits = probe(activations)
            top5_preds = logits.topk(5, dim=-1).indices
            top5_correct += (top5_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()

    top5_acc = top5_correct / test_total

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "top5_accuracy": top5_acc,
        "predictions": all_preds,
        "targets": all_targets
    }


def main():
    parser = argparse.ArgumentParser(description="Train probe to predict future model outputs")
    parser.add_argument("--model_name", type=str, default="gpt2-small",
                        help="Model to probe")
    parser.add_argument("--layer", type=int, default=6,
                        help="Which layer to extract activations from")
    parser.add_argument("--k", type=int, required=True,
                        help="How many tokens in the future to predict")
    parser.add_argument("--probe_type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Type of probe")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trained probe")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to a JSONL dataset containing prompts (supports train/val splits via a 'split' field)")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Optional: path to a JSONL file for training prompts (overrides --dataset_path)")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Optional: path to a JSONL file for validation prompts (overrides --dataset_path)")
    parser.add_argument("--text_field", type=str, default="text",
                        help="JSON key containing the prompt text (default: text)")
    parser.add_argument("--split_field", type=str, default="split",
                        help="JSON key containing the split name when using --dataset_path (default: split)")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Split name to use as training data when using --dataset_path (default: train)")
    parser.add_argument("--val_split", type=str, default="val",
                        help="Split name to use as validation data when using --dataset_path (default: val)")
    parser.add_argument("--max_train_prompts", type=int, default=None,
                        help="Optional cap on number of training prompts loaded")
    parser.add_argument("--max_val_prompts", type=int, default=None,
                        help="Optional cap on number of validation prompts loaded")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate per prompt (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for generation (default: 1.0). Use 0.0 for greedy decoding")
    parser.add_argument("--verify_equivalence", action="store_true",
                        help="Run verification to check if efficient activation extraction is valid for this model")

    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)

    # Run verification if requested
    if args.verify_equivalence:
        print("\n" + "="*60)
        print("RUNNING ACTIVATION EQUIVALENCE VERIFICATION")
        print("="*60)
        test_prompt = "The quick brown fox jumps over the lazy dog."
        is_valid = verify_activation_equivalence(
            model=model,
            prompt=test_prompt,
            layer_idx=args.layer,
            max_new_tokens=min(args.max_new_tokens, 20),  # Use smaller value for faster verification
            device=args.device
        )
        if not is_valid:
            print("\n⚠️  WARNING: Verification failed! The efficient single-pass approach")
            print("may not be suitable for this model. Consider using a token-by-token")
            print("approach or investigating why activations differ.")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Exiting.")
                return
        else:
            print("\n✓ Verification passed! Safe to proceed with efficient extraction.")
        print("="*60 + "\n")

    print("Loading dataset...")
    train_prompts: List[str]
    val_prompts: Optional[List[str]]

    if args.train_path is not None or args.val_path is not None:
        if args.train_path is None:
            raise ValueError("If using separate split files, you must provide --train_path.")
        train_prompts = load_jsonl_prompts(
            args.train_path,
            text_field=args.text_field,
            split_field=None,
            max_examples=args.max_train_prompts,
        )
        val_prompts = None
        if args.val_path is not None:
            val_prompts = load_jsonl_prompts(
                args.val_path,
                text_field=args.text_field,
                split_field=None,
                max_examples=args.max_val_prompts,
            )
    elif args.dataset_path is not None:
        train_prompts = load_jsonl_prompts(
            args.dataset_path,
            text_field=args.text_field,
            split_field=args.split_field,
            split_value=args.train_split,
            max_examples=args.max_train_prompts,
        )
        val_prompts = load_jsonl_prompts(
            args.dataset_path,
            text_field=args.text_field,
            split_field=args.split_field,
            split_value=args.val_split,
            max_examples=args.max_val_prompts,
        )
    else:
        raise ValueError("No dataset provided")

    # Generate and extract activations from actual model generation
    print(f"Generating text and extracting activations from layer {args.layer} with k={args.k}")
    print(f"Generation settings: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}")
    train_acts, train_targets, train_generated = generate_and_extract_activations(
        model=model,
        prompts=train_prompts,
        layer_idx=args.layer,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device
    )

    # Print a few examples of generated text for inspection
    print("\nExample generated texts (first 3):")
    for i, text in enumerate(train_generated[:3]):
        print(f"  {i+1}. {text[:100]}...")  # Print first 100 chars

    val_acts = None
    val_targets = None
    if val_prompts:
        val_acts, val_targets, val_generated = generate_and_extract_activations(
            model=model,
            prompts=val_prompts,
            layer_idx=args.layer,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device
        )

    print(f"Training set: {len(train_acts)} samples")
    if val_acts is not None:
        print(f"Validation set: {len(val_acts)} samples")
    else:
        print("Validation set: (none)")

    # Create datasets and dataloaders
    train_dataset = ActivationDataset(
        activations=train_acts,
        targets=train_targets
    )

    val_dataset = None
    if val_acts is not None and val_targets is not None:
        val_dataset = ActivationDataset(val_acts, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize probe
    print(f"Initializing {args.probe_type} probe")
    probe = FutureTokenProbe(
        input_dim=model.cfg.d_model,
        vocab_size=model.cfg.d_vocab,
        probe_type=args.probe_type
    )

    # Train probe
    print("Training probe...")
    history = train_probe(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Evaluate
    results = None
    if val_loader is not None:
        print("\nEvaluating on validation set...")
        results = evaluate_probe(probe, val_loader, args.device)
        print(f"Final Results:")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    else:
        print("\nSkipping evaluation (no validation dataset provided).")

    # Save probe
    if args.save_path:
        torch.save({
            'probe_state_dict': probe.state_dict(),
            'args': vars(args),
            'history': history,
            'results': results,
        }, args.save_path)
        print(f"Probe saved to {args.save_path}")


if __name__ == "__main__":
    main()
