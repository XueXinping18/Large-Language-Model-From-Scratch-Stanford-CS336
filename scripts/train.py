"""Training script for Transformer LLM.

Usage:
    # From command line:
    python -m scripts.train \
        --train_data output/tinystories_bpe/train.npy \
        --val_data output/tinystories_bpe/valid.npy \
        --vocab_size 10000

    # From a JSON config file:
    python -m scripts.train --config configs/tinystories_overfit.json

    # Config file overrides can be mixed with CLI args (CLI wins):
    python -m scripts.train --config configs/tinystories.json --max_lr 5e-4

All hyperparameters are configurable via command-line arguments or JSON config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import wandb
import numpy as np
import torch

from cs336_basics.nn import TransformerLLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LLM")

    # Config file (values serve as defaults; CLI args override)
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    # Data
    parser.add_argument("--train_data", type=str, default=None, help="Path to training .npy file")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation .npy file")

    # Model
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=None, help="FFN inner dim. Default: round(8/3 * d_model) to multiple of 64")
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--max_iters", type=int, default=5000)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="JIT-compile model with torch.compile")
    parser.add_argument("--overfit_single_batch", action="store_true",
                        help="Cache one batch and reuse it every iteration (sanity check)")

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N iterations")
    parser.add_argument("--val_interval", type=int, default=200, help="Evaluate on val set every N iterations")
    parser.add_argument("--val_batches", type=int, default=20, help="Number of val batches per evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N iterations")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-transformer")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # First parse to get --config path
    args, remaining = parser.parse_known_args()

    # If config file provided, load it and set as defaults (CLI args override)
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        parser.set_defaults(**config)

    # Re-parse with config defaults applied
    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = _detect_device()

    # Validate required fields
    if args.train_data is None or args.vocab_size is None:
        parser.error("--train_data and --vocab_size are required (via CLI or config)")

    return args


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all gradients (before clipping)."""
    total = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    return total ** 0.5


def compute_weight_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all model weights."""
    total = sum(p.norm() ** 2 for p in model.parameters())
    return total ** 0.5


def compute_activation_stats(logits: torch.Tensor) -> dict[str, float]:
    """Compute stats of the final layer output (logits) as a proxy for activation health."""
    return {
        "activation/logits_mean": logits.mean().item(),
        "activation/logits_std": logits.std().item(),
        "activation/logits_max": logits.abs().max().item(),
    }


def compute_d_ff(d_model: int) -> int:
    """Compute d_ff as ~8/3 * d_model, rounded to multiple of 64."""
    return int(round(8 / 3 * d_model / 64)) * 64


@torch.no_grad()
def estimate_val_loss(
    model: TransformerLLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """Estimate validation loss over multiple batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        # Flatten (batch, seq, vocab) -> (batch*seq, vocab) for cross_entropy
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main() -> None:
    args = parse_args()

    # Compute d_ff if not provided
    if args.d_ff is None:
        args.d_ff = compute_d_ff(args.d_model)

    # Load data with memmap (memory-efficient)
    print(f"Loading training data from {args.train_data}...")
    train_data = np.load(args.train_data, mmap_mode="r")
    print(f"  {len(train_data):,} tokens")

    print(f"Loading validation data from {args.val_data}...")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(f"  {len(val_data):,} tokens")

    # Build model
    print(f"\nBuilding model on {args.device}...")
    model = TransformerLLM(
        vocal_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {num_params:,} parameters")

    # JIT compile
    if args.compile:
        if args.device == "mps":
            print("  Compiling model with backend='aot_eager' (MPS)...")
            model = torch.compile(model, backend="aot_eager")
        else:
            print("  Compiling model...")
            model = torch.compile(model)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Checkpoint directory & save config
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Cache a single batch for overfit mode
    if args.overfit_single_batch:
        print("\n*** OVERFIT MODE: reusing a single batch every iteration ***")
        fixed_x, fixed_y = get_batch(train_data, args.batch_size, args.context_length, args.device)

    # Training loop
    total_tokens = args.batch_size * args.max_iters * args.context_length
    print(f"\nTraining for {args.max_iters} iterations ({total_tokens:,} tokens)...")
    model.train()
    start_time = time.perf_counter()

    for it in range(1, args.max_iters + 1):
        # Update learning rate
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        if args.overfit_single_batch:
            x, y = fixed_x, fixed_y
        else:
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = compute_grad_norm(model).item()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log training metrics
        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = it * args.batch_size * args.context_length / elapsed
            train_ppl = math.exp(min(loss.item(), 20))  # cap to avoid overflow
            weight_norm = compute_weight_norm(model).item()
            act_stats = compute_activation_stats(logits)
            print(
                f"  iter {it:>6d}/{args.max_iters} | "
                f"loss {loss.item():.4f} | ppl {train_ppl:.1f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"w_norm {weight_norm:.1f} | "
                f"{tokens_per_sec:,.0f} tok/s | {elapsed:.0f}s"
            )
            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": train_ppl,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/weight_norm": weight_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/wallclock_sec": elapsed,
                    **act_stats,
                }, step=it)

        # Evaluate on validation set
        if it % args.val_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args.batch_size, args.context_length, args.device, args.val_batches)
            val_ppl = math.exp(min(val_loss, 20))
            elapsed = time.perf_counter() - start_time
            print(f"  >>> val loss: {val_loss:.4f} | val ppl: {val_ppl:.1f}")
            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "val/wallclock_sec": elapsed,
                }, step=it)

        # Save checkpoint
        if it % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"  >>> saved checkpoint to {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"\nTraining complete. Final checkpoint saved to {final_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()