"""Training script for Transformer LLM.

Usage:
    python -m scripts.train \
        --train_data output/tinystories_bpe/train.npy \
        --val_data output/tinystories_bpe/valid.npy \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 8 \
        --d_ff 1024 \
        --max_iters 5000 \
        --batch_size 32

All hyperparameters are configurable via command-line arguments.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import wandb

from cs336_basics.nn import TransformerLLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LLM")

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Path to training .npy file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation .npy file")

    # Model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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

    return parser.parse_args()


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

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Wandb
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Training loop
    print(f"\nTraining for {args.max_iters} iterations...")
    model.train()
    start_time = time.perf_counter()

    for it in range(1, args.max_iters + 1):
        # Update learning rate
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log training loss
        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = it * args.batch_size * args.context_length / elapsed
            print(
                f"  iter {it:>6d}/{args.max_iters} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:,.0f} tok/s"
            )
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}, step=it)

        # Evaluate on validation set
        if it % args.val_interval == 0:
            val_loss = estimate_val_loss(model, val_data, args.batch_size, args.context_length, args.device, args.val_batches)
            print(f"  >>> val loss: {val_loss:.4f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss}, step=it)

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