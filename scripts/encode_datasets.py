"""Encode training and validation datasets into NumPy uint16 arrays of token IDs.

Usage:
    python -m scripts.encode_datasets

Outputs:
    output/tinystories_bpe/train.npy
    output/tinystories_bpe/valid.npy
    output/owt_bpe/train.npy
    output/owt_bpe/valid.npy
"""

from __future__ import annotations

import time

import numpy as np

from cs336_basics.tokenizer.tokenizer import Tokenizer

DELIMITER = "<|endoftext|>"

LOG_INTERVAL_MB = 50  # Print progress every this many MB of input

CONFIGS = [
    {
        "name": "TinyStories",
        "vocab": "output/tinystories_bpe/vocab.json",
        "merges": "output/tinystories_bpe/merges.json",
        "datasets": {
            "train": "data/TinyStoriesV2-GPT4-train.txt",
            "valid": "data/TinyStoriesV2-GPT4-valid.txt",
        },
        "output_dir": "output/tinystories_bpe",
    },
    {
        "name": "OpenWebText",
        "vocab": "output/owt_bpe/vocab.json",
        "merges": "output/owt_bpe/merges.json",
        "datasets": {
            "train": "data/owt_train.txt",
            "valid": "data/owt_valid.txt",
        },
        "output_dir": "output/owt_bpe",
    },
]


def encode_file(tokenizer: Tokenizer, filepath: str, output_path: str) -> None:
    """Encode a text file line-by-line and save token IDs as uint16 NumPy array."""
    start = time.perf_counter()
    token_ids: list[int] = []
    bytes_processed = 0
    next_log_at = LOG_INTERVAL_MB * 1024 * 1024

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            token_ids.extend(tokenizer.encode(line))
            bytes_processed += len(line.encode("utf-8"))

            if bytes_processed >= next_log_at:
                elapsed = time.perf_counter() - start
                mb = bytes_processed / 1024**2
                throughput = mb / elapsed
                print(
                    f"    {mb:.0f} MB processed | "
                    f"{len(token_ids):,} tokens | "
                    f"{elapsed:.1f}s | {throughput:.2f} MB/s"
                )
                next_log_at += LOG_INTERVAL_MB * 1024 * 1024

    elapsed = time.perf_counter() - start
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, arr)

    mb = bytes_processed / 1024**2
    print(
        f"  Done: {output_path}\n"
        f"    {mb:.1f} MB input | {len(arr):,} tokens | "
        f"{elapsed:.1f}s | {arr.nbytes / 1024**2:.1f} MB saved"
    )


def main() -> None:
    for config in CONFIGS:
        print(f"\n=== {config['name']} ===")
        tokenizer = Tokenizer.from_files(
            config["vocab"], config["merges"], special_tokens=[DELIMITER]
        )

        for split, data_path in config["datasets"].items():
            output_path = f"{config['output_dir']}/{split}.npy"
            print(f"  Encoding {split}: {data_path}")
            encode_file(tokenizer, data_path, output_path)


if __name__ == "__main__":
    main()