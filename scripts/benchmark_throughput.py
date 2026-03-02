"""Benchmark tokenizer throughput on varying chunk sizes and extrapolate to 825GB.

Reads increasing chunks from each data file, times encode(), and fits a
linear model (bytes vs seconds) to estimate throughput.
"""

from __future__ import annotations

import time

from cs336_basics.tokenizer.tokenizer import Tokenizer

DELIMITER = "<|endoftext|>"

DATASETS = {
    "TinyStories-valid": "data/TinyStoriesV2-GPT4-valid.txt",
    "TinyStories-train": "data/TinyStoriesV2-GPT4-train.txt",
    "OWT-valid": "data/owt_valid.txt",
    "OWT-train": "data/owt_train.txt",
}

# File sizes in bytes (from ls -l)
FILE_SIZES = {
    "TinyStories-valid": 21 * 1024**2,
    "TinyStories-train": 2.1 * 1024**3,
    "OWT-valid": 277 * 1024**2,
    "OWT-train": 11 * 1024**3,
}

TINYSTORIES_VOCAB = "output/tinystories_bpe/vocab.json"
TINYSTORIES_MERGES = "output/tinystories_bpe/merges.json"
OWT_VOCAB = "output/owt_bpe/vocab.json"
OWT_MERGES = "output/owt_bpe/merges.json"

# Chunk sizes to benchmark (in bytes)
CHUNK_SIZES = [100_000, 500_000, 1_000_000, 5_000_000]

PILE_SIZE_BYTES = 825 * 1024**3  # 825 GB


def read_chunk(filepath: str, size_bytes: int) -> str:
    """Read approximately size_bytes of text from the beginning of a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read(size_bytes)


def benchmark_tokenizer(tokenizer: Tokenizer, text: str) -> tuple[int, float]:
    """Encode text and return (num_bytes, elapsed_seconds)."""
    num_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    return num_bytes, elapsed


def main() -> None:
    ts_tokenizer = Tokenizer.from_files(
        TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=[DELIMITER]
    )
    owt_tokenizer = Tokenizer.from_files(
        OWT_VOCAB, OWT_MERGES, special_tokens=[DELIMITER]
    )

    tokenizers = {
        "TinyStories-valid": ts_tokenizer,
        "TinyStories-train": ts_tokenizer,
        "OWT-valid": owt_tokenizer,
        "OWT-train": owt_tokenizer,
    }

    all_points: list[tuple[int, float]] = []  # (bytes, seconds)

    for name, filepath in DATASETS.items():
        tok = tokenizers[name]
        print(f"\n--- {name} ({filepath}) ---")
        for chunk_size in CHUNK_SIZES:
            text = read_chunk(filepath, chunk_size)
            num_bytes, elapsed = benchmark_tokenizer(tok, text)
            throughput = num_bytes / elapsed if elapsed > 0 else float("inf")
            all_points.append((num_bytes, elapsed))
            print(
                f"  {num_bytes:>10,} bytes | {elapsed:>8.3f}s | "
                f"{throughput / 1024**2:>8.2f} MB/s"
            )

    # Fit linear model: seconds = slope * bytes  (no intercept, throughput is constant)
    # slope = sum(b*s) / sum(b*b)
    sum_bs = sum(b * s for b, s in all_points)
    sum_bb = sum(b * b for b, s in all_points)
    slope = sum_bs / sum_bb  # seconds per byte

    throughput_bps = 1.0 / slope
    throughput_mbps = throughput_bps / 1024**2

    pile_seconds = PILE_SIZE_BYTES * slope
    pile_hours = pile_seconds / 3600
    pile_days = pile_hours / 24

    print("\n=== Throughput Estimate ===")
    print(f"Fitted throughput: {throughput_mbps:.2f} MB/s")
    print(f"\nTime to tokenize the Pile (825 GB):")
    print(f"  {pile_seconds:,.0f} seconds")
    print(f"  {pile_hours:,.1f} hours")
    print(f"  {pile_days:,.1f} days")


if __name__ == "__main__":
    main()