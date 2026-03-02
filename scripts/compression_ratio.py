"""Compute compression ratios for BPE tokenizers on sampled documents.

Usage:
    python scripts/compression_ratio.py --m 10 --n 10

    m = number of documents to sample from TinyStories
    n = number of documents to sample from OpenWebText
"""

from __future__ import annotations

import argparse
import random

from cs336_basics.tokenizer.tokenizer import Tokenizer

DELIMITER = "<|endoftext|>"

TINYSTORIES_DATA = "data/TinyStoriesV2-GPT4-valid.txt"
TINYSTORIES_VOCAB = "output/tinystories_bpe/vocab.json"
TINYSTORIES_MERGES = "output/tinystories_bpe/merges.json"

OWT_DATA = "data/owt_valid.txt"
OWT_VOCAB = "output/owt_bpe/vocab.json"
OWT_MERGES = "output/owt_bpe/merges.json"


def count_lines(filepath: str) -> int:
    """Count total lines in a file without loading it into memory."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def sample_documents(filepath: str, num_samples: int, seed: int = 42) -> list[str]:
    """Sample documents by picking random lines and reading the next full document.

    For each random line, scan forward to the next <|endoftext|> (to reach a
    document boundary), then read until the following <|endoftext|> to capture
    one complete document.
    """
    total_lines = count_lines(filepath)
    rng = random.Random(seed)
    # Pick random line numbers, sorted so we can do a single forward pass
    sampled_lines = sorted(rng.sample(range(total_lines), min(num_samples, total_lines)))

    documents: list[str] = []
    sample_idx = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if sample_idx >= len(sampled_lines):
                break

            if line_no < sampled_lines[sample_idx]:
                continue

            # We've reached a sampled line. Now scan forward to find the next
            # <|endoftext|> boundary, then capture the document after it.
            # First, skip to end of current (partial) document.
            found_boundary = DELIMITER in line
            if not found_boundary:
                for line in f:
                    if DELIMITER in line:
                        found_boundary = True
                        break

            if not found_boundary:
                # Reached EOF without finding a delimiter
                sample_idx += 1
                continue

            # Now read the next complete document until <|endoftext|>
            doc_parts: list[str] = []
            for line in f:
                if DELIMITER in line:
                    # Take everything before the delimiter
                    doc_parts.append(line.split(DELIMITER)[0])
                    break
                doc_parts.append(line)

            doc_text = "".join(doc_parts).strip()
            if doc_text:
                documents.append(doc_text)

            sample_idx += 1

    return documents


def compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> float:
    """Compute compression ratio = total bytes / total tokens."""
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
    total_tokens = sum(1 for _ in tokenizer.encode_iterable(documents))
    return total_bytes / total_tokens if total_tokens > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BPE compression ratios")
    parser.add_argument("--m", type=int, default=10, help="Samples from TinyStories")
    parser.add_argument("--n", type=int, default=10, help="Samples from OpenWebText")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Sample documents ---
    print(f"Sampling {args.m} documents from TinyStories...")
    ts_docs = sample_documents(TINYSTORIES_DATA, args.m, seed=args.seed)
    print(f"  Got {len(ts_docs)} documents")

    print(f"Sampling {args.n} documents from OpenWebText...")
    owt_docs = sample_documents(OWT_DATA, args.n, seed=args.seed)
    print(f"  Got {len(owt_docs)} documents")

    # --- Build tokenizers ---
    ts_tokenizer = Tokenizer.from_files(
        TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=[DELIMITER]
    )
    owt_tokenizer = Tokenizer.from_files(
        OWT_VOCAB, OWT_MERGES, special_tokens=[DELIMITER]
    )

    # --- Cross-evaluation: each tokenizer on both datasets ---
    ts_on_ts = compression_ratio(ts_tokenizer, ts_docs)
    ts_on_owt = compression_ratio(ts_tokenizer, owt_docs)
    owt_on_ts = compression_ratio(owt_tokenizer, ts_docs)
    owt_on_owt = compression_ratio(owt_tokenizer, owt_docs)

    # --- Summary ---
    print("\n=== Compression Ratios (bytes/token) ===")
    print(f"{'':>25} {'TinyStories data':>18} {'OpenWebText data':>18}")
    print(f"{'TinyStories tokenizer':>25} {ts_on_ts:>18.2f} {ts_on_owt:>18.2f}")
    print(f"{'OpenWebText tokenizer':>25} {owt_on_ts:>18.2f} {owt_on_owt:>18.2f}")


if __name__ == "__main__":
    main()
