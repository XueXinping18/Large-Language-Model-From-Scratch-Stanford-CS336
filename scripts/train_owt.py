"""Train a byte-level BPE tokenizer on OpenWebText and serialize results."""

import cProfile
import json
import pstats
import time
import tracemalloc
from pathlib import Path

from cs336_basics.tokenizer.train_bpe import train_bpe

INPUT_PATH = Path("data/owt_train.txt")
OUTPUT_DIR = Path("output/owt_bpe")
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE on {INPUT_PATH} (vocab_size={VOCAB_SIZE})")
    print(f"File size: {INPUT_PATH.stat().st_size / 1e9:.2f} GB")

    tracemalloc.start()
    start_time = time.time()

    vocab, merges = train_bpe(
        input_path=INPUT_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    hours = elapsed / 3600
    print(f"\nTraining completed in {elapsed:.1f}s ({hours:.3f} hours)")
    print(f"Peak memory: {peak_mem / 1e6:.1f} MB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token!r} ({len(longest_token)} bytes)")
    try:
        print(f"  Decoded: {longest_token.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("  (not valid UTF-8)")

    # Serialize vocab: {id: hex-encoded bytes}
    vocab_serializable = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in vocab.items()
    }
    vocab_path = OUTPUT_DIR / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_serializable, f, indent=2)
    print(f"Vocab saved to {vocab_path}")

    # Serialize merges: list of [hex1, hex2]
    merges_serializable = [
        [t1.hex(), t2.hex()] for t1, t2 in merges
    ]
    merges_path = OUTPUT_DIR / "merges.json"
    with open(merges_path, "w") as f:
        json.dump(merges_serializable, f, indent=2)
    print(f"Merges saved to {merges_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = OUTPUT_DIR / "profile.prof"

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    profiler.dump_stats(str(profile_path))
    print(f"\nProfile saved to {profile_path}")

    print("\n" + "=" * 80)
    print("PROFILE: Top 30 by cumulative time")
    print("=" * 80)
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("=" * 80)
    print("PROFILE: Top 30 by total (self) time")
    print("=" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(30)