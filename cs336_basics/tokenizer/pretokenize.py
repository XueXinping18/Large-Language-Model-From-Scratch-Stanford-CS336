from __future__ import annotations

import os
from collections import Counter
import regex

# GPT-2 pre-tokenization pattern
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_segment(segment: str) -> Counter[str]:
    """Pre-tokenize a single segment (no special tokens present) using the
    GPT-2 regex pattern.

    Args:
        segment: A string with no special tokens in it.

    Returns:
        Counter mapping each pre-token match to its frequency.
    """
    counts: Counter[str] = Counter()
    for match in regex.finditer(GPT2_PRETOKENIZE_PATTERN, segment):
        counts[match.group()] += 1
    return counts


def pretokenize_chunk(
    text: str,
    special_tokens: list[str],
    byte_start: int = 0,
    byte_end: int = 0,
) -> Counter[str]:
    """Pre-tokenize a chunk of text and return pre-token counts.

    1. Split text on special tokens (so no merging across boundaries).
    2. For each segment, call pretokenize_segment.
    3. Merge counts.

    Args:
        text: A string chunk (already decoded from bytes).
        special_tokens: List of special token strings to split on.
        byte_start: Starting byte offset of this chunk in the original file.
        byte_end: Ending byte offset of this chunk in the original file.

    Returns:
        Counter mapping each pre-token string to its frequency.
    """
    pid = os.getpid()
    print(f"[PID {pid}] pretokenize_chunk: bytes [{byte_start}, {byte_end}), {len(text)} chars")

    # Split on special tokens
    if special_tokens:
        split_pattern = "|".join(regex.escape(t) for t in special_tokens)
        segments = regex.split(split_pattern, text)
    else:
        segments = [text]

    print(f"[PID {pid}] split into {len(segments)} segments")

    counts: Counter[str] = Counter()
    for segment in segments:
        if segment:  # skip empty strings from split
            counts += pretokenize_segment(segment)

    return counts


def _process_chunk(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    """Worker function for multiprocessing. Reads a byte range from the file,
    decodes to string, and pre-tokenizes it."""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="ignore")
    return pretokenize_chunk(text, special_tokens, byte_start=start, byte_end=end)


def count_pretokens_parallel(
    input_path: str,
    special_tokens: list[str],
    num_workers: int | None = None,
    target_chunk_size: int = 256 * 1024 * 1024,  # 256 MB
) -> Counter[str]:
    """Read a file, chunk it using find_chunk_boundaries, and count
    pre-tokens in parallel.

    The number of chunks is decoupled from the number of workers so that
    large files are split into many small chunks.  Only ``num_workers``
    chunks are held in memory at any time.

    Args:
        input_path: Path to the input text file.
        special_tokens: Special tokens to split on.
        num_workers: Number of parallel workers.
        target_chunk_size: Approximate byte size per chunk (default 256 MB).

    Returns:
        Global Counter of pre-token frequencies.
    """
    from multiprocessing import Pool

    from cs336_basics.tokenizer.chunk_utils import find_chunk_boundaries

    if num_workers is None:
        num_workers = 12

    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    file_size = os.path.getsize(input_path)
    num_chunks = max(num_workers, file_size // target_chunk_size + 1)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)

    print(f"[count_pretokens_parallel] {len(boundaries) - 1} chunks, {num_workers} workers")
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        print(f"  chunk {i}: bytes [{start}, {end}) = {end - start} bytes")

    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    global_counts: Counter[str] = Counter()
    with Pool(processes=num_workers) as pool:
        for local_counts in pool.imap_unordered(_process_chunk, chunk_args):
            global_counts += local_counts

    return global_counts
