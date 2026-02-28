from __future__ import annotations

import logging
import os
import time

from cs336_basics.tokenizer.pair_counter import HeapPairCounter, PairCounter
from cs336_basics.tokenizer.pretokenize import count_pretokens_parallel

DEBUG = False
log = logging.getLogger(__name__)

def _build_initial_vocab() -> dict[int, bytes]:
    """Build the base vocabulary: IDs 0-255 mapped to single bytes."""
    return {i: bytes([i]) for i in range(256)}


def _words_to_byte_sequences(
    pretoken_counts: dict[str, int],
) -> tuple[dict[str, list[bytes]], dict[str, int]]:
    """Convert pre-token strings to mutable byte-token lists.

    Args:
        pretoken_counts: Mapping from pre-token string to its frequency.

    Returns:
        word_tokens: dict mapping each pre-token to its current list of byte tokens.
                     e.g. "cat" -> [b'c', b'a', b't']
        word_counts: dict mapping each pre-token to its frequency (same as input).
    """
    word_tokens: dict[str, list[bytes]] = {}
    word_counts: dict[str, int] = {}
    for word, count in pretoken_counts.items():
        word_bytes = word.encode("utf-8")
        word_tokens[word] = [bytes([b]) for b in word_bytes]
        word_counts[word] = count
    return word_tokens, word_counts


def _count_all_pairs(
    word_tokens: dict[str, list[bytes]],
    word_counts: dict[str, int],
    pair_counter: PairCounter,
) -> dict[tuple[bytes, bytes], dict[str, int]]:
    """Populate pair_counter with initial adjacent-pair counts and build
    the reverse index pair_to_words.

    For each word, iterate over consecutive token pairs and:
      - Call pair_counter.update(pair, word_count)
      - Record the occurrence count per word in pair_to_words

    Returns:
        pair_to_words: dict mapping each pair -> {word: occurrence_count_in_word}.
    """
    pair_to_words: dict[tuple[bytes, bytes], dict[str, int]] = {}
    for word, tokens in word_tokens.items():
        count = word_counts[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counter.update(pair, count)
            _p2w_increment(pair_to_words, pair, word)

    if DEBUG:
        for pair in list(pair_to_words):
            _assert_pair_count_invariant(pair, pair_counter, pair_to_words, word_counts)
        log.info("Initial pair count invariant OK for all %d pairs", len(pair_to_words))

    return pair_to_words


def _p2w_decrement(
    pair_to_words: dict[tuple[bytes, bytes], dict[str, int]],
    pair: tuple[bytes, bytes],
    word: str,
) -> None:
    """Decrement pair_to_words[pair][word] by 1. Remove entries that hit 0."""
    if pair not in pair_to_words:
        return
    inner = pair_to_words[pair]
    if word not in inner:
        return
    inner[word] -= 1
    if inner[word] <= 0:
        del inner[word]
        if not inner:
            del pair_to_words[pair]


def _p2w_increment(
    pair_to_words: dict[tuple[bytes, bytes], dict[str, int]],
    pair: tuple[bytes, bytes],
    word: str,
) -> None:
    """Increment pair_to_words[pair][word] by 1."""
    if pair not in pair_to_words:
        pair_to_words[pair] = {}
    pair_to_words[pair][word] = pair_to_words[pair].get(word, 0) + 1


def _assert_pair_count_invariant(
    pair: tuple[bytes, bytes],
    pair_counter: PairCounter,
    pair_to_words: dict[tuple[bytes, bytes], dict[str, int]],
    word_counts: dict[str, int],
) -> None:
    """Assert: pair_counter[pair] == sum(p2w[pair][w] * word_counts[w] for w in p2w[pair]).

    Only called when DEBUG is True.
    """
    if pair not in pair_to_words:
        assert pair not in pair_counter, (
            f"Invariant violated: pair {pair} in pair_counter but not in pair_to_words"
        )
        return
    expected = sum(
        pair_to_words[pair][w] * word_counts[w]
        for w in pair_to_words[pair]
    )
    actual = pair_counter.get_count(pair)
    assert actual == expected, (
        f"Invariant violated for pair {pair}: "
        f"pair_counter={actual}, sum(p2w * freq)={expected}, "
        f"p2w detail={dict(pair_to_words[pair])}"
    )


def _merge_pair_in_words(
    pair: tuple[bytes, bytes],
    merged: bytes,
    word_tokens: dict[str, list[bytes]],
    word_counts: dict[str, int],
    pair_counter: PairCounter,
    pair_to_words: dict[tuple[bytes, bytes], dict[str, int]],
) -> None:
    """Merge all occurrences of `pair` across all words that contain it after we found which pair occurred most frequently.

    For each affected word:
      1. Walk through the token list left-to-right.
      2. When you find (token1, token2) == pair:
         - Decrement counts for destroyed neighbor pairs.
         - Decrement count for the pair itself.
         - Replace token1, token2 with merged token.
         - Increment counts for newly created neighbor pairs.
      3. Update pair_to_words mappings accordingly.

    Args:
        pair: The (token1, token2) pair being merged.
        merged: The new token (token1 + token2).
        word_tokens: Mutable token lists for each word.
        word_counts: Frequency of each word.
        pair_counter: The pair counter to update.
        pair_to_words: Reverse index from pair -> {word: occurrence_count}.
    """
    token1, token2 = pair
    # Copy keys — we'll mutate pair_to_words[pair] during iteration
    affected_words = list(pair_to_words.get(pair, {}).keys())

    if DEBUG:
        log.info(
            "MERGE %s + %s -> %s | affecting %d words",
            token1, token2, merged, len(affected_words),
        )
        modified_pairs: set[tuple[bytes, bytes]] = set()

    for word in affected_words:
        tokens = word_tokens[word]
        count = word_counts[word]
        new_tokens: list[bytes] = []
        i = 0

        if DEBUG:
            log.info("  word=%r (freq=%d) tokens_before=%s", word, count, tokens)

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == token1
                and tokens[i + 1] == token2
            ):
                # --- Decrement old pairs ---
                pair_counter.update(pair, -count)
                _p2w_decrement(pair_to_words, pair, word)
                if DEBUG:
                    modified_pairs.add(pair)

                if new_tokens:
                    left_pair = (new_tokens[-1], token1)
                    if DEBUG:
                        log.info("    i=%d destroy left_pair=%s (-%d)", i, left_pair, count)
                        modified_pairs.add(left_pair)
                    pair_counter.update(left_pair, -count)
                    _p2w_decrement(pair_to_words, left_pair, word)

                if i + 2 < len(tokens):
                    right_pair = (token2, tokens[i + 2])
                    if DEBUG:
                        log.info("    i=%d destroy right_pair=%s (-%d)", i, right_pair, count)
                        modified_pairs.add(right_pair)
                    pair_counter.update(right_pair, -count)
                    _p2w_decrement(pair_to_words, right_pair, word)

                # --- Replace with merged token ---
                new_tokens.append(merged)

                # --- Increment new pairs ---
                if len(new_tokens) >= 2:
                    new_left_pair = (new_tokens[-2], merged)
                    if DEBUG:
                        log.info("    i=%d create new_left_pair=%s (+%d)", i, new_left_pair, count)
                        modified_pairs.add(new_left_pair)
                    pair_counter.update(new_left_pair, count)
                    _p2w_increment(pair_to_words, new_left_pair, word)

                if i + 2 < len(tokens):
                    new_right_pair = (merged, tokens[i + 2])
                    if DEBUG:
                        log.info("    i=%d create new_right_pair=%s (+%d)", i, new_right_pair, count)
                        modified_pairs.add(new_right_pair)
                    pair_counter.update(new_right_pair, count)
                    _p2w_increment(pair_to_words, new_right_pair, word)

                i += 2  # skip both tokens of the merged pair
            else:
                new_tokens.append(tokens[i])
                i += 1

        word_tokens[word] = new_tokens

        if DEBUG:
            log.info("  word=%r tokens_after=%s", word, new_tokens)

    if DEBUG:
        for p in modified_pairs:
            _assert_pair_count_invariant(p, pair_counter, pair_to_words, word_counts)
        log.info("  invariant OK for %d modified pairs", len(modified_pairs))


def _add_special_tokens(
    vocab: dict[int, bytes],
    special_tokens: list[str],
    next_id: int,
) -> None:
    """Assign IDs to special tokens, starting from next_id."""
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to a text file with training data.
        vocab_size: Maximum final vocabulary size.
        special_tokens: Special tokens to add to vocabulary (don't affect merging).

    Returns:
        vocab: dict[int, bytes] — the tokenizer vocabulary.
        merges: list[tuple[bytes, bytes]] — ordered list of BPE merges.
    """
    # Step 1: initial vocab
    vocab = _build_initial_vocab()
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    if num_merges <= 0:
        _add_special_tokens(vocab, special_tokens, 256)
        return vocab, merges

    # Step 2 & 3: pre-tokenize and count
    t0 = time.time()
    print(f"[train_bpe] Step 2-3: Pre-tokenizing {input_path} ...")
    pretoken_counts = count_pretokens_parallel(str(input_path), special_tokens)
    word_tokens, word_counts = _words_to_byte_sequences(pretoken_counts)
    print(f"[train_bpe] Step 2-3 done: {len(word_tokens)} unique pre-tokens ({time.time() - t0:.1f}s)")

    # Step 4: count initial pairs and build reverse index
    t1 = time.time()
    print("[train_bpe] Step 4: Counting initial pairs ...")
    pair_counter: PairCounter = HeapPairCounter()
    pair_to_words = _count_all_pairs(word_tokens, word_counts, pair_counter)
    print(f"[train_bpe] Step 4 done: {len(pair_to_words)} unique pairs ({time.time() - t1:.1f}s)")

    # Step 5: merge loop
    t2 = time.time()
    print(f"[train_bpe] Step 5: Running {num_merges} merges ...")
    next_id = 256
    for merge_i in range(num_merges):
        best_pair = pair_counter.get_max()
        if best_pair is None:
            break

        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        merges.append(best_pair)

        if DEBUG:
            log.info(
                "merge #%d: %s + %s -> %s (id=%d, count=%d)",
                merge_i, best_pair[0], best_pair[1], merged,
                next_id, pair_counter.get_count(best_pair),
            )

        _merge_pair_in_words(
            best_pair, merged, word_tokens, word_counts,
            pair_counter, pair_to_words,
        )
        next_id += 1

        if (merge_i + 1) % 500 == 0 or merge_i + 1 == num_merges:
            print(f"[train_bpe]   merge {merge_i + 1}/{num_merges} ({time.time() - t2:.1f}s)")

    print(f"[train_bpe] Step 5 done: {len(merges)} merges ({time.time() - t2:.1f}s)")

    # Step 6: add special tokens to vocab
    _add_special_tokens(vocab, special_tokens, next_id)
    print(f"[train_bpe] Total time: {time.time() - t0:.1f}s")

    return vocab, merges
