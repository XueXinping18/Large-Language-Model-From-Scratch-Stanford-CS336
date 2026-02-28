from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import regex

from cs336_basics.tokenizer.pretokenize import GPT2_PRETOKENIZE_PATTERN


class Tokenizer:
    """BPE Tokenizer that encodes text to token IDs and decodes back."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Reverse mapping: bytes -> token ID
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Merge priority: (token1, token2) -> rank (lower = merge first)
        self.merge_priority: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # Special token handling: add to vocab if not already present
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.bytes_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = st_bytes
                self.bytes_to_id[st_bytes] = new_id

        # Build a regex pattern that matches any special token
        # Sort by length descending so longer tokens match first
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_tokens]
            self._special_token_pattern: re.Pattern | None = re.compile(
                "|".join(escaped)
            )
        else:
            self._special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Construct a Tokenizer from serialized vocab and merges files."""
        import json

        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {
            int(k): bytes.fromhex(v) for k, v in raw_vocab.items()
        }

        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges: list[tuple[bytes, bytes]] = [
            (bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        1. Split text on special tokens (longest match first).
        2. For special token segments: map directly to ID.
        3. For regular text segments:
           a. Pre-tokenize with GPT-2 regex.
           b. Convert each pre-token to single-byte tokens.
           c. Iteratively apply merges (lowest rank first) until no more apply.
           d. Map resulting tokens to IDs.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        ids: list[int] = []

        # Split text into segments: (text, is_special) pairs
        if self._special_token_pattern:
            segments: list[tuple[str, bool]] = []
            last_end = 0
            for m in self._special_token_pattern.finditer(text):
                if m.start() > last_end:
                    segments.append((text[last_end : m.start()], False))
                segments.append((m.group(), True))
                last_end = m.end()
            if last_end < len(text):
                segments.append((text[last_end:], False))
        else:
            segments = [(text, False)]

        for segment, is_special in segments:
            if is_special:
                ids.append(self.bytes_to_id[segment.encode("utf-8")])
            else:
                # Pre-tokenize with GPT-2 regex, then BPE-encode each pre-token
                for pretoken in regex.findall(GPT2_PRETOKENIZE_PATTERN, segment):
                    ids.extend(self._encode_chunk(pretoken.encode("utf-8")))

        return ids

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """Encode a single pre-token (as bytes) using the merge list.

        1. Start with a list of single-byte tokens.
        2. Repeatedly find the pair with the lowest merge rank and merge it.
        3. Stop when no more merges apply.

        Args:
            text_bytes: Bytes of a single pre-token.

        Returns:
            List of token IDs for this pre-token.
        """
        if len(text_bytes) == 0:
            return []

        tokens: list[bytes] = [bytes([b]) for b in text_bytes]

        while len(tokens) >= 2:
            # Find the adjacent pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of best_pair left to right
            merged = best_pair[0] + best_pair[1]
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.bytes_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, lazily yield token IDs.

        Args:
            iterable: An iterable of strings (e.g., a file handle).

        Yields:
            Token IDs one at a time.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.

        1. Look up each ID in vocab to get bytes.
        2. Concatenate all bytes.
        3. Decode as UTF-8.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")