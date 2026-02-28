from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import pytest

from cs336_basics.tokenizer.pretokenize import (
    GPT2_PRETOKENIZE_PATTERN,
    _process_chunk,
    count_pretokens_parallel,
    pretokenize_chunk,
    pretokenize_segment,
)

from .common import FIXTURES_PATH


# ---------------------------------------------------------------------------
# pretokenize_segment
# ---------------------------------------------------------------------------
class TestPretokenizeSegment:
    def test_empty_string(self):
        assert pretokenize_segment("") == Counter()

    def test_single_word(self):
        result = pretokenize_segment("hello")
        assert result == Counter({"hello": 1})

    def test_multiple_words(self):
        result = pretokenize_segment("hello world")
        # GPT-2 pattern: leading space attaches to the next word
        assert result == Counter({"hello": 1, " world": 1})

    def test_repeated_words(self):
        result = pretokenize_segment("cat cat cat")
        assert result == Counter({"cat": 1, " cat": 2})

    def test_punctuation(self):
        result = pretokenize_segment("Hello, world!")
        assert "Hello" in result
        assert "," in result
        assert " world" in result
        assert "!" in result

    def test_contractions(self):
        result = pretokenize_segment("I'm don't she'll we've you're")
        # GPT-2 pattern splits contractions: 'm, n't -> 't, 'll, 've, 're
        assert "'m" in result
        assert "'t" in result
        assert "'ll" in result
        assert "'ve" in result
        assert "'re" in result

    def test_numbers(self):
        result = pretokenize_segment("test 123 456")
        assert " 123" in result
        assert " 456" in result

    def test_whitespace_preserved(self):
        # Trailing whitespace (not followed by non-whitespace) is its own token
        result = pretokenize_segment("hello   ")
        assert "hello" in result
        assert "   " in result

    def test_newlines(self):
        result = pretokenize_segment("line1\nline2")
        assert "\n" in result

    def test_unicode(self):
        result = pretokenize_segment("café résumé")
        assert "café" in result or "caf" in result  # depends on regex L match
        assert result.total() > 0

    def test_returns_counter_type(self):
        result = pretokenize_segment("hello")
        assert isinstance(result, Counter)


# ---------------------------------------------------------------------------
# pretokenize_chunk
# ---------------------------------------------------------------------------
class TestPretokenizeChunk:
    def test_no_special_tokens(self):
        result = pretokenize_chunk("hello world", special_tokens=[])
        assert result == Counter({"hello": 1, " world": 1})

    def test_empty_special_tokens_list(self):
        result = pretokenize_chunk("hello world", special_tokens=[])
        expected = pretokenize_segment("hello world")
        assert result == expected

    def test_split_on_special_token(self):
        text = "hello<|endoftext|>world"
        result = pretokenize_chunk(text, special_tokens=["<|endoftext|>"])
        assert "hello" in result
        assert "world" in result
        # The special token itself should NOT appear as a pre-token
        assert "<|endoftext|>" not in result

    def test_multiple_special_tokens(self):
        text = "aaa<|endoftext|>bbb<|pad|>ccc"
        result = pretokenize_chunk(
            text, special_tokens=["<|endoftext|>", "<|pad|>"]
        )
        assert "aaa" in result
        assert "bbb" in result
        assert "ccc" in result
        assert "<|endoftext|>" not in result
        assert "<|pad|>" not in result

    def test_adjacent_special_tokens(self):
        text = "hello<|endoftext|><|endoftext|>world"
        result = pretokenize_chunk(text, special_tokens=["<|endoftext|>"])
        assert "hello" in result
        assert "world" in result

    def test_special_token_at_start(self):
        text = "<|endoftext|>hello world"
        result = pretokenize_chunk(text, special_tokens=["<|endoftext|>"])
        assert "hello" in result
        assert " world" in result

    def test_special_token_at_end(self):
        text = "hello world<|endoftext|>"
        result = pretokenize_chunk(text, special_tokens=["<|endoftext|>"])
        assert "hello" in result
        assert " world" in result

    def test_empty_text(self):
        result = pretokenize_chunk("", special_tokens=["<|endoftext|>"])
        assert result == Counter()

    def test_only_special_tokens(self):
        text = "<|endoftext|><|endoftext|>"
        result = pretokenize_chunk(text, special_tokens=["<|endoftext|>"])
        assert result == Counter()

    def test_byte_offsets_dont_affect_result(self):
        """byte_start and byte_end are for logging only; result should be the same."""
        text = "hello world"
        r1 = pretokenize_chunk(text, special_tokens=[], byte_start=0, byte_end=11)
        r2 = pretokenize_chunk(text, special_tokens=[], byte_start=100, byte_end=200)
        assert r1 == r2


# ---------------------------------------------------------------------------
# _process_chunk  (reads a byte range from a file)
# ---------------------------------------------------------------------------
class TestProcessChunk:
    def test_reads_full_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        result = _process_chunk((str(p), 0, p.stat().st_size, []))
        assert result == Counter({"hello": 1, " world": 1})

    def test_reads_byte_range(self, tmp_path):
        p = tmp_path / "test.txt"
        content = "aaaa bbbb cccc"
        p.write_text(content, encoding="utf-8")
        # Read only "bbbb cccc" (bytes 5 through end)
        result = _process_chunk((str(p), 5, len(content.encode()), []))
        assert "bbbb" in result
        assert " cccc" in result
        assert "aaaa" not in result

    def test_with_special_tokens(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello<|endoftext|>world", encoding="utf-8")
        result = _process_chunk(
            (str(p), 0, p.stat().st_size, ["<|endoftext|>"])
        )
        assert "hello" in result
        assert "world" in result
        assert "<|endoftext|>" not in result

    def test_utf8_decoding(self, tmp_path):
        p = tmp_path / "test.txt"
        text = "café"
        p.write_bytes(text.encode("utf-8"))
        result = _process_chunk((str(p), 0, p.stat().st_size, []))
        assert result.total() > 0


# ---------------------------------------------------------------------------
# count_pretokens_parallel
# ---------------------------------------------------------------------------
class TestCountPretokensParallel:
    def test_small_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world hello world", encoding="utf-8")
        result = count_pretokens_parallel(str(p), special_tokens=[], num_workers=1)
        assert result["hello"] == 1
        assert " world" in result
        assert " hello" in result

    def test_with_special_tokens(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text(
            "aaa<|endoftext|>bbb<|endoftext|>ccc", encoding="utf-8"
        )
        result = count_pretokens_parallel(
            str(p), special_tokens=["<|endoftext|>"], num_workers=1
        )
        assert "aaa" in result
        assert "bbb" in result
        assert "ccc" in result

    def test_multiple_workers_same_result(self, tmp_path):
        p = tmp_path / "test.txt"
        text = "the quick brown fox jumps over the lazy dog\n" * 100
        p.write_text(text, encoding="utf-8")
        r1 = count_pretokens_parallel(str(p), special_tokens=[], num_workers=1)
        r2 = count_pretokens_parallel(str(p), special_tokens=[], num_workers=4)
        assert r1 == r2

    def test_fixture_corpus(self):
        """Smoke test on the actual fixture corpus."""
        corpus_path = FIXTURES_PATH / "corpus.en"
        if not corpus_path.exists():
            pytest.skip("corpus.en fixture not found")
        result = count_pretokens_parallel(
            str(corpus_path),
            special_tokens=["<|endoftext|>"],
            num_workers=2,
        )
        assert isinstance(result, Counter)
        assert result.total() > 0