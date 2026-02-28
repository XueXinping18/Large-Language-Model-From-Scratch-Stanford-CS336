from __future__ import annotations

from abc import ABC, abstractmethod


class PairCounter(ABC):
    """Interface for tracking (pair -> count) during BPE training."""

    @abstractmethod
    def get_max(self) -> tuple[bytes, bytes] | None:
        """Return the pair with the highest count.
        Tie-break: prefer lexicographically greater pair.
        Return None if empty."""

    @abstractmethod
    def update(self, pair: tuple[bytes, bytes], delta: int) -> None:
        """Add delta to the count of pair.
        If the resulting count is <= 0, the pair should be removed."""

    @abstractmethod
    def get_count(self, pair: tuple[bytes, bytes]) -> int:
        """Return the count for a pair, or 0 if not present."""

    @abstractmethod
    def __contains__(self, pair: tuple[bytes, bytes]) -> bool:
        """Check if a pair exists with count > 0."""


class NaivePairCounter(PairCounter):
    """Dict-backed pair counter. O(n) get_max, O(1) update."""

    def __init__(self) -> None:
        self._counts: dict[tuple[bytes, bytes], int] = {}

    def get_max(self) -> tuple[bytes, bytes] | None:
        if not self._counts:
            return None
        # Find pair with highest count; tie-break by lexicographically greater pair
        return max(self._counts, key=lambda p: (self._counts[p], p))

    def get_count(self, pair: tuple[bytes, bytes]) -> int:
        return self._counts.get(pair, 0)

    def update(self, pair: tuple[bytes, bytes], delta: int) -> None:
        new_count = self._counts.get(pair, 0) + delta
        if new_count <= 0:
            self._counts.pop(pair, None)
        else:
            self._counts[pair] = new_count

    def __contains__(self, pair: tuple[bytes, bytes]) -> bool:
        return pair in self._counts


class HeapPairCounter(PairCounter):
    """Augmented max-heap with index map. O(1) get_max, O(log n) update.
    No lazy deletion — heap always reflects true state.

    Each heap entry is a list [count, pair] so counts can be mutated in-place.
    Ordering: higher count wins; ties broken by lexicographically greater pair.
    """

    def __init__(self) -> None:
        # Each entry: [count, pair].  Using a list so count is mutable.
        self._heap: list[list[int | tuple[bytes, bytes]]] = []
        self._index: dict[tuple[bytes, bytes], int] = {}

    # -- public API --

    def get_max(self) -> tuple[bytes, bytes] | None:
        if not self._heap:
            return None
        return self._heap[0][1]  # type: ignore[return-value]

    def get_count(self, pair: tuple[bytes, bytes]) -> int:
        idx = self._index.get(pair)
        if idx is None:
            return 0
        return self._heap[idx][0]  # type: ignore[return-value]

    def update(self, pair: tuple[bytes, bytes], delta: int) -> None:
        if pair in self._index:
            idx = self._index[pair]
            old_count = self._heap[idx][0]
            new_count = old_count + delta  # type: ignore[operator]
            if new_count <= 0:
                self._delete(idx)
            else:
                self._heap[idx][0] = new_count
                if new_count > old_count:  # type: ignore[operator]
                    self._sift_up(idx)
                else:
                    self._sift_down(idx)
        else:
            if delta <= 0:
                return
            # Insert new entry at the end, then sift up
            idx = len(self._heap)
            self._heap.append([delta, pair])
            self._index[pair] = idx
            self._sift_up(idx)

    def __contains__(self, pair: tuple[bytes, bytes]) -> bool:
        return pair in self._index

    # -- internal helpers --

    def _key(self, idx: int) -> tuple[int, tuple[bytes, bytes]]:
        """Comparison key for heap entry at idx: (count, pair)."""
        entry = self._heap[idx]
        return (entry[0], entry[1])  # type: ignore[return-value]

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index[self._heap[i][1]] = i  # type: ignore[index]
        self._index[self._heap[j][1]] = j  # type: ignore[index]

    def _sift_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) >> 1
            if self._key(idx) > self._key(parent):
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        n = len(self._heap)
        while True:
            largest = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            if left < n and self._key(left) > self._key(largest):
                largest = left
            if right < n and self._key(right) > self._key(largest):
                largest = right
            if largest == idx:
                break
            self._swap(idx, largest)
            idx = largest

    def _delete(self, idx: int) -> None:
        """Remove the entry at idx from the heap."""
        pair = self._heap[idx][1]
        last = len(self._heap) - 1
        if idx == last:
            self._heap.pop()
            del self._index[pair]  # type: ignore[arg-type]
            return
        # Swap with last, pop last, then restore heap property
        self._swap(idx, last)
        self._heap.pop()
        del self._index[pair]  # type: ignore[arg-type]
        # The element now at idx could need to go up or down
        self._sift_up(idx)
        self._sift_down(idx)
