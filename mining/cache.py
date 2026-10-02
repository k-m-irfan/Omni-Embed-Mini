"""Thread-safe cache for hard negative mining results."""

import threading


class NegativeCache:
    """Thread-safe dictionary mapping sample IDs to hard negative IDs.

    The miner writes to this cache, and the collator reads from it.
    All operations are guarded by a single lock.
    """

    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def put(self, sample_id, negative_ids):
        """Store hard negative IDs for a sample.

        Args:
            sample_id: int
            negative_ids: list of int
        """
        with self._lock:
            self._data[sample_id] = list(negative_ids)

    def get(self, sample_id, n=5):
        """Get hard negative IDs for a sample.

        Args:
            sample_id: int
            n: int — max number of negatives to return

        Returns:
            list of int, or None if not in cache
        """
        with self._lock:
            negs = self._data.get(sample_id)
        if negs is None:
            return None
        return negs[:n]

    def clear(self):
        """Clear all cached negatives."""
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)

    def hit_rate(self, sample_ids):
        """Calculate cache hit rate for a set of sample IDs.

        Args:
            sample_ids: iterable of int

        Returns:
            float — fraction of IDs present in cache
        """
        with self._lock:
            hits = sum(1 for sid in sample_ids if sid in self._data)
        return hits / max(len(list(sample_ids)), 1)
