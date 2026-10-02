"""FAISS index wrapper for hard negative mining."""

import numpy as np
import faiss

# FAISS ships its own vendored OpenMP runtime; when its multi-threaded index
# ops (add/search) run from the miner thread concurrently with torch's OpenMP
# on ROCm, the two runtimes clash and segfault (Signal 11) mid-mining. Mining is
# off the training critical path, so pinning FAISS to a single OpenMP thread
# removes the clash with no effect on results (identical indices/neighbours).
faiss.omp_set_num_threads(1)


class FAISSIndex:
    """Wrapper around FAISS IndexFlatIP for cosine similarity search.

    All embeddings should be L2-normalized before adding, so that
    inner product = cosine similarity.

    Args:
        dim: int — embedding dimension
        use_gpu: bool — whether to use GPU FAISS (if available)
    """

    def __init__(self, dim, use_gpu=False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

        if use_gpu:
            # No try/except — if the user asked for GPU FAISS and it fails,
            # crash loudly so they know they're not getting GPU acceleration.
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add(self, embeddings):
        """Add embeddings to the index.

        Args:
            embeddings: np.ndarray (N, dim) — L2-normalized
        """
        self.index.add(embeddings.astype(np.float32))

    def search(self, queries, k):
        """Search for k nearest neighbors.

        Args:
            queries: np.ndarray (N, dim) — L2-normalized
            k: int — number of neighbors

        Returns:
            distances: np.ndarray (N, k) — similarity scores
            indices: np.ndarray (N, k) — neighbor indices
        """
        return self.index.search(queries.astype(np.float32), k)

    def reset(self):
        """Clear all embeddings from the index."""
        self.index.reset()

    @property
    def ntotal(self):
        return self.index.ntotal
