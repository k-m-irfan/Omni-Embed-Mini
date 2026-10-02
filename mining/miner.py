"""Hybrid hard negative mining for contrastive training.

Two mining strategies run in a single background thread:
  1. Text mining (cheap, frequent): embed captions via text-only encoder,
     find semantically similar captions via FAISS. Runs every N steps.
  2. Media mining (expensive, infrequent): embed actual media samples
     (audio, image, video) through full encoder pipeline, find perceptually
     similar samples. Cycles through all media modalities. Runs every M steps.

Results from both strategies are merged into a shared thread-safe cache.
The collator reads from this cache during batch assembly.
"""

import os
import threading
import time
import numpy as np
import torch
from tqdm import tqdm

from .index import FAISSIndex
from .cache import NegativeCache


class HybridMiner:
    """Background thread for hybrid text + media hard negative mining.

    Text mining: embeds captions only (fast).
    Media mining: embeds actual audio/image/video through full model.
        Cycles through ALL media modalities each cycle — not just one.

    Args:
        dataset: ContrastiveOmniDataset
        text_index_size: int — captions per text cycle (default 50k)
        media_index_size: int — samples PER MODALITY per media cycle (default 5k)
        n_hard_negs: int — negatives per sample
        false_neg_threshold: float — filter pairs above this similarity
    """

    def __init__(
        self,
        dataset=None,
        text_encode_fn=None,
        media_encode_fn=None,
        text_index_size=50000,
        media_index_size=5000,
        n_hard_negs=5,
        false_neg_threshold=0.92,
    ):
        self.dataset = dataset
        self.text_encode_fn = text_encode_fn
        self.media_encode_fn = media_encode_fn
        self.text_index_size = text_index_size
        self.media_index_size = media_index_size
        self.n_hard_negs = n_hard_negs
        self.false_neg_threshold = false_neg_threshold

        self.cache = NegativeCache()
        self._thread = None
        self._stop_event = threading.Event()
        self._text_refresh = threading.Event()
        self._media_refresh = threading.Event()
        self._lock = threading.Lock()
        # Serializes the miner's GPU forward calls with the main
        # training thread's forward+backward. Required because
        # concurrent forwards through `vision_encoder.visual` corrupt
        # gradient-checkpointing recompute (FlashAttention dispatch
        # diverges between no-grad original forward and grad-enabled
        # recompute when another thread's ops are interleaved).
        # Main thread acquires for the whole step body; miner acquires
        # per encode-batch.
        self.forward_lock = threading.Lock()

        # Stats
        self.text_mine_count = 0
        self.media_mine_count = 0

        # Modality index — set externally from batch_sampler.modality_indices
        # (which already handles omni split classification)
        self._mod_indices = {}

    def set_modality_indices(self, modality_indices):
        """Set modality index from the batch sampler (already classifies omni samples)."""
        self._mod_indices = {mod: list(ids) for mod, ids in modality_indices.items()}
        print(f"[Miner] Modality index set: "
              + ", ".join(f"{m}={len(ids)}" for m, ids in sorted(self._mod_indices.items())))

    def set_text_encode_fn(self, fn):
        with self._lock:
            self.text_encode_fn = fn

    def set_media_encode_fn(self, fn):
        with self._lock:
            self.media_encode_fn = fn

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Miner] Background thread started")

    def stop(self):
        print("[Miner] Stopping background thread...")
        self._stop_event.set()
        self._text_refresh.set()
        self._media_refresh.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        print(f"[Miner] Stopped. Total: {self.text_mine_count} text cycles, "
              f"{self.media_mine_count} media cycles, "
              f"{len(self.cache)} cached entries")

    def request_text_refresh(self):
        self._text_refresh.set()

    def request_media_refresh(self):
        """Signal the miner to run media mining across ALL modalities."""
        self._media_refresh.set()

    def _run(self):
        # No try/except here. Errors propagate to threading.excepthook
        # (installed in train.py), which prints the traceback and SIGTERMs
        # the main process. We do NOT want silent recovery in the miner.
        while not self._stop_event.is_set():
            if self._text_refresh.wait(timeout=5):
                if self._stop_event.is_set():
                    break
                self._text_refresh.clear()
                self._mine_text()

            if self._media_refresh.is_set():
                self._media_refresh.clear()
                if self._stop_event.is_set():
                    break
                self._mine_media()

    # ── Text mining ──────────────────────────────────────────────

    @torch.no_grad()
    def _mine_text(self):
        t_start = time.time()
        with self._lock:
            if self.text_encode_fn is None or self.dataset is None:
                return

        captions = self.dataset.captions
        sample_ids = list(range(len(captions)))

        if len(captions) > self.text_index_size:
            indices = np.random.choice(len(captions), self.text_index_size, replace=False)
            captions = [captions[i] for i in indices]
            sample_ids = [sample_ids[i] for i in indices]

        valid = [(c, s) for c, s in zip(captions, sample_ids) if c.strip()]
        if not valid:
            return
        captions, sample_ids = zip(*valid)
        captions, sample_ids = list(captions), list(sample_ids)

        print(f"[Miner] Text mining: embedding {len(captions)} captions...")

        with self._lock:
            encode_fn = self.text_encode_fn

        embeddings = []
        # Embed batch size only affects throughput, not which hard negatives
        # are found (identical embeddings → identical FAISS results). Lower it
        # via OMNI_MINE_EMBED_BS if long captions exhaust memory in the
        # O(B·S²) attention.
        batch_size = int(os.environ.get("OMNI_MINE_EMBED_BS", "256"))
        pbar = tqdm(
            range(0, len(captions), batch_size),
            desc="[Miner] Text embed",
            unit="batch",
            leave=False,
        )
        for start in pbar:
            end = min(start + batch_size, len(captions))
            # Acquire forward_lock per batch so the main training thread
            # can interleave its forward+backward in between batches.
            with self.forward_lock:
                embs = encode_fn(captions[start:end])
            # .float() before .numpy(): numpy has no bf16 dtype, and the
            # encoder runs under bf16 autocast → silent crash without this.
            embeddings.append(embs.cpu().float().numpy())
            pbar.set_postfix(samples=end)
        pbar.close()

        all_embs = np.concatenate(embeddings, axis=0)
        all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
        embed_sec = time.time() - t_start

        print(f"[Miner] Text mining: building FAISS index ({len(all_embs)} vectors, "
              f"dim={all_embs.shape[1]})...")
        # Serialize FAISS index build/search with the main training thread.
        # FAISS runs its own native (OpenMP) threads; on ROCm it must not run
        # concurrently with torch on the main thread or the two runtimes race —
        # either segfaulting or deadlocking mid-mining. forward_lock (already
        # held for the per-batch embed above) makes the index step exclusive too.
        with self.forward_lock:
            n_new = self._index_and_cache(all_embs, sample_ids)

        total_sec = time.time() - t_start
        self.text_mine_count += 1
        print(f"[Miner] Text mining #{self.text_mine_count} complete: "
              f"{len(captions)} captions → {n_new} samples with hard negs | "
              f"embed {embed_sec:.1f}s, total {total_sec:.1f}s | "
              f"cache size: {len(self.cache)}")

    # ── Media mining ─────────────────────────────────────────────

    @torch.no_grad()
    def _mine_media(self):
        """Mine hard negatives from actual media — cycles through ALL modalities.

        For each modality (audio, image, video, visual_doc):
          1. Subsample to media_index_size
          2. Encode through full model pipeline
          3. Build FAISS index for that modality
          4. Find hard negatives within the same modality
          5. Write to shared cache
        """
        t_start = time.time()
        with self._lock:
            if self.media_encode_fn is None or self.dataset is None:
                return
            encode_fn = self.media_encode_fn

        modalities = sorted(self._mod_indices.keys())
        total_new = 0
        total_samples = 0

        print(f"[Miner] Media mining: cycling through {len(modalities)} modalities: "
              f"{modalities}")

        for mod in modalities:
            mod_ids = self._mod_indices[mod]
            if not mod_ids:
                continue

            # Subsample
            if len(mod_ids) > self.media_index_size:
                selected = np.random.choice(
                    mod_ids, self.media_index_size, replace=False
                ).tolist()
            else:
                selected = list(mod_ids)

            print(f"[Miner]   {mod}: encoding {len(selected)} samples...")

            # Encode in small batches with progress bar. No try/except —
            # an encode failure should crash the whole run via the global
            # threading.excepthook so we see what broke instead of silently
            # corrupting the negatives cache.
            embeddings = []
            batch_size = 16
            pbar = tqdm(
                range(0, len(selected), batch_size),
                desc=f"[Miner] {mod} embed",
                unit="batch",
                leave=False,
            )
            for start in pbar:
                if self._stop_event.is_set():
                    pbar.close()
                    return
                end = min(start + batch_size, len(selected))
                batch_ids = selected[start:end]
                # forward_lock — see _mine_text for the explanation.
                with self.forward_lock:
                    embs = encode_fn(batch_ids)
                # bf16 autocast → numpy needs fp32 (see _mine_text).
                embeddings.append(embs.cpu().float().numpy())
                pbar.set_postfix(ok=sum(e.shape[0] for e in embeddings))
            pbar.close()

            if not embeddings:
                raise RuntimeError(
                    f"[Miner] {mod}: encoded zero embeddings — selected "
                    f"{len(selected)} ids but none made it through. "
                    "Check encode_fn, modality_indices, or dataset."
                )

            all_embs = np.concatenate(embeddings, axis=0)
            all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
            sample_ids = selected[:all_embs.shape[0]]

            print(f"[Miner]   {mod}: FAISS index ({len(all_embs)} vectors)...")
            # Serialize FAISS with training — see _mine_text for the rationale.
            with self.forward_lock:
                n_new = self._index_and_cache(all_embs, sample_ids)

            total_new += n_new
            total_samples += len(sample_ids)
            print(f"[Miner]   {mod}: {n_new} samples with hard negs")

        total_sec = time.time() - t_start
        self.media_mine_count += 1
        print(f"[Miner] Media mining #{self.media_mine_count} complete: "
              f"{total_samples} total samples → {total_new} hard negs across "
              f"{len(modalities)} modalities | total {total_sec:.1f}s | "
              f"cache size: {len(self.cache)}")

    # ── FAISS index + cache ──────────────────────────────────────

    def _index_and_cache(self, embeddings, sample_ids):
        """Build FAISS index, search, filter, write to cache."""
        dim = embeddings.shape[1]
        index = FAISSIndex(dim)
        index.add(embeddings)

        k = self.n_hard_negs + 10
        distances, indices = index.search(embeddings, k)

        n_new = 0
        for i in range(len(sample_ids)):
            sid = sample_ids[i]
            negatives = []
            for j in range(k):
                neighbor_idx = indices[i, j]
                if neighbor_idx == i:
                    continue
                sim = distances[i, j]
                if sim > self.false_neg_threshold:
                    continue
                if neighbor_idx < len(sample_ids):
                    negatives.append(sample_ids[neighbor_idx])
                if len(negatives) >= self.n_hard_negs:
                    break

            if negatives:
                self.cache.put(sid, negatives)
                n_new += 1

        return n_new

    def get_hard_negatives(self, sample_id, n=5):
        return self.cache.get(sample_id, n)
