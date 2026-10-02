"""Teacher embedding cache for self-distillation training.

The teacher is the frozen backbone encoding the "cascaded" caption text
for each sample. Because the backbone is permanently frozen (no LoRA, no
full-tune), the teacher embedding is a deterministic function of the
sample — recomputing it every step is wasted work. We precompute once,
persist to disk, and look up per-step.

Cache identity
--------------
Cache is keyed by BACKBONE NAME only (not data_pct). Each entry is keyed
by a STABLE sample key of the form "<split_name>:<original_local_idx>",
where original_local_idx is the sample's position in the full (pre-stride)
split. Stride sampling guarantees that a 20% run's sample set is a subset
of a 100% run's sample set under the same stride rule, so caches built at
different data_pct values are compatible and can be extended incrementally.

Layout
------
    cache/teacher/<backbone_safe>/
        embeddings.pt        # (N, hidden_size) float32 CPU tensor
        keys.json            # list of N stable-key strings (row-aligned)
        meta.json            # schema_version, backbone, hidden_size, num_samples

On startup we load whatever is there, diff the current dataset's stable
keys against the cached set, compute the MISSING ones only, append to
both tensors, and persist. 100% → 20% → 50% all share one cache file.
"""

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm


def cache_dir_for(backbone_name, root="cache/teacher", suffix=None):
    """Deterministic cache directory for a backbone.

    `suffix` separates caches for ablations whose positive text differs
    while stable keys coincide (e.g. the --original-caption ablation,
    which uses "orig-cap").
    """
    safe = backbone_name.replace("/", "_").replace(":", "_")
    if suffix:
        safe = f"{safe}-{suffix}"
    return Path(root) / safe


class TeacherEmbeddingCache:
    """Disk-backed, incremental cache of frozen-backbone teacher embeddings.

    Lookup is O(1) via a stable-key-string → row-index dict built at load.
    """

    # Bump whenever the caption-extraction logic changes. A cache built
    # under a lower schema version is discarded on load.
    #   v1: original — full assistant content for all splits
    #   v2: speech extracts only <transcription>...</transcription> content
    #   v3: teacher uses vanilla Qwen3-Embedding format (raw caption, no
    #       chat wrapper, no "{label}:" modality prefix, explicit
    #       <|endoftext|>, left-padded) so embeddings live in the same
    #       subspace as vanilla Qwen3-Embedding-0.6B text embeddings.
    SCHEMA_VERSION = 3

    EMB_FILE = "embeddings.pt"
    KEYS_FILE = "keys.json"
    META_FILE = "meta.json"

    def __init__(self, cache_dir, embeddings, keys, hidden_size, backbone_name):
        self.cache_dir = Path(cache_dir)
        self.embeddings = embeddings     # (N, D) float32 on CPU
        self.keys = list(keys)           # list[str]
        self.hidden_size = hidden_size
        self.backbone_name = backbone_name
        self._key_to_row = {k: i for i, k in enumerate(self.keys)}

    # ── Persistence ──────────────────────────────────────────

    @classmethod
    def _files(cls, cache_dir):
        d = Path(cache_dir)
        return d / cls.EMB_FILE, d / cls.KEYS_FILE, d / cls.META_FILE

    @classmethod
    def exists(cls, cache_dir):
        emb, keys, meta = cls._files(cache_dir)
        return emb.exists() and keys.exists() and meta.exists()

    @classmethod
    def load(cls, cache_dir):
        emb_path, keys_path, meta_path = cls._files(cache_dir)
        embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
        keys = json.loads(keys_path.read_text())
        meta = json.loads(meta_path.read_text())
        return cls(
            cache_dir=cache_dir,
            embeddings=embeddings,
            keys=keys,
            hidden_size=meta["hidden_size"],
            backbone_name=meta["backbone"],
        )

    @classmethod
    def empty(cls, cache_dir, hidden_size, backbone_name):
        return cls(
            cache_dir=cache_dir,
            embeddings=torch.empty(0, hidden_size, dtype=torch.float32),
            keys=[],
            hidden_size=hidden_size,
            backbone_name=backbone_name,
        )

    def save(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, keys_path, meta_path = self._files(self.cache_dir)
        torch.save(self.embeddings.contiguous(), emb_path)
        keys_path.write_text(json.dumps(self.keys))
        meta_path.write_text(json.dumps({
            "schema_version": self.SCHEMA_VERSION,
            "backbone": self.backbone_name,
            "hidden_size": self.hidden_size,
            "num_samples": len(self.keys),
        }, indent=2))

    # ── Lookup ──────────────────────────────────────────────

    def has(self, stable_key):
        return stable_key in self._key_to_row

    def row_indices_for(self, stable_keys):
        """Return a list of row indices, raising KeyError if any missing."""
        return [self._key_to_row[k] for k in stable_keys]

    def get_by_keys(self, stable_keys, device=None):
        rows = self.row_indices_for(stable_keys)
        out = self.embeddings[rows]
        if device is not None:
            out = out.to(device, non_blocking=True)
        return out

    def __len__(self):
        return len(self.keys)

    # ── Incremental append ──────────────────────────────────

    def append(self, new_keys, new_embeddings):
        """Extend cache with newly computed (keys, embeddings). In-memory only."""
        assert new_embeddings.shape[0] == len(new_keys)
        assert new_embeddings.shape[1] == self.hidden_size
        new_embeddings = new_embeddings.to(torch.float32).cpu().contiguous()
        self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
        start = len(self.keys)
        for i, k in enumerate(new_keys):
            self._key_to_row[k] = start + i
        self.keys.extend(new_keys)


# ── High-level entry point used from train.py ─────────────────────────

def load_or_extend(cache_dir, dataset, model, collator, device,
                   backbone_name, batch_size=64, is_main=True,
                   barrier_fn=None, save_every_batches=200):
    """Load existing cache, compute missing stable keys, persist, return it.

    - Rank 0: does all the compute, saves periodically for crash resumption
    - Other ranks: wait at the barrier, then load the finished cache
    """
    cache_dir = Path(cache_dir)
    hidden_size = model.hidden_size

    if is_main:
        if TeacherEmbeddingCache.exists(cache_dir):
            # Check schema version in meta before loading embeddings.
            # No try/except — a corrupt meta.json should crash, not be
            # silently treated as v1 (which would later mismatch and
            # waste time rebuilding under a wrong assumption).
            meta = json.loads((Path(cache_dir) / TeacherEmbeddingCache.META_FILE).read_text())
            cached_version = meta.get("schema_version", 1)
            if cached_version != TeacherEmbeddingCache.SCHEMA_VERSION:
                print(f"[TeacherCache] Cache at {cache_dir} uses schema v{cached_version}, "
                      f"current is v{TeacherEmbeddingCache.SCHEMA_VERSION} — rebuilding.")
                cache = TeacherEmbeddingCache.empty(cache_dir, hidden_size, backbone_name)
            else:
                cache = TeacherEmbeddingCache.load(cache_dir)
                if cache.hidden_size != hidden_size or cache.backbone_name != backbone_name:
                    print(f"[TeacherCache] Cache at {cache_dir} is for a different "
                          f"backbone/hidden_size — starting fresh.")
                    cache = TeacherEmbeddingCache.empty(cache_dir, hidden_size, backbone_name)
                else:
                    print(f"[TeacherCache] Loaded {len(cache)} cached entries from {cache_dir}")
        else:
            print(f"[TeacherCache] No cache at {cache_dir}; building from scratch.")
            cache = TeacherEmbeddingCache.empty(cache_dir, hidden_size, backbone_name)

        # Find missing stable keys in the current dataset
        missing_global_idx = []
        missing_keys = []
        for i in tqdm(range(len(dataset)),
                      desc="[TeacherCache] scanning for missing keys",
                      unit="sample", leave=False):
            key = dataset.stable_key(i)
            if not cache.has(key):
                missing_global_idx.append(i)
                missing_keys.append(key)

        total_missing = len(missing_keys)
        if total_missing == 0:
            print(f"[TeacherCache] All {len(dataset)} dataset samples already cached.")
        else:
            print(f"[TeacherCache] {total_missing}/{len(dataset)} samples missing — "
                  f"computing now (batch_size={batch_size}).")
            _compute_and_append(
                cache=cache,
                dataset=dataset,
                model=model,
                collator=collator,
                device=device,
                missing_global_idx=missing_global_idx,
                missing_keys=missing_keys,
                batch_size=batch_size,
                save_every_batches=save_every_batches,
            )
            cache.save()
            print(f"[TeacherCache] Final cache size: {len(cache)} entries.")

    if barrier_fn is not None:
        barrier_fn()

    # Every rank loads the finished cache from disk.
    return TeacherEmbeddingCache.load(cache_dir)


def _compute_and_append(cache, dataset, model, collator, device,
                        missing_global_idx, missing_keys,
                        batch_size, save_every_batches):
    """Batched forward through backbone.encode_text to fill missing entries."""
    model.eval()
    total = len(missing_keys)
    batches_since_save = 0

    buf_samples = []
    buf_keys = []

    pbar = tqdm(total=total, desc="[TeacherCache] building",
                unit="sample", dynamic_ncols=True)

    def _flush():
        nonlocal batches_since_save
        if not buf_samples:
            return
        raw_texts = [collator.teacher_raw_text(s) for s in buf_samples]
        enc = model.tokenize_text_native(
            raw_texts, is_query=False, device=device,
        )
        with torch.no_grad():
            emb = model.encode_text(enc["input_ids"], enc["attention_mask"])
        cache.append(buf_keys, emb)
        pbar.update(len(buf_keys))
        pbar.set_postfix(cache=len(cache))
        buf_samples.clear()
        buf_keys.clear()
        batches_since_save += 1
        if batches_since_save >= save_every_batches:
            cache.save()
            batches_since_save = 0

    try:
        for gidx, key in zip(missing_global_idx, missing_keys):
            s = dataset.get_text_sample(gidx)
            buf_samples.append(s)
            buf_keys.append(key)
            if len(buf_samples) >= batch_size:
                _flush()
        _flush()
    finally:
        pbar.close()
