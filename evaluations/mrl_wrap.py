"""Matryoshka-truncation wrapper.

When OMNI_TRUNCATE_DIM is set, post-processes every encoded embedding to
truncate to that prefix length and re-L2-normalize, and tags the
mteb_model_meta.name with a `_d-<N>` suffix so per-dim results land in
separate cache directories.

Used by run_mteb.py (MTEB-v2, MAEB, ViDoRe-V3) via maybe_wrap_for_mrl(model);
mmeb/run_mmeb.py applies the same truncation itself.

No-op when OMNI_TRUNCATE_DIM is unset.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


def _truncate_renorm(emb, dim):
    if isinstance(emb, np.ndarray):
        x = emb[..., :dim]
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-12, None)
    if torch.is_tensor(emb):
        return F.normalize(emb[..., :dim], p=2, dim=-1)
    if isinstance(emb, list):
        return [_truncate_renorm(x, dim) for x in emb]
    if isinstance(emb, dict):
        return {k: _truncate_renorm(v, dim) for k, v in emb.items()}
    return emb


def maybe_wrap_for_mrl(model):
    """If OMNI_TRUNCATE_DIM env-var is set, monkey-patch model's encode-like
    methods to truncate+renorm output, and tag mteb_model_meta.name.

    Returns the same model object (mutated).
    """
    raw = os.environ.get("OMNI_TRUNCATE_DIM")
    if not raw:
        return model
    dim = int(raw)
    if dim <= 0:
        raise ValueError(f"OMNI_TRUNCATE_DIM must be positive, got {raw!r}")

    for attr in ("encode", "encode_query", "encode_corpus",
                 "encode_documents", "encode_text", "encode_image",
                 "encode_audio", "encode_video"):
        if not hasattr(model, attr):
            continue
        fn = getattr(model, attr)
        if not callable(fn):
            continue
        # Skip already-wrapped attributes (idempotent).
        if getattr(fn, "_mrl_wrapped", False):
            continue

        def make_wrapper(orig, d):
            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                return _truncate_renorm(out, d)
            wrapped._mrl_wrapped = True
            return wrapped

        setattr(model, attr, make_wrapper(fn, dim))

    meta = getattr(model, "mteb_model_meta", None)
    if meta is not None and not getattr(meta, "_mrl_tagged", False):
        meta.name = f"{meta.name}_d-{dim}"
        meta._mrl_tagged = True

    return model
