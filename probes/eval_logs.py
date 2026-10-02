"""Media ↔ caption alignment probe, run alongside SciFact during training.

For each modality we keep a small held-out set of (media, caption) pairs
drawn from the LAST N caption-category samples in each Omni-Sets split
(held out from training only when data_pct < 1). On every eval call:

  * encode `media`   through the full student model   → media_emb
  * encode `caption` through the frozen backbone      → caption_emb
  * report  mean cosine(media_emb, caption_emb)  and  R@1/R@5  among the N

As training progresses the mean cosine should rise — that's the direct
signal that the student is learning to land where the backbone's text
embedding for the caption lives.

Held-out sets are persisted under:
  cache/probes/{speech,audio,image,video,visual_doc}/eval_set*.pt

so the expensive preprocessing runs once per environment and is reused
for all eval calls (and across training runs).
"""

import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


_BENCH_DIR = Path(__file__).resolve().parent.parent / "cache" / "probes"
_NUM_SAMPLES = 50
_MAX_AUDIO_DURATION = 60.0


# ── Held-out set builders (one per modality) ──────────────────────────

def _extract_caption_for_eval(messages, split_name):
    """Same semantics as data.dataset._extract_caption — speech uses
    <transcription> content only, other splits use full assistant text."""
    import re
    parts = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if split_name == "speech":
            hits = re.findall(r"<transcription>(.*?)</transcription>", content, re.DOTALL)
            if hits:
                content = " ".join(h.strip() for h in hits)
        for tag in ("<audio>", "</audio>", "<video>", "</video>", "<image>", "</image>"):
            content = content.replace(tag, "")
        content = content.strip()
        if content:
            parts.append(content)
    return " ".join(parts)


def _get_audio_array(media):
    """Delegate to the dataset's decoder for consistency with training.

    Handles: torchcodec AudioStreamer, {array, sampling_rate} dicts, raw
    {bytes, path} dicts (decode=False), resampling to 16 kHz, channel
    downmix. Raises loudly on any decode failure.
    """
    from data.dataset import _get_audio_array as _ds_get_audio_array
    return _ds_get_audio_array(media, target_sr=16000)


def _build_audio_set(split_name, whisper_fe, n=_NUM_SAMPLES):
    """Load LAST n caption samples from the split and preprocess.

    Returns list of dicts with: caption, mel_features, dasheng_audio.
    """
    from datasets import load_dataset, Audio
    ds = load_dataset("MBZUAI/Omni-Sets", split_name, split="train")
    # datasets>=4.0 defaults Audio decoding to torchcodec, which is not
    # ROCm-compatible. decode=False returns the raw {bytes, path} dict and
    # we decode manually via soundfile in _get_audio_array — matches the
    # training-time dataset loader (data/dataset.py).
    ds = ds.cast_column("media", Audio(sampling_rate=16000, decode=False))
    # Filter to caption-only (can't .filter() after cast — decode-heavy).
    keep = [i for i, c in enumerate(ds["category"]) if c == "caption"]
    ds = ds.select(keep[-n * 4:])  # keep the last 4n caption samples as headroom

    out = []
    max_samples = int(_MAX_AUDIO_DURATION * 16000)
    for idx in range(len(ds) - 1, -1, -1):
        if len(out) >= n:
            break
        item = ds[idx]  # raises on decode failure; we want a loud build error
        cap = _extract_caption_for_eval(item.get("messages", []), split_name)
        assert cap, (
            f"[EvalLogs] {split_name}[{idx}] has no usable caption — "
            f"held-out set build aborted so the issue is visible"
        )
        arr = _get_audio_array(item.get("media"))
        assert arr is not None and len(arr) > 0, (
            f"[EvalLogs] {split_name}[{idx}] produced empty audio array"
        )
        if len(arr) > max_samples:
            arr = arr[:max_samples]
        mel = whisper_fe(arr, sampling_rate=16000,
                         return_tensors="pt")["input_features"][0]  # (n_mels, T)
        # n_mels depends on the Whisper variant; the cache path is keyed by
        # n_mels (see _cache_path), so switching variants triggers a rebuild.
        dasheng = torch.from_numpy(arr).view(1, -1)  # (1, S)
        out.append({"caption": cap, "mel_features": mel, "dasheng_audio": dasheng})
    return out


def _build_image_set(split_name, n=_NUM_SAMPLES):
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset("MBZUAI/Omni-Sets", split_name, split="train")
    keep = [i for i, c in enumerate(ds["category"]) if c == "caption"]
    ds = ds.select(keep[-n * 3:])

    out = []
    for idx in range(len(ds) - 1, -1, -1):
        if len(out) >= n:
            break
        item = ds[idx]
        cap = _extract_caption_for_eval(item.get("messages", []), split_name)
        assert cap, (
            f"[EvalLogs] {split_name}[{idx}] has no usable caption"
        )
        media = item.get("media")
        assert media is not None and isinstance(media, Image.Image), (
            f"[EvalLogs] {split_name}[{idx}] media is not a PIL Image: "
            f"type={type(media).__name__}"
        )
        img = media.convert("RGB").copy()
        out.append({"caption": cap, "image": img})
    return out


def _build_video_set(split_name, n=_NUM_SAMPLES, max_frames=16):
    """Sample a small number of frames per video to keep the probe fast."""
    from datasets import load_dataset, Video
    from data.video_loader import load_video_frames
    ds = load_dataset("MBZUAI/Omni-Sets", split_name, split="train")
    # datasets>=4.0 auto-decodes Video via torchcodec (not ROCm-compatible).
    # decode=False → raw {bytes, path} dict; load_video_frames decodes via
    # pyav. Matches data/dataset.py.
    ds = ds.cast_column("media", Video(decode=False))
    keep = [i for i, c in enumerate(ds["category"]) if c == "caption"]
    ds = ds.select(keep[-n * 3:])

    out = []
    from PIL import Image
    for idx in range(len(ds) - 1, -1, -1):
        if len(out) >= n:
            break
        item = ds[idx]
        cap = _extract_caption_for_eval(item.get("messages", []), split_name)
        assert cap, (
            f"[EvalLogs] {split_name}[{idx}] has no usable caption"
        )
        media = item.get("media")
        assert media is not None, (
            f"[EvalLogs] {split_name}[{idx}] has no media — "
            f"dataset row is malformed"
        )
        # load_video_frames raises on decode failure; let it propagate.
        frames_tensor, _ = load_video_frames(media, max_frames=max_frames, image_size=224)
        # frames_tensor: (F, 3, H, W) — convert to list of PIL
        from torchvision.transforms.functional import to_pil_image
        # undo Normalize [0.5] std/mean back to [0,1]
        f = frames_tensor.clone() * 0.5 + 0.5
        f = f.clamp(0, 1)
        pil_frames = [to_pil_image(f[j]) for j in range(f.shape[0])]
        out.append({"caption": cap, "frames": pil_frames})
    return out


# ── Eval-set cache on disk ────────────────────────────────────────────

def _cache_path(split_name, whisper_fe=None):
    """Return the on-disk cache path for a split's held-out eval set.

    Audio/speech caches are keyed by the whisper model's mel-bin count
    so switching whisper variants triggers a
    rebuild instead of silently loading a tensor of the wrong shape.
    """
    d = _BENCH_DIR / split_name
    d.mkdir(parents=True, exist_ok=True)
    if split_name in ("speech", "audio") and whisper_fe is not None:
        n_mels = getattr(whisper_fe, "feature_size", 80)
        return d / f"eval_set_mels{n_mels}.pt"
    return d / "eval_set.pt"


def _load_or_build(split_name, whisper_fe, is_main=True):
    path = _cache_path(split_name, whisper_fe)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    if is_main:
        print(f"[EvalLogs] Building eval set for {split_name}…")
    if split_name in ("speech", "audio"):
        samples = _build_audio_set(split_name, whisper_fe)
    elif split_name in ("image", "visual_doc"):
        samples = _build_image_set(split_name)
    elif split_name == "video":
        samples = _build_video_set(split_name)
    else:
        raise ValueError(split_name)
    if is_main:
        print(f"[EvalLogs]   {split_name}: {len(samples)} samples")
    torch.save(samples, path)
    return samples


# ── Encoding helpers ──────────────────────────────────────────────────

@torch.no_grad()
def _encode_caption_batch(model, captions, device):
    """Encode captions through the frozen backbone in the vanilla
    Qwen3-Embedding passage subspace (no chat wrap, no modality prefix,
    explicit <|endoftext|>, left-padded) — the exact same format the
    teacher cache uses, so the probe actually measures student alignment
    to the distillation target.
    """
    enc = model.tokenize_text_native(
        captions, is_query=False, max_length=2048, device=device,
    )
    return model.encode_text(enc["input_ids"], enc["attention_mask"]).float()


@torch.no_grad()
def _encode_audio_samples(model, collator, samples, device):
    """Feed audio samples through the full student (media encoders + backbone)."""
    batch = []
    for s in samples:
        batch.append({
            "modality": "speech",
            "pair_type": "caption",
            "anchor_text": "",
            "positive_text": s["caption"],
            "sample_id": 0,
            "mel_features": s["mel_features"],
            "dasheng_audio": s["dasheng_audio"],
        })
    out = collator._build_audio_anchor(batch)
    kwargs = {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}
    return model(**kwargs)["embedding"].float()


@torch.no_grad()
def _encode_image_samples(model, collator, samples, device):
    batch = []
    # Convert PIL to the tensor shape the collator's _build_image_anchor expects.
    # The normal anchor builder calls _tensor_to_pil(s["pixel_values"]) — we can
    # skip that and directly give collator pre-built pil images by monkey-patching
    # just this batch's samples to expose 'pixel_values' as a tensor.
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    for s in samples:
        batch.append({
            "modality": "image",
            "pair_type": "caption",
            "anchor_text": "",
            "positive_text": s["caption"],
            "sample_id": 0,
            "pixel_values": tfm(s["image"]),
        })
    out = collator._build_image_anchor(batch)
    kwargs = {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}
    return model(**kwargs)["embedding"].float()


@torch.no_grad()
def _encode_video_samples(model, collator, samples, device):
    batch = []
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    for s in samples:
        frames = torch.stack([tfm(f) for f in s["frames"]])  # (F, 3, H, W)
        batch.append({
            "modality": "video",
            "pair_type": "caption",
            "anchor_text": "",
            "positive_text": s["caption"],
            "sample_id": 0,
            "frames": frames,
        })
    out = collator._build_video_anchor(batch)
    kwargs = {k: v.to(device) for k, v in out.items() if isinstance(v, torch.Tensor)}
    return model(**kwargs)["embedding"].float()


# ── Metrics ───────────────────────────────────────────────────────────

def _cos_and_retrieval(media_emb, caption_emb):
    """Compute paired cosine similarity and top-k retrieval.

    media_emb[i] ↔ caption_emb[i]. Retrieval: for each media, rank all
    captions by similarity; check whether caption i is in top-k.
    """
    m = F.normalize(media_emb, p=2, dim=-1)
    c = F.normalize(caption_emb, p=2, dim=-1)
    # Per-pair cosine
    cos = (m * c).sum(dim=-1)                           # (N,)
    # Cross-similarity for retrieval
    sim = m @ c.T                                        # (N, N)
    N = sim.shape[0]
    target = torch.arange(N, device=sim.device)
    rank = sim.argsort(dim=-1, descending=True)
    rank_of_target = (rank == target[:, None]).float().argmax(dim=-1)
    r1 = float((rank_of_target < 1).float().mean())
    r5 = float((rank_of_target < 5).float().mean())
    return {
        "mean_cos": float(cos.mean()),
        "median_cos": float(cos.median()),
        "recall@1": r1,
        "recall@5": r5,
        "n": N,
    }


# ── Public entry point ────────────────────────────────────────────────

_CACHED_SETS: Dict[str, list] = {}


def evaluate_media_caption(model, collator, device, whisper_fe,
                           splits=("speech", "audio", "image", "video", "visual_doc"),
                           is_main=True):
    """Run cross-modal media↔caption similarity probe.

    Args:
        model:         unwrapped OmniEmbedModel
        collator:      ContrastiveCollator instance (for anchor builders)
        device:        torch device
        whisper_fe:    WhisperFeatureExtractor for audio preprocessing
        splits:        tuple of split names to evaluate

    Returns:
        dict keyed by split with per-split metrics, plus
        'overall_mean_cos' (mean of per-split mean_cos) and 'elapsed_s'.
    """
    global _CACHED_SETS
    t0 = time.time()
    model.eval()

    results = {}
    per_split_cos = []

    for split in splits:
        if split not in _CACHED_SETS:
            _CACHED_SETS[split] = _load_or_build(split, whisper_fe, is_main=is_main)
        samples = _CACHED_SETS[split]
        assert samples, (
            f"[EvalLogs] held-out set for {split!r} is empty — the build "
            f"pipeline silently produced zero samples; check the cache at "
            f"{_cache_path(split, whisper_fe)} and delete it to force a rebuild."
        )

        # Student (media) embedding — no try/except; a crash here is the
        # correct signal that the student path is broken.
        if split in ("speech", "audio"):
            media_emb = _encode_audio_samples(model, collator, samples, device)
        elif split in ("image", "visual_doc"):
            media_emb = _encode_image_samples(model, collator, samples, device)
        elif split == "video":
            media_emb = _encode_video_samples(model, collator, samples, device)
        else:
            raise ValueError(f"unsupported split for media encode: {split!r}")

        # Teacher (caption) embedding — native Qwen3 passage subspace.
        captions = [s["caption"] for s in samples]
        caption_emb = _encode_caption_batch(model, captions, device)

        metrics = _cos_and_retrieval(media_emb, caption_emb)
        results[split] = metrics
        per_split_cos.append(metrics["mean_cos"])

    results["overall_mean_cos"] = float(np.mean(per_split_cos)) if per_split_cos else 0.0
    results["elapsed_s"] = time.time() - t0
    return results


def format_block(step, results):
    """Pretty-print block, mirroring the scifact log style."""
    lines = [
        "=" * 60,
        f"  Media↔Caption @ step {step}",
    ]
    for split in ("speech", "audio", "image", "video", "visual_doc"):
        if split not in results:
            continue
        m = results[split]
        lines.append(
            f"  {split:<11} cos={m['mean_cos']:+.4f}  "
            f"R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  (n={m['n']})"
        )
    lines.append(f"  overall mean cos:    {results.get('overall_mean_cos', 0):+.4f}")
    lines.append(f"  eval time:           {results.get('elapsed_s', 0):.1f}s")
    lines.append("=" * 60)
    return "\n".join(lines)
