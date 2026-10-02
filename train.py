"""Omni-Embed-Mini training script.

Usage (8 GPUs, bf16):
    accelerate launch --num_processes=8 --mixed_precision=bf16 \
        train.py --config configs/omni_embed_mini_0.9b.yaml --text_mining --media_mining

or simply:  bash scripts/train.sh configs/omni_embed_mini_0.9b.yaml
"""

import os
import json
import sys
import argparse
import math
import time
import yaml

# ── NFS flock workaround (multinode only) ─────────────────────────────────
# HF `datasets` guards its cache with fcntl.flock, which returns OSError 37
# (ENOLCK) when many processes across nodes hit a shared NFS cache at once.
# SoftFileLock uses atomic lockfile creation instead of fcntl, which works over
# NFS. Must run before `datasets` is first imported (via data.dataset below).
# Enabled only when OMNI_SOFT_FILELOCK=1.
if os.environ.get("OMNI_SOFT_FILELOCK") == "1":
    import filelock as _filelock
    _filelock.FileLock = _filelock.SoftFileLock

# Dump all thread stacks on SIGUSR1 (debugging hangs in multi-GPU runs).
import faulthandler, signal, threading, os as _os, sys as _sys, traceback as _tb
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)


def _crash_on_thread_error(args):
    """Make any background-thread crash kill the main process.

    Daemon-thread exceptions are silent by default — the miner thread, the
    DataLoader workers, etc. would error and the main loop would continue
    on garbage. We want loud failures: print the full traceback and
    SIGTERM the whole process so the run aborts and shows the cause.
    """
    print(f"\n[FATAL] Uncaught exception in thread {args.thread.name}:",
          flush=True, file=_sys.stderr)
    _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                        file=_sys.stderr)
    _sys.stderr.flush()
    _os.kill(_os.getpid(), signal.SIGTERM)


threading.excepthook = _crash_on_thread_error

import torch
import torch.nn as nn

# ROCm SDPA: the flash kernel segfaults on ROCm for large attention workloads, so
# it is disabled. The memory-efficient kernel is stable and avoids O(B*H*S^2)
# materialization; math is kept as the final fallback.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from torch.utils.data import DataLoader
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

from model.omni_embed import OmniEmbedModel
from data.dataset import ContrastiveOmniDataset, ModalityBatchSampler
from data.collator import ContrastiveCollator
from loss.mrl_loss import MRLLoss
from train.scheduler import get_cosine_schedule_with_warmup
from utils.checkpoint import CheckpointManager
from utils.logger import TrainingLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume", type=str, default="auto",
        help="Checkpoint dir to resume from. 'auto' (default) resumes "
             "from <output_dir>/latest if it exists, else starts fresh. "
             "Pass an explicit path to resume from a specific checkpoint.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing <output_dir>/latest and start from scratch "
             "(overrides --resume).",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument("--text_mining", action="store_true",
                        help="Enable text-based hard negative mining (FAISS over captions)")
    parser.add_argument("--media_mining", action="store_true",
                        help="Enable media hard negative mining (audio/image/video). "
                             "Independent of --text_mining: combine them (tm-mm) or "
                             "use alone (mm).")
    parser.add_argument("--original-caption", dest="original_caption",
                        action="store_true",
                        help="Ablation: use the dataset's original_text column "
                             "(short, non-dense caption) as the positive for "
                             "caption-category samples, instead of the "
                             "regenerated dense caption from messages. "
                             "Default: disabled.")
    parser.add_argument("--encoder_adapt", choices=["lora", "full", "frozen"],
                        default="lora",
                        help="Phase-2 media-encoder adaptation mode (ablation). "
                             "'lora' (default): inject LoRA adapters. "
                             "'full': full fine-tune encoder weights (no LoRA). "
                             "'frozen': keep encoders frozen (projectors only). "
                             "Run tag suffix: -encfull / -encfrozen (none for lora).")
    parser.add_argument("--text_backbone_lora", action="store_true",
                        help="Forgetting ablation: also inject LoRA into "
                             "the text backbone at phase 2. Normally the backbone "
                             "is never adapted (it is the frozen self-distillation "
                             "teacher). This flag adapts it to demonstrate that "
                             "touching the text path induces forgetting (text/MTEB "
                             "is expected to degrade vs the frozen main recipe). "
                             "Run tag suffix: -txtlora.")
    parser.add_argument("--text_backbone_full", action="store_true",
                        help="Forgetting ablation: unfreeze the entire "
                             "text backbone at phase 2 (full fine-tune, no LoRA) "
                             "at lr_backbone_full (falls back to lr_backbone_lora). "
                             "The aggressive sibling of --text_backbone_lora; "
                             "expected to forget MORE than LoRA. Mutually "
                             "exclusive with --text_backbone_lora. "
                             "Run tag suffix: -txtfull.")
    parser.add_argument("--override", action="append", default=[],
                        metavar="key.path=value",
                        help="Override any config key. Value is parsed as YAML "
                             "so lists/ints/floats/bools work, e.g. "
                             "--override 'data.omni_sets_splits=[speech,audio,image,video]' "
                             "--override 'training.batch_size=16'. Repeatable.")
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    base = config.pop("base_config", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        with open(base_path) as f:
            base_config = yaml.safe_load(f)
        merged = {**base_config}
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return config


def apply_overrides(config, overrides):
    """Apply --override key.path=value strings to a nested config dict.

    Value is parsed as YAML so lists/ints/floats/bools/strings all work
    naturally (e.g. `[speech,audio]` becomes a list, `0.07` a float, `true`
    a bool, anything else a string).
    """
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--override expects 'key.path=value', got {spec!r}")
        key_path, raw_value = spec.split("=", 1)
        keys = key_path.strip().split(".")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as e:
            raise ValueError(f"--override {spec!r}: failed to parse value as YAML: {e}")
        node = config
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return config


def build_optimizer(model, config):
    """Build AdamW optimizer with per-component learning rates.

    Phase 1 trainables: audio projectors + vision projectors. Backbone and
    media encoders stay frozen; no MRL head exists.
    """
    training_cfg = config.get("training", {})
    lr_audio = training_cfg.get("lr_audio", 2e-4)
    lr_vision = training_cfg.get("lr_vision", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 0.01)

    param_groups = []

    audio_params = list(model.whisper_down_proj.parameters()) + \
                   list(model.dasheng_down_proj.parameters())
    if audio_params:
        param_groups.append({"params": audio_params, "lr": lr_audio, "name": "audio_proj"})

    # Vision projectors exist only in vision_mode == custom.
    if model.vision_mode == "custom":
        vision_params = list(model.image_projector.parameters()) + \
                        list(model.video_pooler.parameters()) + \
                        list(model.video_projector.parameters())
        if vision_params:
            param_groups.append({"params": vision_params, "lr": lr_vision, "name": "vision_proj"})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def add_lora_params_to_optimizer(model, optimizer, config):
    """Add media-encoder LoRA parameters to optimizer at phase transition.

    The backbone is never adapted — it is the frozen teacher in
    self-distillation and the source of Qwen3-Embedding's text capability.
    """
    training_cfg = config.get("training", {})
    lr_encoder_lora = training_cfg.get("lr_encoder_lora", 1e-4)

    encoder_lora = model.get_encoder_lora_params()

    if encoder_lora:
        optimizer.add_param_group({
            "params": encoder_lora, "lr": lr_encoder_lora, "name": "encoder_lora"
        })


def activate_phase2_mode(model, optimizer, config, encoder_adapt, is_main=False,
                         text_backbone_lora=False, text_backbone_full=False):
    """Activate the phase-2 encoder-adaptation mode (ablation dispatch).

    'lora'   → inject LoRA adapters + add them to the optimizer (default path).
    'full'   → full fine-tune encoder weights (no LoRA) at lr_encoder_full.
    'frozen' → keep encoders frozen; nothing added to the optimizer.

    If text_backbone_lora is True (forgetting ablation), also inject LoRA
    into the text backbone and add those params to the optimizer at
    lr_backbone_lora (falls back to lr_encoder_lora). This composes with any
    encoder_adapt mode.

    Returns the list of newly-added trainable params (so the caller can
    broadcast them across ranks for DDP agreement); empty for 'frozen'
    without text_backbone_lora.
    """
    training_cfg = config.get("training", {})

    def _add_backbone_lora(new_params_so_far):
        """Inject + register text-backbone LoRA. Appends to param list."""
        if not text_backbone_lora:
            return new_params_so_far
        bb_lora = model.activate_text_backbone_lora(config.get("lora", {}))
        if bb_lora:
            lr_bb = training_cfg.get("lr_backbone_lora",
                                     training_cfg.get("lr_encoder_lora", 1e-4))
            optimizer.add_param_group({
                "params": bb_lora, "lr": lr_bb, "name": "backbone_lora"
            })
            if is_main:
                print(f"  Text-backbone LoRA — {len(bb_lora)} tensors "
                      f"@ lr={lr_bb} (forgetting ablation)")
        return list(new_params_so_far) + bb_lora

    def _add_backbone_full(new_params_so_far):
        """Unfreeze + register the entire text backbone. Appends to
        param list. Uses the same device-safety as the encoder_full path so a
        resumed Adam state can't land on a device mismatch at step()."""
        if not text_backbone_full:
            return new_params_so_far
        bb_full = model.activate_text_backbone_full()
        if bb_full:
            lr_bb = training_cfg.get("lr_backbone_full",
                                     training_cfg.get("lr_backbone_lora", 1e-5))
            dev = optimizer.param_groups[0]["params"][0].device
            for p in bb_full:
                if p.device != dev:
                    p.data = p.data.to(dev)
                if p.grad is not None and p.grad.device != dev:
                    p.grad = p.grad.to(dev)
            optimizer.add_param_group({
                "params": bb_full, "lr": lr_bb, "name": "backbone_full"
            })
            for p in bb_full:
                st = optimizer.state.get(p, {})
                for k, v in st.items():
                    if torch.is_tensor(v) and v.device != dev:
                        st[k] = v.to(dev)
            if is_main:
                print(f"  Text-backbone full unfreeze — {len(bb_full)} "
                      f"tensors @ lr={lr_bb} (forgetting ablation)")
        return list(new_params_so_far) + bb_full

    def _add_backbone(params):
        """Apply whichever text-backbone ablation is enabled (mutually
        exclusive; a no-op when neither flag is set)."""
        return _add_backbone_full(_add_backbone_lora(params))

    if encoder_adapt == "lora":
        freeze_vision = training_cfg.get("freeze_vision_in_phase2", False)
        model.activate_phase2(config.get("lora", {}), freeze_vision=freeze_vision)
        if is_main and freeze_vision:
            print("  freeze_vision_in_phase2=True: custom vision tower stays "
                  "FROZEN in phase 2 (audio LoRA still active)")
        add_lora_params_to_optimizer(model, optimizer, config)
        return _add_backbone(model.get_encoder_lora_params())
    elif encoder_adapt == "full":
        model.activate_phase2_full()
        lr_full = training_cfg.get("lr_encoder_full", 2.0e-5)
        full_params = model.get_encoder_full_params()
        if full_params:
            # The encoder params must live on the SAME device Adam runs its math
            # on, or optimizer.step() dies with "exp_avg ... cuda:N and cpu".
            # Derive that device from a param already in the optimizer (the
            # projectors, which train fine), and force the encoder params +
            # any existing grad + any loaded Adam state onto it.
            dev = optimizer.param_groups[0]["params"][0].device
            moved = 0
            for p in full_params:
                if p.device != dev:
                    p.data = p.data.to(dev)
                    moved += 1
                if p.grad is not None and p.grad.device != dev:
                    p.grad = p.grad.to(dev)
            optimizer.add_param_group({
                "params": full_params, "lr": lr_full, "name": "encoder_full"
            })
            # Relocate any (resumed) optimizer state for these params too.
            for p in full_params:
                st = optimizer.state.get(p, {})
                for k, v in st.items():
                    if torch.is_tensor(v) and v.device != dev:
                        st[k] = v.to(dev)
            if is_main:
                print(f"  Full-FT encoders: {len(full_params)} tensors @ "
                      f"lr={lr_full}, target device={dev}, moved {moved} onto it")
        return _add_backbone(full_params)
    elif encoder_adapt == "frozen":
        model.activate_phase2_frozen()
        if is_main:
            print("  Frozen encoders: no encoder params added (projectors only)")
        return _add_backbone([])
    else:
        raise ValueError(f"unknown encoder_adapt: {encoder_adapt!r}")


def resolve_run_config(args, config):
    """Resolve run-time knobs and output dir / run tag.

    Run tag encodes data_pct + mining config so different runs don't
    collide on wandb names or checkpoint dirs:
      - data_pct < 1.0 → `pct<NN>` (e.g. pct20 for 20%); full runs omit
      - mining: `no-mining` | `tm` | `mm` | `tm-mm`
      - original-caption ablation: appends `orig-cap`

    Returns:
        use_text_mining, use_media_mining, mrl_dims, output_dir, run_tag,
        use_original_caption
    """
    model_cfg = config.get("model", {})
    use_media_mining = args.media_mining
    use_text_mining = args.text_mining

    mrl_dims = model_cfg.get("mrl_dims_coarse", [128, 256, 512, 1024])

    if use_text_mining and use_media_mining:
        mining_tag = "tm-mm"
    elif use_media_mining:
        mining_tag = "mm"
    elif use_text_mining:
        mining_tag = "tm"
    else:
        mining_tag = "no-mining"

    # data_pct tag: omit for full runs, include for partial
    parts = []
    if args.data_pct < 1.0:
        parts.append(f"pct{int(round(args.data_pct * 100)):02d}")
    parts.append(mining_tag)
    if args.original_caption:
        parts.append("orig-cap")
    # encoder-adaptation ablation tag (default 'lora' adds no suffix).
    encoder_adapt = getattr(args, "encoder_adapt", "lora")
    if encoder_adapt == "full":
        parts.append("encfull")
    elif encoder_adapt == "frozen":
        parts.append("encfrozen")
    # Text-backbone LoRA ablation: distinct checkpoint dir so it never collides
    # with the frozen-backbone main recipe.
    if getattr(args, "text_backbone_lora", False):
        parts.append("txtlora")
    # Full text-backbone unfreeze ablation: distinct dir too.
    if getattr(args, "text_backbone_full", False):
        parts.append("txtfull")
    run_tag = "-".join(parts)

    base_dir = config.get("training", {}).get("output_dir", "checkpoints/omni-embed")
    output_dir = f"{base_dir}-{run_tag}"

    return (use_text_mining, use_media_mining, mrl_dims, output_dir, run_tag,
            args.original_caption, encoder_adapt)


# ── Miner setup helper ────────────────────────────────────────────

def _setup_miner_fns(miner, model, device, dataset, collator):
    """Set text and media encode functions on the miner.

    Text encode: fast path through backbone only (no media encoders).
    Media encode: full model forward for actual media samples.
    `collator` must be passed in — media_encode_fn closes over it for
    batch assembly. (It is NOT a free variable from main(); top-level
    function closures don't see the caller's locals.)
    """
    # IMPORTANT: load a SEPARATE tokenizer instance for the miner. The
    # miner runs in a background thread inside the main process; it calls
    # tokenizer.__call__ → set_truncation_and_padding which acquires the
    # HF fast-tokenizer's internal Mutex. If a DataLoader worker forks
    # from the main process while the miner holds that lock, the worker
    # inherits the locked tokenizer with no holder thread and crashes
    # with `RuntimeError: Already borrowed` on the first encode call.
    # A separate instance has its own Mutex → no shared state, no race.
    from transformers import AutoTokenizer
    miner_tokenizer = AutoTokenizer.from_pretrained(
        model.tokenizer.name_or_path, padding_side=model.tokenizer.padding_side,
    )
    # `model.tokenizer` is NOT the base tokenizer: OmniEmbedModel.__init__ adds
    # AUDIO_SPECIAL_TOKENS (and CUSTOM_VISION_SPECIAL_TOKENS in custom vision
    # mode). A tokenizer reloaded from `name_or_path` lacks <|audio_pad|>, so
    # audio features would be silently dropped from mined audio anchors (every
    # anchor collapses to the same embedding and no audio negatives survive the
    # false-negative filter). Re-add the same tokens in the same order so the
    # ids match the model's, and fail loudly if they do not.
    from model.omni_embed import (
        AUDIO_SPECIAL_TOKENS, CUSTOM_VISION_SPECIAL_TOKENS, AUDIO_PAD_TOKEN,
    )
    _miner_tokens = list(AUDIO_SPECIAL_TOKENS)
    if getattr(model, "vision_mode", None) == "custom":
        _miner_tokens += CUSTOM_VISION_SPECIAL_TOKENS
    miner_tokenizer.add_special_tokens(
        {"additional_special_tokens": _miner_tokens}
    )
    _miner_pad_id = miner_tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)
    if _miner_pad_id != model._audio_pad_id:
        raise RuntimeError(
            f"miner tokenizer {AUDIO_PAD_TOKEN} id {_miner_pad_id} != model's "
            f"{model._audio_pad_id}; media mining would silently drop audio "
            "features"
        )

    # Miner-private collator: same construction as the shared one, but bound
    # to `miner_tokenizer` so that media_encode_fn → collator(samples) →
    # tokenizer(...) does NOT touch the shared fast-tokenizer Mutex used by
    # the main thread / DataLoader workers. Without this we crash with
    # `RuntimeError: Already borrowed` (and an occasional follow-on segfault
    # when a worker forks while the lock is held).
    # neg_cache=None: mining itself doesn't need negatives in its anchor batch.
    miner_collator = ContrastiveCollator(
        tokenizer=miner_tokenizer,
        vision_processor=collator.vision_processor,
        max_seq_length=collator.max_seq_length,
        tokens_per_encoder=collator.tokens_per_encoder,
        num_video_tokens=collator.num_video_tokens,
        neg_cache=None,
        dataset=collator.dataset,
        vision_mode=collator.vision_mode,
        video_as_images=collator.video_as_images,
    )

    # Text encode function
    def text_encode_fn(texts):
        enc = miner_tokenizer(texts, padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(device)
        return model.encode_text(enc["input_ids"], enc["attention_mask"])

    miner.set_text_encode_fn(text_encode_fn)

    # Media encode function: takes sample IDs, loads + collates a batch, encodes through full model
    def media_encode_fn(sample_ids):
        # Group by underlying HF dataset for bulk .select() access
        from collections import defaultdict
        groups = defaultdict(list)  # ds_id -> [(global_idx, split_info, local_idx)]
        for sid in sample_ids:
            (split_name, ds, count), local_idx = dataset.get_split_and_local(sid)
            groups[id(ds)].append((sid, split_name, ds, local_idx))

        # Bulk-load samples using .select() per underlying dataset
        samples = []
        for ds_id, entries in groups.items():
            ds = entries[0][2]
            local_indices = [e[3] for e in entries]
            batch_ds = ds.select(local_indices)
            for i, (gid, split_name, _, _) in enumerate(entries):
                samples.append(dataset._process_sample(batch_ds[i], split_name, gid))

        batch = miner_collator(samples)
        # Move to device
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)

        # Build anchor kwargs (same logic as training loop)
        kwargs = {
            "input_ids": batch_gpu.get("anchor_input_ids"),
            "attention_mask": batch_gpu.get("anchor_attention_mask"),
        }
        if "anchor_input_features" in batch_gpu:
            kwargs["input_features"] = batch_gpu["anchor_input_features"]
            kwargs["dasheng_audio"] = batch_gpu["anchor_dasheng_audio"]
        if "anchor_image_pixel_values" in batch_gpu:
            kwargs["image_pixel_values"] = batch_gpu["anchor_image_pixel_values"]
            kwargs["image_grid_thw"] = batch_gpu["anchor_image_grid_thw"]
        if "anchor_video_pixel_values" in batch_gpu:
            kwargs["video_pixel_values"] = batch_gpu["anchor_video_pixel_values"]
            kwargs["video_grid_thw"] = batch_gpu["anchor_video_grid_thw"]
            if "anchor_video_num_frames" in batch_gpu:
                kwargs["video_num_frames"] = batch_gpu["anchor_video_num_frames"]

        out = model(**kwargs)
        return out["embedding"]

    miner.set_media_encode_fn(media_encode_fn)


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.text_backbone_lora and args.text_backbone_full:
        raise SystemExit("--text_backbone_lora and --text_backbone_full are "
                         "mutually exclusive: pick one.")
    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    training_cfg = config.get("training", {})
    mining_cfg = config.get("mining", {})

    # ── Global seed ──
    # Seed python/numpy/torch (mining-pool selection and adapter init) before
    # model build. Set via --override training.seed=N. Same seed on every rank.
    import random as _random
    import numpy as _np
    _seed = int(training_cfg.get("seed", 42))
    _random.seed(_seed)
    _np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    print(f"[seed] global seed = {_seed} (torch + numpy + random)")

    # ── Resolve run config (media mining on/off) ──
    (use_text_mining, use_media_mining, mrl_dims, output_dir, run_tag,
     use_original_caption, encoder_adapt) = resolve_run_config(args, config)

    # ── Autoresume: turn --resume auto into a concrete path (or None) ──
    # --fresh wins over --resume; an explicit path is respected as-is.
    if args.fresh:
        args.resume = None
    elif args.resume == "auto":
        latest_weights = os.path.join(output_dir, "latest", "trainable_weights.pt")
        if os.path.exists(latest_weights):
            args.resume = os.path.join(output_dir, "latest")
        else:
            args.resume = None
    elif args.resume and args.resume.lower() in ("none", "off", "no"):
        args.resume = None

    # ── AMD ROCm environment ──
    os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen-user-db")
    os.environ["HIPFFT_PLAN_CACHE_MAX_SIZE"] = "0"

    # ── Accelerator setup ──
    # Keep find_unused_parameters=True: our batch sampler groups by modality,
    # so different batches activate different encoder subsets (audio batches
    # skip vision, image batches skip whisper/dasheng, etc.). The unused
    # param set therefore varies per step, which rules out static_graph=True.
    # gradient_as_bucket_view=True avoids a full grad-bucket copy each step.
    # broadcast_buffers=False: this model has no BN/running-stat buffers that
    # need cross-rank sync. Every buffer is either a constant (RMSNorm/LN have
    # none, RoPE inv_freq is init-time) or deterministically recomputed per
    # forward (Qwen3-VL `rope_deltas` from input shapes). With broadcast on,
    # the miner thread's text-only forward takes a backbone code path that
    # leaves `rope_deltas` unset on its rank while other ranks compute it on
    # GPU; the next main forward's `_sync_buffers` then NCCL-broadcasts a
    # CPU/None tensor and crashes ("No backend type associated with device
    # type cpu"). Disabling the broadcast eliminates this race; identical
    # inputs already give identical recomputed buffers per rank.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    # Long NCCL timeout: rank 0 builds the teacher cache (can take tens of
    # minutes on the full data); other ranks are stuck at wait_for_everyone
    # during that time. Default 10 min timeout would abort them.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
        gradient_accumulation_steps=training_cfg.get("grad_accum_steps", 1),
    )

    if accelerator.is_main_process:
        print("=" * 60)
        print("Omni-Embed Training")
        print("=" * 60)
        print(f"Config:       {args.config}")
        print(f"Run tag:      {run_tag}")
        print(f"MRL dims:     {mrl_dims}")
        print(f"Text mining:  {use_text_mining}")
        print(f"Media mining: {use_media_mining}")
        print(f"Orig caption: {use_original_caption}")
        print(f"Distill:      ON (always)")
        print(f"Output dir:   {output_dir}")
        print(f"Processes:    {accelerator.num_processes}")
        if args.resume:
            print(f"Resume:       {args.resume}")
        elif args.fresh:
            print(f"Resume:       OFF (--fresh)")
        else:
            print(f"Resume:       fresh start (no latest/ checkpoint found)")

    # ── Wandb ──
    wandb_name = args.wandb_run_name
    if wandb_name:
        wandb_name = f"{wandb_name}-{run_tag}"
    if accelerator.is_main_process and wandb_name:
        import wandb
        wandb.init(project="omni-embed", name=wandb_name, config={
            **config,
            "run": {
                "text_mining": use_text_mining,
                "media_mining": use_media_mining,
                "original_caption": use_original_caption,
                "tag": run_tag,
            },
            "mrl_dims": mrl_dims,
        })

    # ── Model ──
    model = OmniEmbedModel(config)
    if accelerator.is_main_process:
        model.print_param_summary()

    # ── Loss ──
    distill_cfg = config.get("distillation", {})
    distill_weight = distill_cfg.get("weight", 1.0)  # always on
    criterion = MRLLoss(
        dims=mrl_dims,
        temperature_init=training_cfg.get("temperature_init", 0.07),
        gather_distributed=False,
        distill_weight=distill_weight,
    )

    # ── Data (full dataset, no filtering) ──
    from transformers import WhisperFeatureExtractor
    whisper_name = config.get("audio", {}).get("whisper", "openai/whisper-small")
    whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_name)

    data_cfg = config.get("data", {})
    dataset = ContrastiveOmniDataset(
        split_names=data_cfg.get("omni_sets_splits"),
        whisper_fe=whisper_fe,
        image_size=config.get("vision", {}).get("image_size", 224),
        max_video_frames=config.get("vision", {}).get("max_video_frames", 196),
        max_audio_duration=config.get("audio", {}).get("max_audio_duration", 300.0),
        data_pct=args.data_pct,
        use_original_caption=use_original_caption,
        max_duration_s=data_cfg.get("max_duration_s"),
    )

    batch_size = training_cfg.get("batch_size", 8)
    batch_sampler = ModalityBatchSampler(
        dataset, batch_size, shuffle=True,
        seed=int(training_cfg.get("seed", 42)),
        batch_size_overrides=training_cfg.get("batch_size_overrides", {"video": 2}),
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )

    # ── Hard negative miner (activated at phase 2) ──
    # Gated by CLI flags: --text_mining and --media_mining.
    #
    # Cache lives on EVERY rank so the collator can look up hard negs on
    # every batch (not just rank 0's). Only rank 0 runs the actual FAISS /
    # encoder pipeline; the resulting cache state is broadcast from rank 0
    # to all ranks on the text-refresh schedule (see training loop).
    from mining.cache import NegativeCache
    any_mining = use_text_mining or use_media_mining
    neg_cache = NegativeCache() if any_mining else None

    miner = None
    use_mining = any_mining and accelerator.is_main_process
    if use_mining:
        from mining import HybridMiner
        miner = HybridMiner(
            dataset=dataset,
            text_index_size=mining_cfg.get("text_index_size", 50000),
            media_index_size=mining_cfg.get("media_index_size", 5000),
            n_hard_negs=mining_cfg.get("n_hard_negs", 5),
            false_neg_threshold=mining_cfg.get("false_neg_threshold", 0.92),
        )
        # Replace the miner's local cache with the shared (all-ranks) one
        # so that rank 0's writes are what we then broadcast each cycle.
        miner.cache = neg_cache
        # Use batch_sampler's modality classification (handles omni split)
        miner.set_modality_indices(batch_sampler.modality_indices)

    # ── Collator (with optional hard neg cache) ──
    # Image processor source depends on vision_mode:
    #   custom — our Qwen3.5 wrapper exposes the image processor
    #   native — pull the backbone's own image processor via AutoImageProcessor
    #   none   — no vision; no processor needed
    unwrapped_model = model  # not yet wrapped by accelerator
    if unwrapped_model.vision_mode == "custom":
        vision_processor = unwrapped_model.vision_encoder.image_processor
    elif unwrapped_model.vision_mode == "native":
        from transformers import AutoImageProcessor
        vision_processor = AutoImageProcessor.from_pretrained(
            config["model"]["backbone"], trust_remote_code=True,
        )
    else:
        vision_processor = None

    collator = ContrastiveCollator(
        tokenizer=model.tokenizer,
        vision_processor=vision_processor,
        max_seq_length=training_cfg.get("max_seq_length", 2048),
        tokens_per_encoder=config.get("audio", {}).get("tokens_per_encoder", 128),
        num_video_tokens=config.get("vision", {}).get("num_video_tokens", 196),
        neg_cache=neg_cache,
        dataset=dataset,
        vision_mode=unwrapped_model.vision_mode,
        video_as_images=config.get("data", {}).get("video_as_images", False),
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=data_cfg.get("num_workers", 4),
        # pin_memory=True allocates page-locked host RAM; with heavy audio
        # batches × 8 ranks × multiple workers this competes with the OS
        # page cache and can cause host OOM. The h2d copy cost of
        # unpinned memory is a few ms/batch — worth it for the headroom.
        pin_memory=False,
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )

    # ── Optimizer + Scheduler ──
    optimizer = build_optimizer(model, config)

    epochs = training_cfg.get("epochs", 5)
    # Sampler already handles DDP sharding — len() returns per-GPU count
    steps_per_epoch = len(batch_sampler)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(training_cfg.get("warmup_pct", 0.02) * total_steps))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Phase transition ──
    phase_switch_pct = training_cfg.get("phase_switch_pct", 0.2)
    phase_switch_step = int(phase_switch_pct * total_steps)

    # ── Mining schedule (resolved to steps below, after steps_per_epoch is known) ──

    # ── Checkpoint ──
    ckpt_manager = CheckpointManager(output_dir, keep_best=3,
                                     num_replicas=accelerator.num_processes)

    # ── Prepare with accelerator (dataloader excluded — sampler handles DDP) ──
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    # ── Resume ──
    global_step = 0
    start_epoch = 0
    resume_skip = 0   # #batches to fast-forward in the resume epoch (set below)
    best_loss = float("inf")

    if args.resume:
        # Peek at training_state.json BEFORE loading optimizer state. If the
        # checkpoint was saved in phase 2, the optimizer has extra LoRA param
        # groups; we must activate phase 2 + add those groups before
        # optimizer.load_state_dict, otherwise it raises
        # "loaded state dict has a different number of parameter groups".
        peek_state_path = os.path.join(os.path.dirname(args.resume), "training_state.json")
        if os.path.exists(peek_state_path):
            with open(peek_state_path) as f:
                _peek = json.load(f)
            _peek_step = _peek.get("global_step", 0)
            _peek_phase = _peek.get("phase", 1)
            # Use saved `phase` as the authoritative signal — and ONLY
            # the phase. Step-at-boundary is NOT a reliable proxy: the
            # end-of-epoch save fires AFTER global_step += 1, so it can
            # write step=phase_switch_step with phase=1 (phase 2 hasn't
            # activated yet because the phase-transition check in the loop
            # only fires when global_step is read at the *start* of the
            # next iteration). If we pre-activated phase 2 here based on
            # step alone, the saved 2-group optimizer would mismatch our
            # 3-group optimizer at load time.
            _need_phase2 = (_peek_phase >= 2)
            if accelerator.is_main_process:
                print(f"[Resume] training_state.json: step={_peek_step}, "
                      f"phase={_peek_phase} → activating phase 2: {_need_phase2}",
                      flush=True)
            if _need_phase2:
                unwrapped = accelerator.unwrap_model(model)
                if not unwrapped._phase2_active:
                    activate_phase2_mode(unwrapped, optimizer, config, encoder_adapt,
                                         is_main=accelerator.is_main_process,
                                         text_backbone_lora=args.text_backbone_lora,
                                         text_backbone_full=args.text_backbone_full)
                    if accelerator.is_main_process:
                        n_groups = len(optimizer.param_groups)
                        print(f"[Resume] phase 2 activated (encoder_adapt={encoder_adapt}): "
                              f"optimizer now has {n_groups} groups", flush=True)

        state = ckpt_manager.load(args.resume, model, optimizer, scheduler, accelerator,
                                    miner_cache=miner.cache if miner else None)
        if state:
            global_step = state.get("global_step", 0)
            # Derive epoch + intra-epoch offset from global_step — the single
            # source of truth. steps_per_epoch is deterministic
            # (len(batch_sampler)) at a fixed world size, so this is exact.
            start_epoch = global_step // steps_per_epoch
            resume_skip = global_step % steps_per_epoch
            best_loss = state.get("best_loss", float("inf"))

            # World size MUST match: steps_per_epoch, phase_switch_step, warmup
            # and the skip math all scale with num_replicas.
            _saved_nr = state.get("num_replicas")
            if _saved_nr is not None and _saved_nr != accelerator.num_processes:
                raise RuntimeError(
                    f"World size changed on resume: checkpoint saved with "
                    f"{_saved_nr} replicas, launched with "
                    f"{accelerator.num_processes}. steps_per_epoch / "
                    f"phase_switch / LR schedule would silently rescale. "
                    f"Relaunch with {_saved_nr} processes."
                )
            if accelerator.is_main_process:
                print(f"[Resume] global_step={global_step} -> start_epoch="
                      f"{start_epoch}, skip {resume_skip} batches "
                      f"(steps_per_epoch={steps_per_epoch}, "
                      f"world={accelerator.num_processes})", flush=True)

            # ── Rebuild LR schedule when the optimizer/scheduler state was lost ──
            # If optim_state.pt was absent (e.g. deleted or never written),
            # the LR scheduler position was NOT
            # restored — it sits at last_epoch=0 and would (wrongly) restart the
            # cosine from warmup at the resumed global_step. Deterministically
            # replay the REAL wrapped scheduler the same number of times the loop
            # advanced it: it uses the exact same objects/lambdas the training
            # loop uses (phase-2 groups were already activated in the resume peek
            # above), so the final LR matches what an uninterrupted run applies at
            # global_step. LambdaLR is stateless per step, so only the final
            # position matters. Adam moments stay reset.
            #
            # COUNT: the loop increments global_step every MICRO-batch, but
            # optimizer.step()/scheduler.step() sit inside accelerator.accumulate()
            # and only advance once per grad_accum micro-batches. So the scheduler
            # was advanced global_step // grad_accum times, NOT global_step times.
            # (Normal resumes load optim_state.pt -> _optim_loaded=True -> skipped.)
            if not state.get("_optim_loaded", True) and global_step > 0:
                _accum = max(1, int(training_cfg.get("grad_accum_steps", 1)))
                _n_advances = global_step // _accum
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    for _ in range(_n_advances):
                        scheduler.step()
                if accelerator.is_main_process:
                    try:
                        _lrs = scheduler.get_last_lr()
                    except Exception:
                        _lrs = [pg["lr"] for pg in optimizer.param_groups]
                    print(f"[Resume] optim_state.pt was absent — rebuilt LR schedule "
                          f"by replaying scheduler {_n_advances} advances "
                          f"(global_step={global_step} // grad_accum={_accum}); "
                          f"Adam moments reset (self-heal). LR now: {_lrs}", flush=True)

    # ── Logger ──
    logger = TrainingLogger(
        accelerator=accelerator,
        log_every=training_cfg.get("log_every", 10),
        use_wandb=wandb_name is not None,
        total_steps=total_steps,
        phase_switch_step=phase_switch_step,
    )

    # ── Resolve percentage-based intervals to steps ──
    save_every = max(1, int(training_cfg.get("save_every_pct", 0.25) * steps_per_epoch))
    eval_every = max(1, int(training_cfg.get("eval_every_pct", 0.1) * steps_per_epoch))
    mc_eval_every = max(1, int(training_cfg.get("mc_eval_every_pct", 0.03) * steps_per_epoch))
    text_refresh_every = max(1, int(mining_cfg.get("text_refresh_pct", 0.1) * steps_per_epoch))
    media_refresh_every = max(1, int(mining_cfg.get("media_refresh_pct", 0.5) * steps_per_epoch))

    from probes.eval_scifact import evaluate_scifact
    tokens_per_encoder = config.get("audio", {}).get("tokens_per_encoder", 128)

    if accelerator.is_main_process:
        print(f"\nTotal steps: {total_steps}")
        print(f"Phase switch at step: {phase_switch_step}")
        print(f"Warmup: {warmup_steps} steps")
        mining_str = "OFF"
        if use_mining:
            parts_m = []
            if use_text_mining:
                parts_m.append(f"text every {text_refresh_every}")
            if use_media_mining:
                parts_m.append(f"media every {media_refresh_every}")
            mining_str = ", ".join(parts_m)
        print(f"Mining: {mining_str}")
        print(f"Intervals (steps): save={save_every}, scifact_eval={eval_every}, mc_eval={mc_eval_every}")
        print(f"Starting from step: {global_step}")

    # ── Teacher embedding cache ──
    # The teacher (frozen backbone over the positive caption) is a
    # deterministic function of each sample. We precompute once per
    # backbone, then look up per-step instead of re-running the backbone.
    from utils.teacher_cache import cache_dir_for, load_or_extend
    backbone_name = config["model"]["backbone"]
    teacher_cache_dir = cache_dir_for(
        backbone_name,
        suffix="orig-cap" if use_original_caption else None,
    )
    unwrapped_for_cache = accelerator.unwrap_model(model)
    teacher_cache = load_or_extend(
        cache_dir=teacher_cache_dir,
        dataset=dataset,
        model=unwrapped_for_cache,
        collator=collator,
        device=accelerator.device,
        backbone_name=backbone_name,
        batch_size=training_cfg.get("teacher_cache_batch_size", 64),
        is_main=accelerator.is_main_process,
        barrier_fn=accelerator.wait_for_everyone,
    )
    if accelerator.is_main_process:
        print(f"[TeacherCache] Ready with {len(teacher_cache)} entries "
              f"(covers {len(dataset)} samples in this run).")

    # ── Training loop ──
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

    from tqdm import tqdm

    for epoch in range(start_epoch, epochs):
        model.train()
        batch_sampler._epoch = epoch
        # Fast-forward already-completed batches on the resumed epoch only.
        # The sampler reseeds deterministically per epoch, so the skipped
        # batches are exactly the ones already trained. Skip BEFORE any model
        # call and WITHOUT touching global_step, so global_step stays exactly
        # epoch*steps_per_epoch + batch_idx across resumes.
        _skip = resume_skip if epoch == start_epoch else 0

        pbar = tqdm(
            enumerate(dataloader), total=steps_per_epoch,
            desc=f"Epoch {epoch}", disable=not accelerator.is_main_process,
        )
        for batch_idx, batch in pbar:
            if batch_idx < _skip:
                continue

            # Move batch to device (dataloader not prepared by accelerate)
            batch = {
                k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }


            unwrapped = accelerator.unwrap_model(model)

            # ── Lazy miner start (covers resume past phase 2) ──
            # Started here, NOT in the resume block, so it fires AFTER
            # DataLoader workers have already forked. Starting it earlier
            # would race the workers' fork against the miner's tokenizer
            # mutex → "RuntimeError: Already borrowed" in the worker on
            # first batch (the tokenizer Mutex held mid-encode is
            # inherited locked by the child). Workers spawn at the start
            # of `for batch in dataloader`, so by the time we reach this
            # line they've already initialized their own tokenizer copies.
            if (miner is not None and miner._thread is None
                    and unwrapped._phase2_active):
                device = next(unwrapped.parameters()).device
                _setup_miner_fns(miner, unwrapped, device, dataset, collator)
                miner.start()
                # On resume the miner cache was already restored from the
                # checkpoint (ckpt_manager.load(..., miner_cache=miner.cache)),
                # so an immediate full re-mine is redundant and can OOM when it
                # interleaves with a heavy training step (and, with a seeded
                # sampler, repeat on every resume). Periodic mining (below) still
                # fires on the normal cadence using the restored cache.
                # Opt-in via OMNI_SKIP_RESUME_REMINE=1.
                _skip_remine = (os.environ.get("OMNI_SKIP_RESUME_REMINE") == "1"
                                and len(miner.cache) > 0)
                if use_text_mining and not _skip_remine:
                    miner.request_text_refresh()
                if use_media_mining and not _skip_remine:
                    miner.request_media_refresh()
                if accelerator.is_main_process:
                    mode_parts = []
                    if use_text_mining: mode_parts.append("text")
                    if use_media_mining: mode_parts.append("media")
                    _skipmsg = (f" [resume re-mine SKIPPED, using restored cache "
                                f"of {len(miner.cache)} entries]" if _skip_remine
                                else "")
                    print(f"[Resume] Started hard-neg miner on resume "
                          f"({' + '.join(mode_parts)}) at step {global_step}"
                          f"{_skipmsg}",
                          flush=True)

            # ── Phase transition ──
            if global_step == phase_switch_step and not unwrapped._phase2_active:
                if accelerator.is_main_process:
                    print(f"\n{'='*60}")
                    print(f"PHASE 2 (encoder_adapt={encoder_adapt}) at step {global_step}")
                    print(f"{'='*60}")
                activate_phase2_mode(unwrapped, optimizer, config, encoder_adapt,
                                     is_main=accelerator.is_main_process,
                                     text_backbone_lora=args.text_backbone_lora,
                                     text_backbone_full=args.text_backbone_full)

                # LoRA adapters are injected with rank-dependent Kaiming init;
                # broadcast from rank 0 so all ranks agree (DDP would silently
                # drift otherwise). Full-FT / frozen add no randomly-initialized
                # params (encoder weights are identical pretrained across ranks),
                # so no broadcast is needed in those modes. Text-backbone LoRA
                # is also Kaiming-init'd → broadcast when it is enabled.
                # Full text-backbone unfreeze uses pretrained weights that are
                # identical across ranks → no broadcast needed for it.
                if ((encoder_adapt == "lora" or args.text_backbone_lora)
                        and accelerator.num_processes > 1
                        and torch.distributed.is_initialized()):
                    for name, p in unwrapped.named_parameters():
                        if "lora_" in name:
                            torch.distributed.broadcast(p.data, src=0)

                if accelerator.is_main_process:
                    unwrapped.print_param_summary()

                # Start hard negative miner at phase 2
                if miner is not None:
                    device = next(unwrapped.parameters()).device
                    _setup_miner_fns(miner, unwrapped, device, dataset, collator)
                    miner.start()
                    if use_text_mining:
                        miner.request_text_refresh()
                    if use_media_mining:
                        miner.request_media_refresh()
                    if accelerator.is_main_process:
                        mode_parts = []
                        if use_text_mining: mode_parts.append("text")
                        if use_media_mining: mode_parts.append("media")
                        print(f"  Hard negative miner started ({' + '.join(mode_parts)})")

            # ── Refresh miner periodically (rank 0 only) ──
            if (miner is not None and miner._thread is not None
                    and global_step > phase_switch_step):
                device = next(unwrapped.parameters()).device

                # Text mining (cheap, frequent)
                if use_text_mining and global_step % text_refresh_every == 0:
                    _setup_miner_fns(miner, unwrapped, device, dataset, collator)
                    miner.request_text_refresh()

                # Media mining (expensive, infrequent — cycles all modalities)
                if use_media_mining and global_step % media_refresh_every == 0:
                    _setup_miner_fns(miner, unwrapped, device, dataset, collator)
                    miner.request_media_refresh()

            # ── Broadcast cache to all ranks on text-refresh boundaries ──
            # The miner runs only on rank 0; other ranks need the same cache
            # so the collator can attach hard negatives to every batch.
            # Broadcast picks up whatever rank 0 has produced by this step
            # (one-cycle lag is fine — the cache accumulates over time).
            _broadcast_now = (
                any_mining and global_step > phase_switch_step
                and (
                    (use_text_mining and global_step % text_refresh_every == 0)
                    or (use_media_mining and global_step % media_refresh_every == 0)
                )
            )
            if _broadcast_now:
                if accelerator.is_main_process:
                    with neg_cache._lock:
                        snapshot = dict(neg_cache._data)
                else:
                    snapshot = None
                from accelerate.utils import broadcast_object_list
                payload = [snapshot]
                broadcast_object_list(payload, from_process=0)
                if not accelerator.is_main_process and payload[0] is not None:
                    with neg_cache._lock:
                        neg_cache._data = dict(payload[0])
                if accelerator.is_main_process and snapshot:
                    print(f"[Miner] Broadcast cache to all ranks: "
                          f"{len(snapshot)} entries")

            # ── Forward: anchor ──
            anchor_kwargs = {
                "input_ids": batch["anchor_input_ids"],
                "attention_mask": batch["anchor_attention_mask"],
            }
            if "anchor_input_features" in batch:
                anchor_kwargs["input_features"] = batch["anchor_input_features"]
                anchor_kwargs["dasheng_audio"] = batch["anchor_dasheng_audio"]
            if "anchor_image_pixel_values" in batch:
                anchor_kwargs["image_pixel_values"] = batch["anchor_image_pixel_values"]
                anchor_kwargs["image_grid_thw"] = batch["anchor_image_grid_thw"]
            if "anchor_video_pixel_values" in batch:
                anchor_kwargs["video_pixel_values"] = batch["anchor_video_pixel_values"]
                anchor_kwargs["video_grid_thw"] = batch["anchor_video_grid_thw"]
                if "anchor_video_num_frames" in batch:
                    anchor_kwargs["video_num_frames"] = batch["anchor_video_num_frames"]

            # Teacher embedding: cache lookup. The backbone is frozen, so
            # the teacher output is deterministic per sample — precomputed
            # once in utils.teacher_cache, just indexed here.
            stable_keys = [dataset.stable_key(int(sid))
                           for sid in batch["sample_ids"].tolist()]
            positive_emb = teacher_cache.get_by_keys(
                stable_keys, device=accelerator.device
            ).to(dtype=torch.bfloat16).detach()

            # Acquire forward_lock for the whole step body. Required to
            # serialize main forward+backward against the miner thread's
            # concurrent encode calls — without this, gradient
            # checkpointing's recompute (visual.forward) sees corrupted
            # state because the miner ran self.visual on its own batch
            # between our forward and backward, breaking attention-kernel
            # dispatch consistency. The miner acquires the same lock
            # per encode-batch, so it can interleave between our steps.
            from contextlib import nullcontext
            _step_lock = miner.forward_lock if miner is not None else nullcontext()
            with _step_lock:
                hard_neg_emb = None
                with torch.no_grad():
                    if "hardneg_input_ids" in batch:
                        hn_ids = batch["hardneg_input_ids"]
                        hn_mask = batch["hardneg_attention_mask"]
                        B_hn, K, S = hn_ids.shape
                        hn_out = unwrapped(
                            input_ids=hn_ids.reshape(B_hn * K, S),
                            attention_mask=hn_mask.reshape(B_hn * K, S),
                        )
                        hard_neg_emb = hn_out["embedding"].reshape(B_hn, K, -1).detach()

                # Single DDP forward pass for anchor (media encoding)
                with accelerator.accumulate(model):
                    anchor_out = model(**anchor_kwargs)
                    anchor_emb = anchor_out["embedding"]

                    # ── Loss ──
                    # positive_emb serves as both SigLIP positive and distillation target
                    # (it's the frozen text backbone's embedding of the cascaded text)
                    distill_target = positive_emb  # always distill from cascaded teacher
                    loss, per_dim_losses = criterion(
                        anchor_emb, positive_emb, hard_neg_emb,
                        distill_target=distill_target,
                    )

                    # ── Backward ──
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # ── Logging ──
            # Hit rate is per-rank: each rank queries its own batch's
            # sample_ids against its local cache (kept in sync via the
            # broadcast above). Now meaningful on all ranks, not just rank 0.
            cache_hit_rate = None
            if neg_cache is not None and "sample_ids" in batch:
                cache_hit_rate = neg_cache.hit_rate(batch["sample_ids"].tolist())

            logger.log_step(
                step=global_step,
                loss=loss.item(),
                per_dim_losses=per_dim_losses,
                lr=scheduler.get_last_lr()[0],
                temperature=criterion.siglip.temperature.item(),
                modality=batch.get("modality", "unknown"),
                phase=unwrapped._phase,
                cache_hit_rate=cache_hit_rate,
            )

            # ── Checkpoint ──
            if global_step > 0 and global_step % save_every == 0:
                if accelerator.is_main_process:
                    ckpt_manager.save(
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        best_loss=best_loss,
                        miner_cache=miner.cache if miner else None,
                    )
                    if loss.item() < best_loss:
                        best_loss = loss.item()

            # ── Media↔caption probe (cheap, frequent) ──
            if global_step > 0 and global_step % mc_eval_every == 0:
                if accelerator.is_main_process:
                    # Release fragmented allocator blocks before eval (rank 0
                    # still has training buffers live; avoids HIP allocator
                    # failures on the video eval forward).
                    torch.cuda.empty_cache()
                    from probes.eval_logs import evaluate_media_caption, format_block
                    mc_results = evaluate_media_caption(
                        model=unwrapped,
                        collator=collator,
                        device=accelerator.device,
                        whisper_fe=whisper_fe,
                        # Only probe splits that were actually loaded/trained.
                        splits=tuple(data_cfg.get("omni_sets_splits", [])),
                        is_main=True,
                    )
                    print(format_block(global_step, mc_results))
                    print()

                    if wandb_name:
                        import wandb
                        log = {"eval/mc_overall_mean_cos": mc_results.get("overall_mean_cos", 0.0)}
                        for split_name, m in mc_results.items():
                            if not isinstance(m, dict):
                                continue
                            log[f"eval/mc_{split_name}_cos"] = m["mean_cos"]
                            log[f"eval/mc_{split_name}_r1"] = m["recall@1"]
                            log[f"eval/mc_{split_name}_r5"] = m["recall@5"]
                        wandb.log(log, step=global_step)
                accelerator.wait_for_everyone()

            # ── SciFact (expensive, less frequent) ──
            if global_step > 0 and global_step % eval_every == 0:
                if accelerator.is_main_process:
                    torch.cuda.empty_cache()  # same rationale as mc_eval above
                    eval_results = evaluate_scifact(
                        unwrapped, unwrapped.tokenizer, whisper_fe,
                        device=accelerator.device,
                        tokens_per_encoder=tokens_per_encoder,
                    )
                    print(f"\n{'='*60}")
                    print(f"  SciFact @ step {global_step}")
                    print(f"  Text  NDCG@10: {eval_results['text_ndcg10']:.4f}")
                    if eval_results["speech"] is not None:  # needs cache/probes/speech_cache
                        print(f"  Speech NDCG@10: {eval_results['speech_ndcg10']:.4f}")
                        print(f"  Gap (text-speech): {eval_results['gap']:+.4f}")
                    print(f"  Eval time: {eval_results['elapsed_s']:.1f}s")
                    print(f"{'='*60}\n")

                    if wandb_name:
                        import wandb
                        wandb.log({
                            "eval/scifact_text_ndcg10": eval_results["text_ndcg10"],
                            "eval/scifact_speech_ndcg10": eval_results["speech_ndcg10"],
                            "eval/scifact_gap": eval_results["gap"],
                        }, step=global_step)
                accelerator.wait_for_everyone()

            global_step += 1

        # End of epoch save
        if accelerator.is_main_process:
            _unwrapped_for_save = accelerator.unwrap_model(model)
            ckpt_manager.save(
                model=_unwrapped_for_save,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                epoch=epoch + 1,
                loss=loss.item(),
                best_loss=best_loss,
                miner_cache=miner.cache if miner else None,
            )
            # Un-pruned per-epoch snapshot (the best_step pool keeps only 3),
            # so every epoch is kept for later export and evaluation.
            ckpt_manager.save_epoch_snapshot(
                _unwrapped_for_save, epoch=epoch + 1, step=global_step)

    # ── Cleanup ──
    if miner is not None:
        miner.stop()

    if accelerator.is_main_process:
        print("\nTraining complete!")
        if wandb_name:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
