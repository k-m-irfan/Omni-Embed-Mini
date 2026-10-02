"""Evaluate an Omni-Embed-Mini model on the MMEB-V2 image + video subset.

Uses VLM2Vec's evaluation-dataset parsers (vendored under ./vlm2vec, minimal
subset) to build each task's (query, candidate) pairs, encodes them with the
Omni-Embed-Mini wrapper, and writes <out_dir>/<task>_score.json (hit@1 etc.).

Video is scored by encoding the 8 pre-extracted frames per clip as images and
mean-pooling them. Images/frames come from evaluations/mmeb/prepare_data.py.

    python evaluations/mmeb/run_mmeb.py --model MBZUAI/Omni-Embed-Mini-0.9B \
        --tasks VOC2007,MSR-VTT --out_dir evaluations/results/mmeb/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
EVAL_ROOT = THIS_DIR.parent
VLM2VEC_ROOT = THIS_DIR / "vlm2vec"
DEFAULT_DATA_DIR = Path(os.environ.get(
    "MMEB_DATA_DIR", Path.home() / ".cache" / "huggingface" / "mmeb_v2_extracted"))
sys.path.insert(0, str(VLM2VEC_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

# Stub baseline_backbone submodules that processor.py top-level imports
# (we never invoke them through this runner — the omni model never goes through
# VLM2Vec's MMEBModel). Keeping the parent as a real package via a namespace stub.
import types as _types

class _DummyProcessor:
    def __init__(self, *a, **k):
        raise RuntimeError("Baseline backbone stub — not used by run_mmeb.py")

def _stub(modpath: str, attrs: dict):
    m = _types.ModuleType(modpath)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[modpath] = m

_pkg = _types.ModuleType("src.model.baseline_backbone")
_pkg.__path__ = []  # mark as a package so submodule lookups work
sys.modules["src.model.baseline_backbone"] = _pkg
_stub("src.model.baseline_backbone.colpali",
      {"ColPaliProcessor": _DummyProcessor, "ColPali": _DummyProcessor})
_stub("src.model.baseline_backbone.llava_next",
      {"LlavaNextForConditionalGeneration": _DummyProcessor})
_stub("src.model.baseline_backbone.phi3_v", {})
_stub("src.model.baseline_backbone.phi3_v.modeling_phi3_v",
      {"Phi3VForCausalLM": _DummyProcessor})
_stub("src.model.baseline_backbone.internvideo2", {})
_stub("src.model.baseline_backbone.internvideo2.modeling_internvideo2",
      {"InternVideo2_Stage2": _DummyProcessor})

# Stub `src.model.processor` so dataset parsers can `from src.model.processor
# import process_input_text` without triggering the transformers-pinned vendor
# chain (qwen2_vl modeling needs SlidingWindowCache; llava_next needs older
# transformers). For our use, parsers run with model_backbone="gme", which
# returns clean `instruction + " " + text` (no special tokens).
_processor_stub = _types.ModuleType("src.model.processor")

GME = "gme"
LamRA = "lamra"
LamRA_QWEN2_5 = "lamra_qwen2_5"
COLPALI = "colpali"

def _process_input_text(instruction, model_backbone, text=None,
                        add_video_token=False, add_image_token=False):
    if text:
        return (instruction or "") + " " + text
    return (instruction or "") + " "

_processor_stub.process_input_text = _process_input_text
_processor_stub.GME = GME
_processor_stub.LamRA = LamRA
_processor_stub.LamRA_QWEN2_5 = LamRA_QWEN2_5
_processor_stub.COLPALI = COLPALI
# Empty token maps so any stray reference falls back to "" and parsers don't crash.
_processor_stub.VLM_IMAGE_TOKENS = {GME: "", "qwen2_vl": "", "qwen2_5_vl": ""}
_processor_stub.VLM_VIDEO_TOKENS = {GME: "", "qwen2_vl": "", "qwen2_5_vl": ""}
_processor_stub.PROMPT_TEMPLATE_DICT = {}
_pkg_model = _types.ModuleType("src.model")
_pkg_model.__path__ = []
sys.modules.setdefault("src.model", _pkg_model)
sys.modules["src.model.processor"] = _processor_stub

# The eval_dataset/__init__.py imports every parser including video-QA ones
# that need cv2/decord. Pre-register the package as a stub so that the
# `from src.data.eval_dataset.X import Y` statements still resolve via the
# real .py files on disk, but the __init__.py itself never runs.
_pkg_data = _types.ModuleType("src.data")
_pkg_data.__path__ = [str(VLM2VEC_ROOT / "src" / "data")]
sys.modules.setdefault("src.data", _pkg_data)
_pkg_eds = _types.ModuleType("src.data.eval_dataset")
_pkg_eds.__path__ = [str(VLM2VEC_ROOT / "src" / "data" / "eval_dataset")]
sys.modules["src.data.eval_dataset"] = _pkg_eds
# A `src.utils` shim for the same reason — parser modules import from it.
_pkg_utils = _types.ModuleType("src.utils")
_pkg_utils.__path__ = [str(VLM2VEC_ROOT / "src" / "utils")]
sys.modules.setdefault("src.utils", _pkg_utils)

# Now load each parser module directly.
import logging as _logging
import src.data.eval_dataset.base_eval_dataset      # noqa: F401  (registry init)
import src.data.eval_dataset.image_cls_dataset       # noqa: F401
import src.data.eval_dataset.image_qa_dataset        # noqa: F401
import src.data.eval_dataset.image_t2i_eval          # noqa: F401
import src.data.eval_dataset.image_i2t_eval          # noqa: F401
import src.data.eval_dataset.image_i2i_vg_dataset    # noqa: F401
import src.data.eval_dataset.video_classification_datasets  # noqa: F401
import src.data.eval_dataset.msrvtt_dataset          # noqa: F401
import src.data.eval_dataset.msvd_dataset            # noqa: F401
import src.data.eval_dataset.didemo_dataset          # noqa: F401
import src.data.eval_dataset.vatex_dataset           # noqa: F401
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset,
    generate_cand_dataset,
)
from src.utils.eval_utils.metrics import RankingMetrics
for _name in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)

from load_model import load_wrapper


# Pretend to be `gme` so process_input_text returns plain "instruction text".
class _ModelArgs:
    model_backbone = "gme"
    model_name = ""  # unused by parsers
    model_type = ""


class _DataArgs:
    image_resolution = "high"
    data_basedir = None
    encode_output_path = ""
    dataset_config = ""


VIDEO_PARSERS = {
    "video_classification", "msrvtt", "msvd", "didemo", "vatex",
    "videocls", "moment_retrieval", "momentseeker",
    "ssv2", "mvbench", "videomme", "videommmu", "youcook2",
    "activitynetqa", "nextqa", "egoschema",
}

def _ensure_data_paths(task_config: dict, data_basedir: str | None):
    if not data_basedir:
        return task_config
    for key in ("image_root", "video_root", "frame_root", "clip_root", "data_path"):
        if task_config.get(key):
            task_config[key] = os.path.join(data_basedir, task_config[key])
    return task_config


def _entry_paths(entry):
    """Pull frame/image paths out of either an ImageVideoInstance-dict or raw dict."""
    if entry is None:
        return []
    if hasattr(entry, "to_dict"):
        entry = entry.to_dict()
    if isinstance(entry, dict):
        return list(entry.get("paths") or [])
    return []


def _load_pils(paths):
    """Open each path as RGB PIL. Raises on missing/unreadable — no silent
    fallback: if data is broken we want to fix it, not train/score on gaps."""
    pils = []
    for p in paths:
        if not p:
            raise FileNotFoundError(f"empty image path in paths={paths!r}")
        pils.append(Image.open(p).convert("RGB"))
    return pils


def _encode_payloads(model, payloads, side: str, task_name: str | None = None):
    """Encode a list of {"text": str | None, "frames": [PIL, ...]} payloads."""
    return _encode_payloads_omni(model, payloads, side)


def _maybe_truncate_renorm(emb: np.ndarray) -> np.ndarray:
    """Apply OMNI_TRUNCATE_DIM if set — truncate + L2-renormalize.

    Applied to every encoded payload so MRL eval works on MMEB. No-op when
    the env var is unset.

    """
    import os as _os
    raw = _os.environ.get("OMNI_TRUNCATE_DIM")
    if not raw:
        return emb
    d = int(raw)
    x = emb[..., :d]
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.clip(n, 1e-12, None)).astype(np.float32)


def _encode_payloads_omni(model, payloads, side: str):
    """Per-item encode via OmniEmbedMTEBWrapper.

    MMEB tasks are always cross-modal, so text-only payloads use the chat
    recipe on both sides (same pooling subspace as media: text would pool at
    <|endoftext|> under the native recipe vs <|im_end|> for media).

    For video, per-frame embed + mean-pool keeps scoring comparable across
    checkpoints. No silent fallbacks: if the model lacks a recipe method for
    text, we crash rather than silently using the wrong one.
    """
    import os as _os
    is_query = (side == "qry")
    _is_native = getattr(model.model, "vision_mode", "custom") == "native"
    _default = "chat" if _is_native else "native"
    q_recipe = _os.environ.get("OMNI_QUERY_RECIPE", _default).lower()
    c_recipe = _os.environ.get("OMNI_CORPUS_RECIPE", _default).lower()
    # MMEB is always multimodal (image/video tasks): force chat for BOTH
    # sides to keep query text and candidate images in the same pooling
    # subspace. The text-only query payloads don't have frames, but the
    # task itself is cross-modal — matching omni_embed_wrapper.py's
    # task-level modality override.
    q_recipe = c_recipe = "chat"
    # Native backbone (2.3B): MMEB is always image/video (never audio), so
    # use Qwen3-VL's full ST chat template (system+user+assistant), the
    # backbone's contrastive-pretraining format. Custom (0.9B) ignores the
    # flag (gated on vision_mode inside the processor / wrapper text path).
    if _is_native:
        _os.environ["OMNI_NATIVE_FULL_CHAT"] = "1"
    recipe = q_recipe if is_query else c_recipe
    if recipe not in ("native", "chat"):
        raise RuntimeError(
            f"OMNI_{'QUERY' if is_query else 'CORPUS'}_RECIPE must be "
            f"'native' or 'chat', got {recipe!r}"
        )

    # Batch text-only payloads through _encode_text_batch / _encode_chat_text_batch
    # (matches omni_embed_wrapper.encode()). Media items stay per-item.
    out = [None] * len(payloads)
    text_only_idx, text_only_strings = [], []
    media_idx = []
    for i, p in enumerate(payloads):
        frames = p.get("frames") or []
        if not frames:
            text_only_idx.append(i)
            text_only_strings.append((p.get("text") or "").strip())
        else:
            media_idx.append(i)

    # Text-only: batched recipe path. Chunk to the wrapper's batch_size so
    # large corpora (e.g. VisualNews_i2t with ~20k text candidates) don't OOM
    # from a single all-text tokenize_text_native call.
    if text_only_strings:
        if recipe == "native":
            fn = getattr(model, "_encode_text_batch", None)
            if fn is None:
                raise RuntimeError(
                    "wrapper missing _encode_text_batch for native recipe"
                )
            def _encode_chunk(chunk):
                return fn(chunk, is_query=is_query, instruction=None)
        else:  # chat
            fn = getattr(model, "_encode_chat_text_batch", None)
            if fn is None:
                raise RuntimeError(
                    "wrapper missing _encode_chat_text_batch for chat recipe"
                )
            def _encode_chunk(chunk):
                return fn(chunk)
        bs = getattr(model, "batch_size", 8)
        emb_chunks = []
        for i in range(0, len(text_only_strings), bs):
            emb_chunks.append(_encode_chunk(text_only_strings[i:i + bs]))
        emb = np.concatenate(emb_chunks, axis=0)
        for idx, row in zip(text_only_idx, emb):
            out[idx] = np.asarray(row, dtype=np.float32).reshape(-1)

    # Media: per-item through processor (always chat-wrapped internally).
    for i in media_idx:
        p = payloads[i]
        text = (p.get("text") or "").strip()
        frames = p["frames"]
        if len(frames) > 1:
            vecs = []
            for f in frames:
                media = {"image": f}
                if text:
                    media["text"] = text
                v = model._encode_one(is_query=is_query, **media)
                vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
            out[i] = np.stack(vecs, axis=0).mean(axis=0).astype(np.float32)
        else:
            media = {"image": frames[0]}
            if text:
                media["text"] = text
            v = model._encode_one(is_query=is_query, **media)
            out[i] = np.asarray(v, dtype=np.float32).reshape(-1)

    # Sanity: every slot filled.
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise RuntimeError(f"encode left {len(missing)} payload slots empty")
    return _maybe_truncate_renorm(np.stack(out, axis=0))


def _build_payloads(rows, side: str, is_video: bool):
    """side='qry' uses row['query_text'][0] + row['query_image'][0] paths.
       side='cand' uses row['cand_text'][i] + row['cand_image'][i] paths.

    For 'cand' we flatten all candidates from all rows (caller is responsible
    for pairing back via cand_names — only used in local-eval mode).
    """
    out = []
    for row in rows:
        if side == "qry":
            text = (row["query_text"][0] if row.get("query_text") else "") or ""
            entry = row["query_image"][0] if row.get("query_image") else None
            paths = _entry_paths(entry)
            out.append({"text": text.strip(), "frames": _load_pils(paths)})
        else:
            for c_text, c_entry in zip(row["cand_text"], row["cand_image"]):
                paths = _entry_paths(c_entry)
                out.append({"text": (c_text or "").strip(),
                            "frames": _load_pils(paths)})
    return out


def _score_global(qry_embeds, cand_embeds, cand_keys, gt_infos, metrics):
    cosine = qry_embeds @ cand_embeds.T
    ranked = np.argsort(-cosine, axis=1)
    pred_dicts = []
    for ranked_row, gt in zip(ranked, gt_infos):
        rel = gt["label_name"] if isinstance(gt["label_name"], list) else [gt["label_name"]]
        pred_dicts.append({
            "prediction": [cand_keys[i] for i in ranked_row],
            "label": rel,
            "rel_scores": gt.get("rel_scores"),
        })
    return metrics.evaluate(pred_dicts), pred_dicts


def _score_local(qry_embeds, cand_dataset_rows, gt_infos, metrics, cand_to_emb):
    pred_dicts = []
    for q_emb, gt in zip(qry_embeds, gt_infos):
        cand_names = gt["cand_names"]
        cand_embeds = np.stack([cand_to_emb[n] for n in cand_names])
        cosine = q_emb @ cand_embeds.T
        ranked = np.argsort(-cosine)
        rel = gt["label_name"] if isinstance(gt["label_name"], list) else [gt["label_name"]]
        pred_dicts.append({
            "prediction": [cand_names[i] for i in ranked],
            "label": rel,
            "rel_scores": gt.get("rel_scores"),
        })
    return metrics.evaluate(pred_dicts), pred_dicts


def _run_one_task(model, dataset_name, task_config, data_basedir, encode_dir):
    score_path = encode_dir / f"{dataset_name}_score.json"
    if score_path.exists():
        print(f"[skip] {dataset_name} — {score_path} exists")
        return

    parser = task_config["dataset_parser"]
    is_video = parser in VIDEO_PARSERS
    task_config = _ensure_data_paths(dict(task_config), data_basedir)
    model_args, data_args = _ModelArgs(), _DataArgs()
    data_args.data_basedir = data_basedir
    print(f"\n=== {dataset_name} (parser={parser}) ===", flush=True)
    t0 = time.time()
    qry_dataset, corpus = AutoEvalPairDataset.instantiate(
        model_args=model_args, data_args=data_args, **task_config
    )
    cand_dataset = generate_cand_dataset(qry_dataset, corpus)
    print(f"  loaded: {len(qry_dataset)} queries, {len(cand_dataset)} candidates "
          f"(parser took {time.time()-t0:.1f}s)", flush=True)

    qry_payloads = _build_payloads(qry_dataset, "qry", is_video)
    qry_embeds = _encode_payloads(model, qry_payloads, "qry",
                                  task_name=dataset_name)
    print(f"  qry encoded: {qry_embeds.shape}", flush=True)

    cand_payloads = _build_payloads(cand_dataset, "cand", is_video)
    cand_embeds = _encode_payloads(model, cand_payloads, "cand",
                                   task_name=dataset_name)
    print(f"  cand encoded: {cand_embeds.shape}", flush=True)

    cand_keys = []
    for row in cand_dataset:
        cand_keys.append(row["dataset_infos"]["cand_name"])
    cand_to_emb = dict(zip(cand_keys, cand_embeds))

    # L2-normalize for cosine sim.
    def _norm(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return x / n

    qry_embeds = _norm(qry_embeds)
    cand_embeds_n = _norm(np.stack(list(cand_to_emb.values()), axis=0))
    cand_to_emb = dict(zip(cand_to_emb.keys(), cand_embeds_n))

    gt_infos = [r["dataset_infos"] for r in qry_dataset]
    metrics_to_report = task_config.get("metrics") or [
        "hit", "ndcg", "precision", "recall", "f1", "map", "mrr"
    ]
    metrics = RankingMetrics(metrics_to_report)

    eval_type = task_config.get("eval_type", "global")
    if eval_type == "global":
        score_dict, pred_dicts = _score_global(
            qry_embeds, cand_embeds_n, list(cand_to_emb.keys()), gt_infos, metrics)
    else:
        score_dict, pred_dicts = _score_local(
            qry_embeds, cand_dataset, gt_infos, metrics, cand_to_emb)

    score_dict["num_pred"] = len(pred_dicts)
    score_dict["num_data"] = len(gt_infos)
    score_dict["wall_seconds"] = round(time.time() - t0, 1)
    print(f"  scores: { {k: round(v,4) for k,v in score_dict.items() if isinstance(v,(int,float))} }",
          flush=True)
    score_path.write_text(json.dumps(score_dict, indent=2))
    pred_path = encode_dir / f"{dataset_name}_pred.jsonl"
    with open(pred_path, "w") as f:
        for p in pred_dicts:
            f.write(json.dumps(p) + "\n")
    print(f"  wrote {score_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local export dir.")
    ap.add_argument("--tasks", required=True, help="Comma-separated MMEB task names.")
    ap.add_argument("--data_basedir", default=str(DEFAULT_DATA_DIR),
                    help="Where prepare_data.py extracted the MMEB-V2 images/frames.")
    ap.add_argument("--out_dir", required=True, help="Output dir for <task>_score.json.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    encode_dir = Path(args.out_dir)
    encode_dir.mkdir(parents=True, exist_ok=True)

    yaml_cfg = {}
    for name in ("image_en.yaml", "video_en.yaml"):
        yaml_cfg.update(yaml.safe_load((THIS_DIR / name).read_text()))
    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    missing = [t for t in requested if t not in yaml_cfg]
    if missing:
        raise SystemExit(f"unknown MMEB tasks: {missing}")

    print(f"loading {args.model} on {args.device}", flush=True)
    model = load_wrapper(args.model, device=args.device, batch_size=args.batch_size)
    print("model ready", flush=True)

    for t in requested:
        _run_one_task(model, t, yaml_cfg[t], args.data_basedir, encode_dir)


if __name__ == "__main__":
    main()
