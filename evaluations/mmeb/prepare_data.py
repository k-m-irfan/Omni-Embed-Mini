"""Auto-download MMEB-V2 source tarballs to HF cache and extract once.

VLM2Vec's image+video loaders reference files under `--data_basedir` (e.g.
`image-tasks/VOC2007/image_s0011857.jpg`). The bytes live in
TIGER-Lab/MMEB-V2 as a handful of .tar.gz tarballs.

This script is idempotent:
  1. `snapshot_download` fetches the needed tarballs into the standard HF
     dataset cache (`~/.cache/huggingface/hub/datasets--TIGER-Lab--MMEB-V2/`).
  2. First-run extraction fans each tarball into
     `~/.cache/huggingface/mmeb_v2_extracted/` using the canonical layout
     from TIGER-Lab's `dataset_setup.py`.
  3. A marker file `<extracted>/.prepared` is written after success. On
     subsequent runs the script sees the marker and exits immediately.

Footprint for our 16-task slim subset:
  - image-tasks/mmeb_v1.tar.gz               →  7.0 GB download, ~15 GB extracted
  - video-tasks/frames/video_cls.tar.gz      →  9.6 GB download
  - video-tasks/frames/video_ret.tar.gz      →  7.8 GB download

Skipped: video_qa.tar.gz-{00..04} (51.7 GB; NExTQA + EgoSchema excluded
from our slim task set), video_mret.tar.gz, visdoc-tasks/* (ViDoRe-V3 owns
visdoc), and any training-split tarballs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HF_REPO = "TIGER-Lab/MMEB-V2"
HF_REPO_TYPE = "dataset"

# Tarballs we actually need for the 16-task slim config.
TARBALLS = [
    "image-tasks/mmeb_v1.tar.gz",
    "video-tasks/frames/video_cls.tar.gz",
    "video-tasks/frames/video_ret.tar.gz",
]

EXTRACT_ROOT = Path(os.environ.get(
    "MMEB_DATA_DIR", Path.home() / ".cache" / "huggingface" / "mmeb_v2_extracted"))
MARKER = EXTRACT_ROOT / ".prepared"


def _download(force: bool = False) -> Path:
    """Ensure the needed tarballs are in HF cache; return the snapshot dir."""
    from huggingface_hub import snapshot_download

    snap_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        allow_patterns=TARBALLS,
        # Default cache is ~/.cache/huggingface/hub/ — explicit for clarity.
        cache_dir=None,
        local_files_only=False,
        force_download=force,
    )
    return Path(snap_dir)


def _extract_tarball(tar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {tar_path.name} → {out_dir}")
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(out_dir)], check=True)


def _extract_all(snap_dir: Path):
    """Mirror TIGER-Lab/MMEB-V2 dataset_setup.py layout, slim variant.

    The tarballs unpack into nested subdirs that don't match what VLM2Vec's
    yaml configs reference (image_root: image-tasks/, frame_root:
    video-tasks/frames/video_ret/<task>/). After extraction, we flatten via
    symlinks so the yaml paths resolve.

    Final layout in `EXTRACT_ROOT`:
      image-tasks/<task>/...                       (symlinks → MMEB/<task>)
      video-tasks/frames/video_cls/<task>/...      (already flat from tarball)
      video-tasks/frames/video_ret/<task>/...      (symlinks → data/ziyan/video_retrieval/<task>)
    """
    img_tar = snap_dir / "image-tasks" / "mmeb_v1.tar.gz"
    img_out = EXTRACT_ROOT / "image-tasks"
    if img_tar.exists() and not (img_out / "MMEB").exists():
        _extract_tarball(img_tar, img_out)

    cls_tar = snap_dir / "video-tasks" / "frames" / "video_cls.tar.gz"
    cls_out = EXTRACT_ROOT / "video-tasks" / "frames" / "video_cls"
    if cls_tar.exists() and not (cls_out / "HMDB51").exists():
        _extract_tarball(cls_tar, cls_out)

    ret_tar = snap_dir / "video-tasks" / "frames" / "video_ret.tar.gz"
    ret_out = EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret"
    if ret_tar.exists() and not (ret_out / "data").exists():
        _extract_tarball(ret_tar, ret_out)


def _flatten_layout():
    """Symlink nested extracted dirs to the flat paths referenced by yamls."""
    # image-tasks: MMEB/<task> → image-tasks/<task>
    mmeb_root = EXTRACT_ROOT / "image-tasks" / "MMEB"
    if mmeb_root.is_dir():
        for sub in mmeb_root.iterdir():
            if not sub.is_dir():
                continue
            link = EXTRACT_ROOT / "image-tasks" / sub.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(sub.resolve())
            print(f"  symlink: image-tasks/{sub.name} -> MMEB/{sub.name}")

    # video_ret: data/ziyan/video_retrieval/<task>/frames → video_ret/<task>
    # (The inner task dirs have an extra `frames/` level holding the
    # per-video frame subdirs. VLM2Vec yamls expect frame_root/<video_name>,
    # so we symlink past the `frames/` level.)
    ret_nested = EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret" / "data" / "ziyan" / "video_retrieval"
    ret_root = EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret"
    if ret_nested.is_dir():
        for sub in ret_nested.iterdir():
            if not sub.is_dir():
                continue
            link = ret_root / sub.name
            # If an existing symlink points at the task dir instead of its
            # `frames/` child, remove + recreate it.
            inner_frames = sub / "frames"
            target = inner_frames if inner_frames.is_dir() else sub
            if link.is_symlink() and Path(os.readlink(link)).resolve() != target.resolve():
                link.unlink()
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(target.resolve())
            print(f"  symlink: video_ret/{sub.name} -> {target.relative_to(ret_root)}")


def ensure_prepared(force: bool = False) -> Path:
    """Download + extract + symlink-flatten. Idempotent. Returns EXTRACT_ROOT."""
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    if MARKER.exists() and not force:
        return EXTRACT_ROOT

    print(f"[prepare_data] downloading MMEB-V2 tarballs → HF cache …", flush=True)
    snap_dir = _download(force=force)
    print(f"[prepare_data] snapshot dir: {snap_dir}", flush=True)
    print(f"[prepare_data] extracting to {EXTRACT_ROOT} …", flush=True)
    _extract_all(snap_dir)
    print(f"[prepare_data] flattening layout …", flush=True)
    _flatten_layout()

    # Hard sanity: confirm the flat paths VLM2Vec yamls reference now exist
    # AND that a video_ret task has an inner <video_name>/frame_*.jpg file
    # (so the per-task frame_root path resolves to frames, not the extra
    # `frames/` subdir layer).
    expected = [
        EXTRACT_ROOT / "image-tasks" / "VOC2007",
        EXTRACT_ROOT / "image-tasks" / "Country211",
        EXTRACT_ROOT / "video-tasks" / "frames" / "video_cls" / "HMDB51",
        EXTRACT_ROOT / "video-tasks" / "frames" / "video_cls" / "UCF101",
        EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret" / "MSR-VTT",
        EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret" / "DiDeMo",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"[prepare_data] expected dirs not found after extract+flatten: {missing}\n"
            "Re-run with --force to redo the download and extraction."
        )
    # Verify video_ret has frame files under the per-video subdir layout.
    sample_video_dir = EXTRACT_ROOT / "video-tasks" / "frames" / "video_ret" / "MSR-VTT"
    video_dirs = [d for d in sample_video_dir.iterdir() if d.is_dir()] if sample_video_dir.is_dir() else []
    if not video_dirs or not any(video_dirs[0].glob("*.jpg")):
        raise RuntimeError(
            f"[prepare_data] MSR-VTT frame layout invalid — expected "
            f"{sample_video_dir}/<video_name>/*.jpg but none found. "
            f"Delete extract dir and re-run with --force."
        )
    MARKER.write_text("ok\n")
    print(f"[prepare_data] marker written: {MARKER}", flush=True)
    return EXTRACT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Re-download and re-extract even if marker exists.")
    args = ap.parse_args()
    root = ensure_prepared(force=args.force)
    print(f"[prepare_data] ready: {root}")


if __name__ == "__main__":
    main()
