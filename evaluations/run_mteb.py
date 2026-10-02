"""Evaluate an Omni-Embed-Mini model on mteb-hosted tasks (MTEB-v2, MAEB, ViDoRe-V3).

One process evaluates one or more tasks on a single device; eval.sh fans tasks
out across GPUs. Per-task results are written by mteb to
<results_dir>/mteb/results/<model>/<revision>/<task>.json and finished tasks
are skipped on re-run (pass --force to recompute).

    python evaluations/run_mteb.py --model MBZUAI/Omni-Embed-Mini-0.9B --tasks SciFact
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# datasets' default audio decoder (torchcodec) is unavailable on ROCm; decode
# with soundfile/librosa instead. Must be imported before mteb/datasets load.
import patch_audio  # noqa: E402,F401
import patch_mteb  # noqa: E402,F401  (mteb 2.12.13 task-side fixes)

from load_model import load_wrapper  # noqa: E402
from mrl_wrap import maybe_wrap_for_mrl  # noqa: E402
from tasks import VIDORE_LANGUAGES, VIDORE_TASKS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local export dir")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--results_dir", default=str(EVAL_DIR / "results"))
    ap.add_argument("--force", action="store_true", help="recompute cached tasks")
    args = ap.parse_args()

    import mteb
    from mteb import ResultCache

    cache_dir = Path(args.results_dir) / "mteb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(cache_path=cache_dir)

    print(f"[pid {os.getpid()}] model={args.model} device={args.device} tasks={args.tasks}")
    model = load_wrapper(args.model, device=args.device)
    model = maybe_wrap_for_mrl(model)  # no-op unless OMNI_TRUNCATE_DIM is set

    tasks = mteb.get_tasks(tasks=args.tasks)
    # ViDoRe-V3: English subsets only.
    tasks = [t.filter_languages(VIDORE_LANGUAGES) if t.metadata.name in VIDORE_TASKS else t
             for t in tasks]
    mteb.evaluate(
        model, tasks,
        cache=cache,
        overwrite_strategy="always" if args.force else "only-missing",
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    main()
