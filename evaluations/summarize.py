"""Print per-task scores and per-modality means for an evaluated model.

    python evaluations/summarize.py --model MBZUAI/Omni-Embed-Mini-0.9B

Main metrics: MTEB-v2 / ViDoRe-V3 nDCG@10, MAEB task main score (mean over
subsets), MMEB-V2 hit@1. All reported x100.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from load_model import cache_name  # noqa: E402
from tasks import MODALITIES  # noqa: E402


def _mteb_score(path: Path) -> float | None:
    d = json.loads(path.read_text())
    for split in ("test", "train", "validation", "dev"):
        rows = [r for r in d.get("scores", {}).get(split, []) or []
                if isinstance(r, dict) and "main_score" in r]
        if not rows:
            continue
        english = [r for r in rows if r.get("hf_subset") == "english"]
        rows = english or rows                      # ViDoRe: English subset
        return 100 * sum(r["main_score"] for r in rows) / len(rows)
    return None


def collect(model: str, results_dir: Path) -> dict[str, float]:
    name = cache_name(model)
    scores: dict[str, float] = {}
    mteb_root = results_dir / "mteb" / "results" / name
    if mteb_root.is_dir():
        # newest revision dir wins if a model was evaluated more than once
        for rev in sorted(mteb_root.iterdir(), key=lambda p: p.stat().st_mtime):
            for f in rev.glob("*.json"):
                if f.name != "model_meta.json":
                    s = _mteb_score(f)
                    if s is not None:
                        scores[f.stem] = s
    mmeb_root = results_dir / "mmeb" / name
    if mmeb_root.is_dir():
        for f in mmeb_root.glob("*_score.json"):
            d = json.loads(f.read_text())
            if "hit@1" in d:
                scores[f.name[:-len("_score.json")]] = 100 * d["hit@1"]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results_dir", default=str(EVAL_DIR / "results"))
    ap.add_argument("--json", help="also write the summary to this file")
    args = ap.parse_args()

    scores = collect(args.model, Path(args.results_dir))
    print(f"\n{args.model}")
    summary = {"model": args.model, "tasks": {}, "modalities": {}}
    for mod, bench, tasks in MODALITIES:
        done = [t for t in tasks if t in scores]
        if not done:
            continue
        print(f"\n[{mod}]  ({bench}, {len(done)}/{len(tasks)} tasks)")
        for t in done:
            print(f"  {t:<40s} {scores[t]:6.2f}")
            summary["tasks"][t] = round(scores[t], 2)
        mean = sum(scores[t] for t in done) / len(done)
        partial = "" if len(done) == len(tasks) else "  (partial)"
        print(f"  {'MEAN':<40s} {mean:6.2f}{partial}")
        summary["modalities"][mod] = {"mean": round(mean, 2), "n_tasks": len(done),
                                      "complete": len(done) == len(tasks)}
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
