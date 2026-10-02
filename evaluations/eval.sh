#!/usr/bin/env bash
# Evaluate an Omni-Embed-Mini model on MTEB-v2, MAEB, MMEB-V2 and ViDoRe-V3.
#
#   bash evaluations/eval.sh --model <hf_id|export_dir> [--bench all|mteb|maeb|mmeb|vidore]
#                            [--tasks T1,T2,...] [--n_gpus 8] [--force]
#
# Tasks run in parallel, one per GPU. Results are cached under
# evaluations/results/ (finished tasks are skipped on re-run) and a per-task /
# per-modality summary is printed at the end.
#
# Examples:
#   bash evaluations/eval.sh --model MBZUAI/Omni-Embed-Mini-0.9B
#   bash evaluations/eval.sh --model MBZUAI/Omni-Embed-Mini-2.3B --bench maeb
#   bash evaluations/eval.sh --model exported/omni-embed-mini-0.9b-4 --tasks SciFact,VOC2007
#   OMNI_TRUNCATE_DIM=256 bash evaluations/eval.sh --model ... # Matryoshka dimension
set -uo pipefail   # no -e: one failed task should not stop the others

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=""; BENCH="all"; TASKS=""; N_GPUS=8; FORCE=""
RESULTS_DIR="${RESULTS_DIR:-$EVAL_DIR/results}"
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --bench)   BENCH="$2"; shift 2 ;;
    --tasks)   TASKS="$2"; shift 2 ;;
    --n_gpus)  N_GPUS="$2"; shift 2 ;;
    --force)   FORCE="--force"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }
# Local export dirs are passed as absolute paths.
[ -d "$MODEL" ] && MODEL="$(cd "$MODEL" && pwd)"

# Task list: explicit --tasks, else every task of the selected benchmark(s).
if [ -z "$TASKS" ]; then
  TASKS=$(cd "$EVAL_DIR" && python -c "
import tasks as T
b = '$BENCH'
names = list(T.BENCHMARKS) if b == 'all' else b.split(',')
print(','.join(t for n in names for t in T.BENCHMARKS[n]))")
fi
IFS=',' read -ra TASK_ARR <<< "$TASKS"

# Download the model once before fanning out (avoids parallel-download races).
python -c "import sys; sys.path.insert(0, '$EVAL_DIR'); from load_model import resolve_model; print('model dir:', resolve_model('$MODEL')[0])" || exit 1

export TORCHDYNAMO_DISABLE=1          # eager cosine-similarity kernels (identical results)
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
SAFE="$(basename "$MODEL")"
LOG_DIR="$RESULTS_DIR/logs/$SAFE"; mkdir -p "$LOG_DIR"
MMEB_OUT="$RESULTS_DIR/mmeb/OmniEmbed[$SAFE]"

is_mmeb() { (cd "$EVAL_DIR" && python -c "import tasks,sys; sys.exit(0 if '$1' in tasks.MMEB_TASKS else 1)"); }

echo "=== evaluating $MODEL on ${#TASK_ARR[@]} task(s), $N_GPUS GPU(s) ==="
pids=(); names=()
for i in "${!TASK_ARR[@]}"; do
  t="${TASK_ARR[i]}"
  while [ "$(jobs -rp | wc -l)" -ge "$N_GPUS" ]; do sleep 2; done
  gpu=$(( i % N_GPUS ))
  miopen="/tmp/miopen-${USER}-${SAFE}-${t}-$$"; mkdir -p "$miopen"
  if is_mmeb "$t"; then
    [ -n "$FORCE" ] && rm -f "$MMEB_OUT/${t}_score.json"
    cmd=(python "$EVAL_DIR/mmeb/run_mmeb.py" --model "$MODEL" --tasks "$t" --out_dir "$MMEB_OUT")
  else
    cmd=(python "$EVAL_DIR/run_mteb.py" --model "$MODEL" --tasks "$t" --results_dir "$RESULTS_DIR" $FORCE)
    # MTEB-v2 text retrieval runs batched; media tasks encode one item at a time.
    case "$t" in ArguAna|CQADupstack*|FiQA2018|NFCorpus|SCIDOCS|SciFact) cmd+=(--batch_size 4) ;; esac
  fi
  CUDA_VISIBLE_DEVICES=$gpu HIP_VISIBLE_DEVICES=$gpu \
  MIOPEN_USER_DB_PATH="$miopen" MIOPEN_CUSTOM_CACHE_DIR="$miopen" \
    "${cmd[@]}" > "$LOG_DIR/$t.log" 2>&1 &
  pids+=($!); names+=("$t")
done
fail=0
for k in "${!pids[@]}"; do
  if ! wait "${pids[k]}"; then fail=1; echo "FAILED: ${names[k]} (see $LOG_DIR/${names[k]}.log)"; fi
done

python "$EVAL_DIR/summarize.py" --model "$MODEL" --results_dir "$RESULTS_DIR"
exit $fail
