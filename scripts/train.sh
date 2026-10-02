#!/usr/bin/env bash
# Launch Omni-Embed-Mini training with Accelerate (bf16, data-parallel).
#
#   bash scripts/train.sh <config.yaml> [train.py flags...]
#
# Examples:
#   bash scripts/train.sh configs/omni_embed_mini_0.9b.yaml --text_mining --media_mining
#   bash scripts/train.sh configs/omni_embed_mini_2.3b.yaml --text_mining --media_mining
#   NUM_GPUS=4 bash scripts/train.sh configs/omni_embed_mini_0.9b.yaml --fresh
#
# Useful train.py flags:
#   --text_mining / --media_mining   enable hard-negative mining (released models use both)
#   --fresh                          ignore <output_dir>/latest and start from scratch
#   --resume <dir>                   resume from a specific checkpoint (default: auto-resume)
#   --data_pct <f>                   train on a fraction of Omni-Sets
#   --override key.path=value        override any config value (repeatable)
#   --wandb_run_name <name>          log to Weights & Biases
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?usage: bash scripts/train.sh <config.yaml> [train.py flags...]}"
shift
NUM_GPUS="${NUM_GPUS:-8}"

# Silence TensorFlow import noise pulled in by optional dependencies.
export TF_CPP_MIN_LOG_LEVEL=3

# AMD ROCm: per-job MIOpen cache (avoids db-write contention between ranks)
# and a less fragmenting allocator. Harmless on NVIDIA.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/miopen-${USER}-$$}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_LOG_LEVEL=0
export HIPFFT_PLAN_CACHE_MAX_SIZE=0
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.6}"

# The 2.3B backbone has long multimodal sequences; embed mining captions in
# small batches to bound memory (does not change which negatives are mined).
case "$CONFIG" in
  *2.3b*) export OMNI_MINE_EMBED_BS="${OMNI_MINE_EMBED_BS:-8}" ;;
esac

echo "=== Omni-Embed-Mini training | config=$CONFIG | gpus=$NUM_GPUS | args: $* ==="
accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    train.py --config "$CONFIG" "$@"
