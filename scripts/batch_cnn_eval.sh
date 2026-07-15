#!/usr/bin/env bash
# Batch runner for the out-of-sample CNN highlight-clip evaluation.
#
# Intended for the *batch* machine (not the interactive box). It sweeps the
# train/test split seed (and optionally epochs) and writes each run's
# metrics.json under a per-seed output directory, then aggregates a summary.
#
# Usage:
#   ./scripts/batch_cnn_eval.sh [OUTPUT_ROOT] [EPOCHS]
#
# Requires: a venv with torch at .venv (or edit VENV below) and the
# data/processed/highlights store mounted/accessible on this machine.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_DIR/.venv/bin/activate}"
OUTPUT_ROOT="${1:-$REPO_DIR/data/processed/highlights/training/cnn_eval_sweep}"
EPOCHS="${2:-10}"
SEEDS="${SEEDS:-0 1 2 3 4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/summary.jsonl"
: > "$SUMMARY"

source "$VENV"

for seed in $SEEDS; do
  run_dir="$OUTPUT_ROOT/seed_$seed"
  echo ">>> seed=$seed epochs=$EPOCHS -> $run_dir"
  OMP_NUM_THREADS="$OMP_NUM_THREADS" python "$REPO_DIR/scripts/evaluate_cnn.py" \
    --epochs "$EPOCHS" \
    --batch-size 8 \
    --seed "$seed" \
    --output-dir "$run_dir"
  if [ -f "$run_dir/metrics.json" ]; then
    echo "{\"seed\": $seed, \"epochs\": $EPOCHS, \"metrics\": $(cat "$run_dir/metrics.json")}" >> "$SUMMARY"
  fi
done

echo "wrote $SUMMARY"
