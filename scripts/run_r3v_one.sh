#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DS="${1:?usage: run_r3v_one.sh <DATASET>}"
PY="${PY:-python3}"; GPU="${GPU:-2}"; WORKERS="${WORKERS:-5}"
STATUS="${STATUS:-runs/r3v_full_20261026/${DS}.status}"
mkdir -p "$(dirname "$STATUS")"
log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$STATUS"; }
ds_log="runs/r3v_full_20261026/${DS}.log"
log "start dataset=$DS workers=$WORKERS gpu=$GPU"
"$PY" scripts/run_ablation_matrix.py \
  --dataset "$DS" --ratio 0.02 --seeds 777 888 999 \
  --arms full no_history_context --workers "$WORKERS" --gpu "$GPU" \
  --finetune-scope layerwise --finetune-epochs 200 \
  --finetune-batch-size 32 --finetune-lr 0.0001 \
  --finetune-graph-lr 0.001 --finetune-backbone-lr-scale 0.1 \
  --finetune-max-steps 1200 --finetune-patience 8 \
  --finetune-eval-metric auc --zero-init-gate --gate-init-bias -2.0 \
  --context-stage after_context --graph-window 30 --neighbor-count 16 \
  --external-val \
  --output-root save/ablations_r3v_20261026 \
  --log-root log/ablations_r3v_20261026 \
  --fig-root figure/ablations_r3v_20261026 >> "$ds_log" 2>&1
log "end dataset=$DS"
