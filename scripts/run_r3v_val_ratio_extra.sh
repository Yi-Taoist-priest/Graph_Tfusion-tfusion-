#!/usr/bin/env bash
# validation 阈值 + 比例 sweep 的其余 4 个消融臂
#   no_retrieval / no_gated_residual / no_modal_attn / no_pre_train
#   比例：2% 优先，然后 1% 5% 10% 20% 50% 100%
#   4 臂 × 7 档 × 50 类 × 3 seed = 4200 格
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python3}"
GPU="${GPU:-6}"
WORKERS="${WORKERS:-6}"
STATUS="${STATUS:-runs/r3v_val_ratio_extra_20261026/status.log}"

mkdir -p "$(dirname "$STATUS")"
log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$STATUS"; }

DATASETS=(DDOS_19 IDS2017 UNSW_NB15 IoT-23 Kitsune CSECICIDS2018)
RATIOS=(0.02 0.01 0.05 0.1 0.2 0.5 1.0)
ARMS=(no_retrieval no_gated_residual no_modal_attn no_pre_train)

log "start validation ratio extra arms: arms=${ARMS[*]} ratios=${RATIOS[*]} workers=$WORKERS gpu=$GPU"
for r in "${RATIOS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    ds_log="runs/r3v_val_ratio_extra_20261026/ratio_${r}_${ds}.log"
    log "start ratio=$r dataset=$ds log=$ds_log"
    "$PY" scripts/run_ablation_matrix.py \
      --dataset "$ds" \
      --ratio "$r" \
      --seeds 777 888 999 \
      --arms "${ARMS[@]}" \
      --workers "$WORKERS" \
      --gpu "$GPU" \
      --finetune-scope layerwise \
      --finetune-epochs 200 \
      --finetune-batch-size 32 \
      --finetune-lr 0.0001 \
      --finetune-graph-lr 0.001 \
      --finetune-backbone-lr-scale 0.1 \
      --finetune-max-steps 1200 \
      --finetune-patience 8 \
      --finetune-eval-metric auc \
      --zero-init-gate --gate-init-bias -2.0 \
      --context-stage after_context \
      --graph-window 30 --neighbor-count 16 \
      --external-val \
      --optimize-threshold true \
      --output-root "save/r3v_val_ratio_20261026/ratio_${r}" \
      --log-root "log/r3v_val_ratio_extra_20261026/ratio_${r}" \
      --fig-root "figure/r3v_val_ratio_extra_20261026/ratio_${r}" \
      >> "$ds_log" 2>&1
    log "end ratio=$r dataset=$ds"
  done
done
log "validation ratio extra arms finished"
mkdir -p runs/r3v_val_ratio_extra_20261026/done
date -u '+%Y-%m-%d %H:%M:%S UTC' > runs/r3v_val_ratio_extra_20261026/done/all
