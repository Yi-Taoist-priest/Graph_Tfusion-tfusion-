#!/usr/bin/env bash
# R3-V 全量消融：6 数据集 × 50 类 × 3 seed × 2 臂 = 300 格
#
# R3-V 配方（与 pilot 完全一致）：
#   - 外部验证集早停（data_split.val）
#   - max_steps=1200, batch=32, graph_lr=1e-3 / backbone_lr=1e-5
#   - 门控零初始化（bias=-2）
#   - after_context + 30s 窗口 + 16 邻居
#   - 2% 训练比例，fixed_0.5 阈值，rforest 下游
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python3}"
GPU="${GPU:-2}"
WORKERS="${WORKERS:-6}"
STATUS="${STATUS:-runs/r3v_full_20261026/status.log}"

mkdir -p "$(dirname "$STATUS")"
log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$STATUS"; }

DATASETS=(DDOS_19 IDS2017 UNSW_NB15 IoT-23 Kitsune CSECICIDS2018)

log "start R3-V full matrix: datasets=${DATASETS[*]} workers=$WORKERS gpu=$GPU"
for ds in "${DATASETS[@]}"; do
  ds_log="runs/r3v_full_20261026/${ds}.log"
  log "start dataset=$ds log=$ds_log"
  "$PY" scripts/run_ablation_matrix.py \
    --dataset "$ds" \
    --ratio 0.02 \
    --seeds 777 888 999 \
    --arms full no_history_context \
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
    --output-root save/ablations_r3v_20261026 \
    --log-root log/ablations_r3v_20261026 \
    --fig-root figure/ablations_r3v_20261026 \
    >> "$ds_log" 2>&1
  log "end dataset=$ds"
done
log "R3-V full matrix finished"
mkdir -p runs/r3v_full_20261026/done
date -u '+%Y-%m-%d %H:%M:%S UTC' > runs/r3v_full_20261026/done/all
