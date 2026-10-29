#!/usr/bin/env bash
# 按 config/train-test 下全部逐类配置串行跑一遍（6 个数据集）。
#
# 串行单进程：每个配置只有一个 data_tag，并发会让所有进程挤在 gpu_enable[0] 上；
# 产物落在各配置自己的 log/fig/checkpoint 路径，批次级日志汇总到
# log/run_all_6datasets/batch.log（含每个配置的 OK/FAIL 与总耗时）。
#
# 用法：
#   bash scripts/run_all_6datasets.sh
#   PY=/path/to/python bash scripts/run_all_6datasets.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/liuluping/miniconda3/envs/shenzhiju/bin/python}"
RUN_LOG="log/run_all_6datasets"
mkdir -p "$RUN_LOG"

mapfile -t CONFIGS < <(ls config/train-test/*/*.json | sort)
TOTAL=${#CONFIGS[@]}
STARTED=$(date +%s)

printf '[%s] 批次开始：%d 个配置\n' "$(date '+%F %T')" "$TOTAL" | tee -a "$RUN_LOG/batch.log"
FAILED=0
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  printf '[%d/%d] %s >>> %s\n' "$((i+1))" "$TOTAL" "$(date '+%T')" "$cfg" | tee -a "$RUN_LOG/batch.log"
  if "$PY" scripts/run_graph_tfusion.py -c "$cfg" >>"$RUN_LOG/batch.log" 2>&1; then
    printf '[%d/%d] OK   %s\n' "$((i+1))" "$TOTAL" "$cfg" | tee -a "$RUN_LOG/batch.log"
  else
    FAILED=$((FAILED+1))
    printf '[%d/%d] FAIL %s\n' "$((i+1))" "$TOTAL" "$cfg" | tee -a "$RUN_LOG/batch.log"
  fi
done

printf '[%s] 批次结束：成功 %d / 失败 %d / 共 %d，耗时 %d 分钟\n' \
  "$(date '+%F %T')" "$((TOTAL-FAILED))" "$FAILED" "$TOTAL" \
  "$(( ($(date +%s)-STARTED)/60 ))" | tee -a "$RUN_LOG/batch.log"
