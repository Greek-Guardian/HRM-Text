#!/usr/bin/env bash
# Run HRM XL benchmark (5-loop sweep, train+infer) on 8 GPUs with FSDP,
# log to .log/<params+timestamp>.log via tee. Param count is printed by
# run_benchmark.py (added column `params_M` in the CSV/MD reports).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# -------- params --------
CONFIG="benchmark/configs/hrm_xl.yaml"
MODE="all"          # train | infer | all
WARMUP=5
STEPS=30
NPROC=8
USE_FSDP=1          # 0 to disable
# ------------------------

mkdir -p .log
TS=$(date +%Y%m%d-%H%M%S)
CFG_TAG="$(basename "${CONFIG%.*}")"   # hrm_xl
TAG="${CFG_TAG}_${MODE}_w${WARMUP}_s${STEPS}_${TS}"
LOG=".log/benchmark_${TAG}.log"

EXTRA=()
[[ "${USE_FSDP}" == "1" ]] && EXTRA+=(--use-fsdp)

CMD=(
  torchrun --nproc_per_node="${NPROC}" benchmark/run_benchmark.py
    --backend hrm
    --config "${CONFIG}"
    --mode "${MODE}"
    --warmup "${WARMUP}"
    --steps "${STEPS}"
    --tag "${TAG}"
    "${EXTRA[@]}"
)

{
  echo "=================================================="
  echo " HRM-Text benchmark (hrm backend)"
  echo "--------------------------------------------------"
  echo " started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo " host    : $(hostname)"
  echo " pwd     : $PWD"
  echo " git     : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo " config  : ${CONFIG}"
  echo " mode    : ${MODE}"
  echo " warmup  : ${WARMUP}"
  echo " steps   : ${STEPS}"
  echo " nproc   : ${NPROC}"
  echo " fsdp    : ${USE_FSDP}"
  echo " tag     : ${TAG}"
  echo " log     : ${LOG}"
  echo " cmd     : ${CMD[*]}"
  echo "=================================================="
  T0=$(date +%s)
  "${CMD[@]}"
  RC=$?
  T1=$(date +%s)
  echo "=================================================="
  echo " ended   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo " rc      : ${RC}"
  echo " elapsed : $((T1 - T0))s"
  echo " reports : benchmark/reports/hrm_*_${TAG}.{csv,md}"
  echo "=================================================="
  exit "${RC}"
} 2>&1 | tee "${LOG}"
