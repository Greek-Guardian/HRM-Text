#!/usr/bin/env bash
# Pretrain HRM-Text size=L on 8 GPUs, log to .log/<params+timestamp>.log via tee.
# Edit the params block below if you need other overrides.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# -------- params (mirror torchrun overrides) --------
ARCH=L
LR=2.5e-4
GBS=172032
NPROC=8
# ----------------------------------------------------

mkdir -p .log
TS=$(date +%Y%m%d-%H%M%S)
# Sanitize for filename: e.g. arch=L_lr=2.5e-4_gbs=172032
LOG=".log/pretrain_arch=${ARCH}_lr=${LR}_gbs=${GBS}_${TS}.log"

CMD=(
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  torchrun --nproc_per_node="${NPROC}" pretrain.py
    "arch/size@arch=${ARCH}"
    "lr=${LR}"
    "global_batch_size=${GBS}"
)

{
  echo "=================================================="
  echo " HRM-Text pretrain"
  echo "--------------------------------------------------"
  echo " started : $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo " host    : $(hostname)"
  echo " pwd     : $PWD"
  echo " git     : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo " arch    : ${ARCH}"
  echo " lr      : ${LR}"
  echo " gbs     : ${GBS}"
  echo " nproc   : ${NPROC}"
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
  echo "=================================================="
  exit "${RC}"
} 2>&1 | tee "${LOG}"
