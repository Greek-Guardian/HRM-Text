#!/usr/bin/env bash
# H800 smoke test: 验证 FA3 + HRM 在目标硬件能跑通.
# Usage:
#   bash benchmark/smoke_h800.sh
#
# 期望输出:
#   - GPU sm_90 (Hopper)
#   - flash_attn_3 ops 完整 (含 fwd)
#   - 5 个 loop 配置训练 step 都跑通, CSV 输出非空

set -e
cd "$(dirname "$0")/.."

echo "==================== 环境检查 ===================="
python -c "
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    sm = f'sm_{p.major}{p.minor}'
    flag = 'OK' if p.major >= 9 else 'WARN: FA3 needs sm_90+'
    print(f'  GPU{i}: {p.name} {sm} ({flag})')
print('flash_attn_3.fwd available:', hasattr(torch.ops.flash_attn_3, 'fwd'))
"

echo
echo "==================== 单卡 Smoke (HRM 训练, 2 步) ===================="
python benchmark/run_benchmark.py \
    --backend hrm \
    --config benchmark/configs/hrm_xl.yaml \
    --mode train \
    --warmup 1 --steps 2 \
    --tag h800_smoke

echo
echo "==================== 报告 ===================="
ls -la benchmark/reports/ | tail -10
echo
echo "smoke 完成. 看 benchmark/reports/hrm_train_h800_smoke.md"
