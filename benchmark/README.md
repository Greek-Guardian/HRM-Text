# HRM-Text Benchmark

性能测试: HRM (5个 H/L cycle 配置) vs Qwen, 用于决定项目用哪个模型.

## 输出指标

- **训练**: `median_step_ms`, `global_tokens_per_s`, `peak_mem_gb`
- **推理**: `ttft_ms` (time-to-first-token), `decode_tok_s`

## 5个 Loop 配置

| Loop | H_cycles | L_cycles |
|------|----------|----------|
| Loop1 | 1 | 1 |
| Loop2 | 2 | 2 |
| Loop3 | 2 | 3 (默认) |
| Loop4 | 3 | 4 |
| Loop5 | 4 | 5 |

## 用法

### 单卡 smoke test (验证跑通)

```bash
cd /home/liangzida/HRM/HRM-Text
python benchmark/run_benchmark.py \
    --backend hrm \
    --config benchmark/configs/hrm_xl.yaml \
    --mode train \
    --warmup 1 --steps 3
```

### 8 卡 HRM 训练 + 推理 benchmark

```bash
torchrun --nproc_per_node=8 benchmark/run_benchmark.py \
    --backend hrm \
    --config benchmark/configs/hrm_xl.yaml \
    --mode all \
    --warmup 5 --steps 30 \
    --use-fsdp \
    --tag hrm_xl_8gpu
```

### Qwen 训练 benchmark (8 卡)

先编辑 `benchmark/configs/qwen.yaml` 选 model size:
```yaml
model_name_or_path: Qwen/Qwen2.5-7B   # 或 0.5B / 1.5B / 3B
```

```bash
torchrun --nproc_per_node=8 benchmark/run_benchmark.py \
    --backend qwen \
    --config benchmark/configs/qwen.yaml \
    --mode train \
    --warmup 5 --steps 30 \
    --use-fsdp \
    --tag qwen7b_8gpu
```

### Qwen 推理 (vLLM, 单卡或 TP)

```bash
python benchmark/run_benchmark.py \
    --backend qwen \
    --config benchmark/configs/qwen.yaml \
    --mode infer \
    --prompt-len 512 --max-new 128 \
    --infer-batch-size 8 \
    --tag qwen7b_vllm
```

## 报告输出

`benchmark/reports/<backend>_<mode>_<tag>.{csv,md}`

## 重要说明

1. **公平性**:
   - 训练: HRM/Qwen 都走 PyTorch + FSDP + bf16, 同 batch 同 seq_len, 公平.
   - 推理: HRM=naive PyTorch generate, Qwen=vLLM (PagedAttn). 不公平, 报告里标了.

2. **HRM 实际参数量**: XL 配置 `half_layers=True` -> n_layers 32->16 (H 8 + L 8).

3. **Carry 状态**: 当前用的 `hrm_nocarry_bp_warmup` 变体, carry=None.

4. **真实数据**: 当前只用 dummy. 如需切真实数据, 用 `pretrain.create_dataloader`,
   把 batch+scalars 传给 `backend.train_step` 即可 (接口已兼容).

5. **不修改原模型代码** (约束): 所有 benchmark 代码独立在 `benchmark/` 目录.

## 文件结构

```
benchmark/
├── run_benchmark.py        # 主入口
├── metrics.py              # 计时/显存工具
├── backends/
│   ├── base.py             # BenchmarkBackend 接口
│   ├── hrm_backend.py      # HRM 实现
│   └── qwen_backend.py     # Qwen 实现
├── data/
│   ├── dummy_hrm.py        # HRM 1D-flat dummy batch
│   └── dummy_qwen.py       # Qwen 2D dummy batch
├── configs/
│   ├── hrm_xl.yaml
│   └── qwen.yaml
└── reports/                # 生成的 CSV+MD
```
