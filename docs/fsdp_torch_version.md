# FSDP2 API 与 torch 版本兼容说明

## 背景

代码原本使用 PyTorch **2.8** 引入的两个 FSDP2 方法：

- `FSDPModule.set_gradient_divide_factor(1.0)`
- `FSDPModule.set_force_sum_reduction_for_comms(True)`

在 torch **2.7.x**（当前环境为 `2.7.1+cu126`）上这两个方法不存在，
会抛出：

```
AttributeError: 'FSDPTransformerBlock' object has no attribute 'set_gradient_divide_factor'
```

报错点：

- `pretrain.py` `apply_fsdp()`
- `benchmark/backends/hrm_backend.py` HRM backend FSDP 初始化

## 当前临时方案（方案 B）

在以下文件加了运行时回退：

- `pretrain.py:125-133`
- `benchmark/backends/hrm_backend.py:106-120`

逻辑：

```python
if hasattr(module, "set_gradient_divide_factor"):
    module.set_gradient_divide_factor(1.0)
    module.set_force_sum_reduction_for_comms(True)
else:
    # torch<2.8 fallback
    module.set_reduce_scatter_divide_factor(1.0)
```

### 注意：方案 B 不是严格等价

| 方面 | torch 2.8 (`set_gradient_divide_factor` + `set_force_sum_reduction_for_comms`) | torch 2.7 (`set_reduce_scatter_divide_factor`) |
|---|---|---|
| 通信 op | 强制 SUM（`force_sum_reduction_for_comms=True`） | 仍是 reduce_scatter 默认 mean/avg 路径 |
| 数值 | 在通信里只 sum，divide 推到优化器侧；低精度更稳 | 在通信里直接除，低精度可能轻微损失 |
| 适用 | 推荐做法（upstream 写法） | 凑合能跑 |

对 Adam（scale-invariant）来说收敛影响一般可忽略，但 bf16/fp16
reduce 时数值会有微小差异。**不要把 2.7 上的训练曲线和 2.8 直接对比。**

## 长期方案（方案 A，待切换）

升级 torch 到 ≥ 2.8：

```bash
# 等 2.8 stable，或先用 nightly
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

切换后，**回滚临时 patch**：

1. `pretrain.py` 把 `apply_fsdp` 里的 if/else 删掉，恢复成原来的两行：
   ```python
   module.set_gradient_divide_factor(1.0)
   module.set_force_sum_reduction_for_comms(True)
   ```
2. `benchmark/backends/hrm_backend.py` 同样删掉 `hasattr` 分支，恢复
   `TransformerBlock` 循环和最外层 `model` 的两处调用为原始形式。

## 验证命令

```bash
python -c "from torch.distributed.fsdp import FSDPModule; print(hasattr(FSDPModule,'set_gradient_divide_factor'))"
```

- `True`  → 已是 2.8+，可以走方案 A，删除 fallback。
- `False` → 仍是 2.7.x，保留 fallback。
