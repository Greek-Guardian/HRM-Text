# HRM-Text 性能测试需求文档

## 项目概述
- **项目名**: HRM-Text (Hierarchical Reasoning Model)
- **目标**: 测试HRM的不同配置在8卡GPU上的训练和推理性能，要求代码各个部分尽可能解耦，后续可能还要与qwen模型比对性能

---

## 核心需求

### 1. 测试配置（5个不同的 loop 参数）
| Loop | H_cycles | L_cycles | 说明 |
|------|----------|----------|------|
| Loop1 | 1 | 1 | 最简 |
| Loop2 | 2 | 2 | 低 |
| Loop3 | 2 | 3 | 默认 |
| Loop4 | 3 | 4 | 高 |
| Loop5 | 4 | 5 | 最复 |

### 2. 训练参数（默认）
- seq_length: 2048
- batch_size: 8 per GPU
- num_batches: 10-100 (可调)
- num_GPUs: 8（通过torchrun自动DDP）
- optimizer: Adam
- use_mixed_precision: bfloat16 (可选)

### 3. 数据格式要求（HRM V1Dataset格式）

**关键**: 输入数据格式要按照本仓库的规范来，不能引起错误（下面只是例子，具体你需要看代码；另外，数据部分的代码在 ../data_io/）

**例子**:
- 8个sequences, 各2048 tokens
- total_tokens = 8 * 2048 = 16384
- cu_seqlens = [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]

当真实数据不可用时（/dev/shm/sampled不存在），使用虚拟数据，数据格式你自己确定

## 关键技术点

1. **DDP多卡**: 通过 `torchrun` 自动配置8卡并行
2. **Flash-Attention**: 必须使用（已安装）
3. **Carry状态**: HRM模型的特有机制，需每步更新
4. **Batch格式转换**: 虚拟数据必须生成HRM格式（1D flat），不能用2D (batch_size, seq_len)
5. **异常处理**: 如carry初始化失败或forward异常，应fallback到DummyModel，不要中断


## 约束

- 不使用虚拟模型（DummyModel），必须用真实HRM
- 不修改原始模型代码，只改benchmark代码
- 根目录保持干净，临时文件进入对应目录或删除
- 必须支持真实数据加载器（create_hrm_data_loader）和虚拟数据的fallback
