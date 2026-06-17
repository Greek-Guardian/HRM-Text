"""Backend 抽象基类. HRM/Qwen 各自实现."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainStepResult:
    step_time_s: float       # 单步耗时
    tokens: int              # 该步处理token数
    loss: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class InferResult:
    ttft_s: float            # prefill -> first token (秒)
    decode_tokens_per_s: float
    decode_tokens: int       # 实际生成token数
    extra: dict = field(default_factory=dict)


class BenchmarkBackend(ABC):
    """所有 backend 子类必须实现这些方法.

    生命周期: setup_train -> [train_step xN] -> cleanup
              setup_infer -> infer_run -> cleanup
    """

    name: str = "base"

    @abstractmethod
    def setup_train(self, cfg: dict) -> None:
        """建模型 + optimizer + FSDP/DDP wrap."""

    @abstractmethod
    def train_step(self, batch: Any) -> TrainStepResult:
        """单步 fwd + bwd + optim. 返回耗时/token数."""

    @abstractmethod
    def make_train_batch(self) -> Any:
        """生成一个 dummy batch (符合该 backend 模型期望格式)."""

    @abstractmethod
    def setup_infer(self, cfg: dict) -> None:
        """加载推理引擎 (HRM自带 / vLLM等)."""

    @abstractmethod
    def infer_run(self, prompt_len: int, max_new_tokens: int, batch_size: int) -> InferResult:
        """跑一次推理, 拆分prefill+decode计时."""

    @abstractmethod
    def cleanup(self) -> None:
        """释放VRAM, 清理状态."""

    def peak_mem_gb(self) -> float:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 3)

    def reset_peak_mem(self) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
