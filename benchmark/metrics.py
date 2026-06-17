"""计时 / 显存 / 吞吐工具.

约定:
- cuda.synchronize() 包夹 + cuda.Event 计时
- 多次取中位数(去尾首步编译开销)
- DDP all_reduce 聚合 mean (跨 rank 同步)
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch


@contextmanager
def cuda_timer():
    """Yield 一个 list, 上下文退出后填入耗时(秒)."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        result: list = []
        torch.cuda.synchronize()
        start.record()
        try:
            yield result
        finally:
            end.record()
            torch.cuda.synchronize()
            result.append(start.elapsed_time(end) / 1000.0)
    else:
        import time
        result = []
        t0 = time.perf_counter()
        try:
            yield result
        finally:
            result.append(time.perf_counter() - t0)


def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def ddp_mean(value: float) -> float:
    """all_reduce mean. 单卡返回原值."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def ddp_max(value: float) -> float:
    """all_reduce max. 单卡返回原值."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


@dataclass
class StepStats:
    times: List[float]      # 每步耗时(秒)
    tokens_per_step: int    # 每步处理token数

    @property
    def median_time(self) -> float:
        return median(self.times)

    @property
    def mean_time(self) -> float:
        return mean(self.times)

    @property
    def tokens_per_s(self) -> float:
        t = self.median_time
        return self.tokens_per_step / t if t > 0 else 0.0
