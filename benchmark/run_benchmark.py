"""HRM-Text Benchmark 主入口.

Usage:
    # 单卡 smoke test
    python benchmark/run_benchmark.py --backend hrm --config benchmark/configs/hrm_xl.yaml \
        --mode train --steps 5 --warmup 2

    # 8 卡 DDP/FSDP
    torchrun --nproc_per_node=8 benchmark/run_benchmark.py \
        --backend hrm --config benchmark/configs/hrm_xl.yaml \
        --mode all --warmup 5 --steps 30 --use-fsdp

    # Qwen
    torchrun --nproc_per_node=8 benchmark/run_benchmark.py \
        --backend qwen --config benchmark/configs/qwen.yaml \
        --mode train --warmup 5 --steps 30 --use-fsdp
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import yaml

# 加 repo 根目录到 path (不依赖安装)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.backends.base import BenchmarkBackend, InferResult, TrainStepResult  # noqa: E402
from benchmark.metrics import StepStats, ddp_max, ddp_mean, median  # noqa: E402

LOOP_CONFIGS = [
    ("Loop1", 1, 1),
    ("Loop2", 2, 2),
    ("Loop3", 2, 3),
    ("Loop4", 3, 4),
    ("Loop5", 4, 5),
]


def init_distributed() -> tuple[int, int, int, bool]:
    """复刻 pretrain.py:322-330 的 DDP 初始化模式."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, True
    # 单卡 fallback: 也起一个 1-process group, 因 LMHead 内部用了 dist.all_reduce
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    return 0, 1, 0, False


def make_backend(name: str) -> BenchmarkBackend:
    if name == "hrm":
        from benchmark.backends.hrm_backend import HRMBackend
        return HRMBackend()
    if name == "qwen":
        from benchmark.backends.qwen_backend import QwenBackend
        return QwenBackend()
    raise ValueError(f"unknown backend: {name}")


def run_train_loop(backend: BenchmarkBackend, cfg: dict, *, warmup: int, steps: int, rank: int) -> dict:
    backend.setup_train(cfg)
    if rank == 0:
        n = getattr(backend, "num_params", 0)
        print(f"[model] backend={backend.name} params={n:,} ({n/1e6:.2f} M)")
    batch = backend.make_train_batch()

    # Warmup
    for i in range(warmup):
        backend.train_step(batch)

    # 重置显存峰值统计
    backend.reset_peak_mem()

    # 计时
    times: List[float] = []
    losses: List[float] = []
    tokens_per_step = 0
    for i in range(steps):
        r = backend.train_step(batch)
        times.append(r.step_time_s)
        if r.loss is not None and r.loss == r.loss:  # not NaN
            losses.append(r.loss)
        tokens_per_step = r.tokens

    # 跨 rank 聚合
    local_med = median(times)
    global_med = ddp_mean(local_med)
    global_peak_mem = ddp_max(backend.peak_mem_gb())

    # tokens/s 用 global tokens (= local_tokens * world_size) / median_step_time
    world = dist.get_world_size() if dist.is_initialized() else 1
    global_tps = (tokens_per_step * world) / global_med if global_med > 0 else 0.0

    return {
        "median_step_s": global_med,
        "tokens_per_step_per_gpu": tokens_per_step,
        "global_tokens_per_s": global_tps,
        "peak_mem_gb": global_peak_mem,
        "mean_loss": sum(losses) / len(losses) if losses else float("nan"),
        "num_params": getattr(backend, "num_params", 0),
        "raw_step_times": times if rank == 0 else [],
    }


def run_infer(backend: BenchmarkBackend, cfg: dict, *, prompt_len: int, max_new: int, batch_size: int, rank: int = 0) -> InferResult:
    backend.setup_infer(cfg)
    if rank == 0:
        n = getattr(backend, "num_params", 0)
        if n > 0:
            print(f"[model] backend={backend.name} params={n:,} ({n/1e6:.2f} M)")
    # 预热 1 次
    backend.infer_run(prompt_len=prompt_len, max_new_tokens=min(8, max_new), batch_size=batch_size)
    backend.reset_peak_mem()
    return backend.infer_run(prompt_len=prompt_len, max_new_tokens=max_new, batch_size=batch_size)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, rows: List[dict], note: str = "") -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        if note:
            f.write(f"{note}\n\n")
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join("---" for _ in keys) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[k]) for k in keys) + " |\n")


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hrm", "qwen"])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["train", "infer", "all"], default="train")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--infer-batch-size", type=int, default=8)
    ap.add_argument("--use-fsdp", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmark" / "reports")
    ap.add_argument("--tag", default=None, help="report 文件名 tag, 默认时间戳")
    args = ap.parse_args()

    rank, world, local_rank, ddp = init_distributed()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["use_fsdp"] = args.use_fsdp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")

    train_rows: List[dict] = []
    infer_rows: List[dict] = []

    if args.backend == "hrm":
        # 5-loop sweep
        for loop_name, h, l in LOOP_CONFIGS:
            cfg = dict(base_cfg)
            cfg["H_cycles"] = h
            cfg["L_cycles"] = l

            if args.mode in ("train", "all"):
                backend = make_backend("hrm")
                try:
                    res = run_train_loop(backend, cfg, warmup=args.warmup, steps=args.steps, rank=rank)
                    if rank == 0:
                        train_rows.append({
                            "backend": "hrm",
                            "model": "XL",
                            "loop": loop_name,
                            "H": h, "L": l,
                            "params_M": res["num_params"] / 1e6,
                            "median_step_ms": res["median_step_s"] * 1000,
                            "global_tokens_per_s": res["global_tokens_per_s"],
                            "peak_mem_gb": res["peak_mem_gb"],
                            "mean_loss": res["mean_loss"],
                        })
                        print(f"[train] {loop_name} H={h} L={l} step={res['median_step_s']*1000:.1f}ms "
                              f"tok/s={res['global_tokens_per_s']:.0f} mem={res['peak_mem_gb']:.2f}GB "
                              f"params={res['num_params']/1e6:.2f}M")
                except Exception as e:
                    if rank == 0:
                        print(f"[train] {loop_name} FAILED: {type(e).__name__}: {e}")
                        train_rows.append({
                            "backend": "hrm", "model": "XL", "loop": loop_name,
                            "H": h, "L": l,
                            "params_M": float("nan"),
                            "median_step_ms": float("nan"),
                            "global_tokens_per_s": float("nan"),
                            "peak_mem_gb": float("nan"),
                            "mean_loss": float("nan"),
                        })
                finally:
                    backend.cleanup()

            if args.mode in ("infer", "all"):
                backend = make_backend("hrm")
                try:
                    ir = run_infer(backend, cfg,
                                   prompt_len=args.prompt_len,
                                   max_new=args.max_new,
                                   batch_size=args.infer_batch_size,
                                   rank=rank)
                    if rank == 0:
                        infer_rows.append({
                            "backend": "hrm",
                            "model": "XL",
                            "loop": loop_name,
                            "H": h, "L": l,
                            "params_M": getattr(backend, "num_params", 0) / 1e6,
                            "ttft_ms": ir.ttft_s * 1000,
                            "decode_tok_s": ir.decode_tokens_per_s,
                            "decoded": ir.decode_tokens,
                            "infer_engine": "naive_pytorch",
                        })
                        print(f"[infer] {loop_name} ttft={ir.ttft_s*1000:.1f}ms decode={ir.decode_tokens_per_s:.1f}tok/s "
                              f"params={getattr(backend, 'num_params', 0)/1e6:.2f}M")
                except Exception as e:
                    if rank == 0:
                        print(f"[infer] {loop_name} FAILED: {type(e).__name__}: {e}")
                finally:
                    backend.cleanup()

    elif args.backend == "qwen":
        cfg = dict(base_cfg)
        if args.mode in ("train", "all"):
            backend = make_backend("qwen")
            try:
                res = run_train_loop(backend, cfg, warmup=args.warmup, steps=args.steps, rank=rank)
                if rank == 0:
                    train_rows.append({
                        "backend": "qwen",
                        "model": cfg.get("model_name_or_path", "?"),
                        "loop": "-", "H": "-", "L": "-",
                        "params_M": res["num_params"] / 1e6,
                        "median_step_ms": res["median_step_s"] * 1000,
                        "global_tokens_per_s": res["global_tokens_per_s"],
                        "peak_mem_gb": res["peak_mem_gb"],
                        "mean_loss": res["mean_loss"],
                    })
                    print(f"[train] qwen step={res['median_step_s']*1000:.1f}ms "
                          f"tok/s={res['global_tokens_per_s']:.0f} mem={res['peak_mem_gb']:.2f}GB "
                          f"params={res['num_params']/1e6:.2f}M")
            finally:
                backend.cleanup()

        if args.mode in ("infer", "all") and rank == 0:
            # vLLM 不需 DDP, 单 rank 跑即可
            backend = make_backend("qwen")
            try:
                ir = run_infer(backend, cfg,
                               prompt_len=args.prompt_len,
                               max_new=args.max_new,
                               batch_size=args.infer_batch_size,
                               rank=rank)
                infer_rows.append({
                    "backend": "qwen",
                    "model": cfg.get("model_name_or_path", "?"),
                    "loop": "-", "H": "-", "L": "-",
                    "params_M": getattr(backend, "num_params", 0) / 1e6,
                    "ttft_ms": ir.ttft_s * 1000,
                    "decode_tok_s": ir.decode_tokens_per_s,
                    "decoded": ir.decode_tokens,
                    "infer_engine": "vllm",
                })
                print(f"[infer] qwen+vllm ttft={ir.ttft_s*1000:.1f}ms decode={ir.decode_tokens_per_s:.1f}tok/s "
                      f"params={getattr(backend, 'num_params', 0)/1e6:.2f}M")
            finally:
                backend.cleanup()

    # rank 0 写报告
    if rank == 0:
        if train_rows:
            write_csv(args.out_dir / f"{args.backend}_train_{tag}.csv", train_rows)
            write_md(args.out_dir / f"{args.backend}_train_{tag}.md", train_rows,
                     note=f"# {args.backend.upper()} Training Benchmark ({tag})\n\n"
                          f"world_size={world}, warmup={args.warmup}, steps={args.steps}")
        if infer_rows:
            write_csv(args.out_dir / f"{args.backend}_infer_{tag}.csv", infer_rows)
            note = (f"# {args.backend.upper()} Inference Benchmark ({tag})\n\n"
                    f"prompt_len={args.prompt_len}, max_new={args.max_new}, "
                    f"batch_size={args.infer_batch_size}\n\n"
                    f"**注**: HRM 走 naive PyTorch generate; Qwen 走 vLLM (PagedAttn). "
                    f"推理速度对比仅供\"实际部署上限\"参考, 非架构层面公平对比.")
            write_md(args.out_dir / f"{args.backend}_infer_{tag}.md", infer_rows, note=note)
        print(f"\nReports written to: {args.out_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
