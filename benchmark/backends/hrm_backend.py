"""HRM Backend.

参考 pretrain.py:131-164 (create_model_and_carry) + pretrain.py:206-212 (train_batch).
不调用 @torch.compile (避免首步编译时间污染计时).
推理走 simple_inference_engine 的 _prefill / _batched_decode 同款路径,
但绕开 InferenceCheckpoint (无tokenizer/无ckpt, 用dummy权重).
"""
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn

from benchmark.backends.base import BenchmarkBackend, InferResult, TrainStepResult
from benchmark.data.dummy_hrm import make_hrm_dummy_batch
from benchmark.metrics import cuda_timer


def _build_hrm_config(
    *,
    H_cycles: int,
    L_cycles: int,
    seq_len: int,
    vocab_size: int,
    size_yaml: dict,
    target_only: bool = True,
) -> dict:
    """拼出 HierarchicalReasoningModel 期望的 config dict.

    复刻 pretrain.create_model_and_carry 中:
    `model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()`
    """
    arch = {
        "name": "baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel",
        "head": "lm_head@LMHead",
        "half_layers": True,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "H_override": {},
        "bp_warmup_ratio": 0.0,    # benchmark 不调度 bp
        "bp_min_steps": 2,
        "bp_max_steps": 5,
    }
    metadata = {
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "total_length": seq_len * 1024,  # 占位, benchmark 不用
        "tokenizer_info": {},
    }
    data = {"target_only": target_only, "path": "/dev/null"}
    return arch | size_yaml | metadata | data


class HRMBackend(BenchmarkBackend):
    name = "hrm"

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.optim = None
        self.carry = None
        self.cfg: Dict[str, Any] = {}
        self.fwd_dtype = torch.bfloat16
        self.local_bs = 8
        self.seq_len = 2048
        self.vocab_size = 32000
        self.use_fsdp = False
        self.num_params = 0

    # ------------------------- TRAIN -------------------------
    def setup_train(self, cfg: dict) -> None:
        self.cfg = cfg
        self.local_bs = int(cfg.get("local_batch_size", 8))
        self.seq_len = int(cfg.get("seq_len", 2048))
        self.vocab_size = int(cfg.get("vocab_size", 32000))
        self.fwd_dtype = getattr(torch, cfg.get("fwd_bwd_dtype", "bfloat16"))
        self.use_fsdp = bool(cfg.get("use_fsdp", False)) and dist.is_initialized() and dist.get_world_size() > 1

        from utils.functions import load_model_class

        model_cfg = _build_hrm_config(
            H_cycles=int(cfg["H_cycles"]),
            L_cycles=int(cfg["L_cycles"]),
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            size_yaml=cfg["size_yaml"],
        )

        model_cls = load_model_class(model_cfg["name"])
        head_cls = load_model_class(model_cfg["head"])

        with torch.device("cuda"):
            base = model_cls(model_cfg)
            self.carry = base.initial_carry(self.local_bs, dtype=self.fwd_dtype)
            model = head_cls(base, model_cfg)

        # Broadcast buffers if DDP active
        if dist.is_initialized() and dist.get_world_size() > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

        # Param count BEFORE FSDP wraps params into DTensors.
        self.num_params = sum(p.numel() for p in model.parameters())

        if self.use_fsdp:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
            from models.transformer import TransformerBlock

            mp = MixedPrecisionPolicy(param_dtype=self.fwd_dtype,
                                      reduce_dtype=torch.get_default_dtype())
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    fully_shard(module, mp_policy=mp, reshard_after_forward=False)
                    # torch<2.8 fallback. See docs/fsdp_torch_version.md.
                    if hasattr(module, "set_gradient_divide_factor"):
                        module.set_gradient_divide_factor(1.0)
                        module.set_force_sum_reduction_for_comms(True)
                    else:
                        module.set_reduce_scatter_divide_factor(1.0)
            fully_shard(model, mp_policy=mp, reshard_after_forward=False)
            if hasattr(model, "set_gradient_divide_factor"):
                model.set_gradient_divide_factor(1.0)
                model.set_force_sum_reduction_for_comms(True)
            else:
                model.set_reduce_scatter_divide_factor(1.0)
        else:
            # 单卡: 直接 cast 到 fwd dtype
            model = model.to(self.fwd_dtype)

        self.model = model
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("lr", 1e-4)),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

    def make_train_batch(self):
        batch, scalars = make_hrm_dummy_batch(
            num_seqs=self.local_bs,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            target_only=True,
            device="cuda",
            seed=0,
        )
        return batch, scalars

    def train_step(self, batch_pack) -> TrainStepResult:
        from models.common import wrap_tensor
        assert self.model is not None
        batch, scalars = batch_pack
        # 复刻 pretrain.py:364 — scalars 必须 wrap 成 CPU tensor 通过 batch 传入
        batch_full = batch | {
            k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in scalars.items()
        }
        self.optim.zero_grad(set_to_none=True)
        with cuda_timer() as t:
            out = self.model(batch=batch_full, carry=self.carry)
            new_carry, loss, _metrics = out
            loss.backward()
            self.optim.step()
            self.carry = new_carry
        loss_val = float(loss.detach().item())
        tokens = scalars["total_seqlen"]
        return TrainStepResult(step_time_s=t[0], tokens=tokens, loss=loss_val)

    # ------------------------- INFER -------------------------
    def setup_infer(self, cfg: dict) -> None:
        """复用 setup_train 路径, 但不要 optimizer + 切 eval."""
        self.setup_train(cfg)
        assert self.model is not None
        self.model.eval()

    @torch.inference_mode()
    def infer_run(self, prompt_len: int, max_new_tokens: int, batch_size: int) -> InferResult:
        assert self.model is not None
        device = "cuda"
        # 创建 KV cache (model.create_cache 由 LMHead 透传到 HRM)
        max_tokens = prompt_len + max_new_tokens + 1
        cache = self.model.create_cache(
            max_batch_size=batch_size,
            max_seq_len=max_tokens,
            dtype=self.fwd_dtype,
            device=device,
        )
        # Dummy prompt (B, S)
        prompt = torch.randint(0, self.vocab_size, (batch_size, prompt_len), dtype=torch.long, device=device)
        position_ids = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Prefill (TTFT)
        prefill_batch = {
            "inputs": prompt,
            "position_ids": position_ids,
            "cache": cache,
            "cache_lengths": torch.zeros(batch_size, dtype=torch.int32, device=device),
        }
        with cuda_timer() as t_pf:
            new_carry, logits = self.model(batch=prefill_batch, carry=self.carry)
            next_tok = logits[..., -1, :].argmax(-1)
        ttft = t_pf[0]

        # Decode loop
        cache_lengths = torch.full((batch_size,), prompt_len, dtype=torch.int32, device=device)
        cur_tok = next_tok
        with cuda_timer() as t_dec:
            for _ in range(max_new_tokens):
                step_batch = {
                    "inputs": cur_tok.unsqueeze(-1),
                    "position_ids": cache_lengths.unsqueeze(-1).long(),
                    "cache": cache,
                    "cache_lengths": cache_lengths,
                }
                _, logits = self.model(batch=step_batch, carry=new_carry)
                cur_tok = logits[..., -1, :].argmax(-1)
                cache_lengths = cache_lengths + 1
        decode_time = t_dec[0]
        decode_tps = (max_new_tokens * batch_size) / decode_time if decode_time > 0 else 0.0
        return InferResult(
            ttft_s=ttft,
            decode_tokens_per_s=decode_tps,
            decode_tokens=max_new_tokens * batch_size,
        )

    # ------------------------- CLEAN -------------------------
    def cleanup(self) -> None:
        self.model = None
        self.optim = None
        self.carry = None
        torch.cuda.empty_cache()
