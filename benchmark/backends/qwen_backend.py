"""Qwen Backend.

训练: HuggingFace transformers + FSDP (DDP-fallback) + AdamW + bf16
推理: vLLM (`LLM.generate`)

Note: HRM 走 simple_inference_engine 的 naive PyTorch generate, Qwen 走 vLLM (PagedAttn).
推理速度对比不公平, 但贴近真实部署上限. 报告里需注明.
"""
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn

from benchmark.backends.base import BenchmarkBackend, InferResult, TrainStepResult
from benchmark.data.dummy_qwen import make_qwen_dummy_batch
from benchmark.metrics import cuda_timer


class QwenBackend(BenchmarkBackend):
    name = "qwen"

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.optim = None
        self.cfg: Dict[str, Any] = {}
        self.fwd_dtype = torch.bfloat16
        self.local_bs = 8
        self.seq_len = 2048
        self.model_name_or_path = ""
        self.vocab_size = 32000
        self.use_fsdp = False
        self.num_params = 0
        self._vllm_engine = None  # set in setup_infer

    # ------------------------- TRAIN -------------------------
    def setup_train(self, cfg: dict) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        self.cfg = cfg
        self.local_bs = int(cfg.get("local_batch_size", 8))
        self.seq_len = int(cfg.get("seq_len", 2048))
        self.fwd_dtype = getattr(torch, cfg.get("fwd_bwd_dtype", "bfloat16"))
        self.model_name_or_path = cfg["model_name_or_path"]
        self.use_fsdp = bool(cfg.get("use_fsdp", False)) and dist.is_initialized() and dist.get_world_size() > 1

        # 加载模型 (从权重 or 从config随机初始化)
        load_from_pretrained = bool(cfg.get("load_from_pretrained", False))
        if load_from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.fwd_dtype,
                attn_implementation=cfg.get("attn_impl", "flash_attention_2"),
            ).cuda()
        else:
            hf_cfg = AutoConfig.from_pretrained(self.model_name_or_path)
            with torch.device("cuda"):
                model = AutoModelForCausalLM.from_config(
                    hf_cfg,
                    torch_dtype=self.fwd_dtype,
                    attn_implementation=cfg.get("attn_impl", "flash_attention_2"),
                ).to(self.fwd_dtype)
        self.vocab_size = model.config.vocab_size

        # Param count BEFORE FSDP wraps params into DTensors.
        self.num_params = sum(p.numel() for p in model.parameters())

        if self.use_fsdp:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

            mp = MixedPrecisionPolicy(param_dtype=self.fwd_dtype,
                                      reduce_dtype=torch.get_default_dtype())
            # 对每个 decoder layer 应用 FSDP
            decoder_layers = self._find_decoder_layers(model)
            for layer in decoder_layers:
                fully_shard(layer, mp_policy=mp, reshard_after_forward=False)
            fully_shard(model, mp_policy=mp, reshard_after_forward=False)

        self.model = model
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("lr", 1e-4)),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

    def _find_decoder_layers(self, model: nn.Module):
        # Qwen / Llama 标准: model.model.layers
        m = getattr(model, "model", model)
        layers = getattr(m, "layers", None)
        if layers is None:
            return []
        return list(layers)

    def make_train_batch(self):
        return make_qwen_dummy_batch(
            batch_size=self.local_bs,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            device="cuda",
            seed=0,
        )

    def train_step(self, batch) -> TrainStepResult:
        assert self.model is not None
        self.optim.zero_grad(set_to_none=True)
        with cuda_timer() as t:
            out = self.model(**batch)
            loss = out.loss
            loss.backward()
            self.optim.step()
        loss_val = float(loss.detach().item())
        tokens = batch["input_ids"].numel()
        return TrainStepResult(step_time_s=t[0], tokens=tokens, loss=loss_val)

    # ------------------------- INFER -------------------------
    def setup_infer(self, cfg: dict) -> None:
        """vLLM engine. cfg["model_name_or_path"] 必须是真实可加载权重."""
        from vllm import LLM

        self.cfg = cfg
        self.model_name_or_path = cfg["model_name_or_path"]
        vllm_kwargs = dict(cfg.get("vllm_kwargs", {}))
        self._vllm_engine = LLM(
            model=self.model_name_or_path,
            dtype=cfg.get("fwd_bwd_dtype", "bfloat16"),
            **vllm_kwargs,
        )

    def infer_run(self, prompt_len: int, max_new_tokens: int, batch_size: int) -> InferResult:
        from vllm import SamplingParams
        assert self._vllm_engine is not None

        # Dummy prompt token ids (vLLM 接受 prompt_token_ids 参数)
        prompts = []
        for _ in range(batch_size):
            ids = torch.randint(0, self.vocab_size, (prompt_len,), dtype=torch.long).tolist()
            prompts.append({"prompt_token_ids": ids})

        # TTFT: 单token输出 + 计时
        sp_ttft = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        with cuda_timer() as t_pf:
            self._vllm_engine.generate(prompts, sp_ttft, use_tqdm=False)
        ttft = t_pf[0] / batch_size  # 摊到单条 (vLLM 内部batched, 按总耗时除以条数估算单条TTFT)

        # Decode tok/s: 全长生成 + 时间
        sp_full = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True)
        with cuda_timer() as t_full:
            outs = self._vllm_engine.generate(prompts, sp_full, use_tqdm=False)
        total_decode_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        full_time = t_full[0]
        decode_only_time = max(1e-6, full_time - ttft * batch_size)  # 减去prefill估算的总时间
        decode_tps = total_decode_tokens / decode_only_time if decode_only_time > 0 else 0.0
        return InferResult(
            ttft_s=ttft,
            decode_tokens_per_s=decode_tps,
            decode_tokens=total_decode_tokens,
            extra={"full_time_s": full_time},
        )

    def cleanup(self) -> None:
        if self._vllm_engine is not None:
            del self._vllm_engine
            self._vllm_engine = None
        self.model = None
        self.optim = None
        torch.cuda.empty_cache()
