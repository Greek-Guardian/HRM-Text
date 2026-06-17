"""Qwen / 标准 Causal LM 的 dummy batch (B, S) 格式."""
import torch


def make_qwen_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.long).to(device)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
