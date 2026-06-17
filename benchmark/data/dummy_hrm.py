"""HRM V1Dataset 格式 dummy batch 生成.

参考: dataset_new.py:99-146 + flash_attention_prefixlm_v2.py:10-24
batch dict 全部 1D int32, 长度 = batch_max_length.
scalars: total_seqlen / numseqs / max_seqlen_prefix / max_seqlen_causal / max_seqlen_all (int)
"""
from typing import Tuple

import numpy as np
import torch


def make_hrm_dummy_batch(
    num_seqs: int,
    seq_len: int,
    vocab_size: int,
    target_only: bool = True,
    device: str = "cuda",
    seed: int = 0,
) -> Tuple[dict, dict]:
    """生成符合 V1Dataset _load_batch 输出格式的 dummy.

    Args:
        num_seqs: 序列数 (= batch_size per GPU)
        seq_len: 每条序列长度. prefix_len = causal_len = seq_len // 2
        vocab_size: 词表大小
        target_only: True=只监督 response 部分 (prefix labels 设为 IGNORE_LABEL_ID)
    Returns:
        (batch_dict, scalars_dict)
    """
    rng = np.random.default_rng(seed)
    IGNORE = -100  # 与 models.common.IGNORE_LABEL_ID 一致

    # 每条序列拆 prefix + causal 两半 (类似 V1Dataset 中 inst + resp[:-1])
    prefix_lens = np.full((num_seqs,), seq_len // 2, dtype=np.int32)
    causal_lens = np.full((num_seqs,), seq_len - seq_len // 2, dtype=np.int32)
    total_lens = prefix_lens + causal_lens  # = seq_len each

    batch_max_length = int(total_lens.sum())  # 已 packed, 无 pad

    # 拼 inputs / labels / position_ids (每条序列长度 seq_len, 顺序拼接)
    inputs_chunks = []
    labels_chunks = []
    pos_chunks = []
    for i in range(num_seqs):
        p = int(prefix_lens[i])
        c = int(causal_lens[i])
        # inputs: prefix tokens + causal[:-1]: 长度 p + (c) = seq_len? NO. V1Dataset 是 inst + resp[:-1]
        # 简化: 直接拼出长度 seq_len 的随机 token, 不再做 -1 shift (这里只测速度, 数值正确性不重要)
        inputs_chunks.append(rng.integers(0, vocab_size, size=seq_len, dtype=np.int32))
        # labels: prefix 部分 IGNORE, causal 部分随机 token
        if target_only:
            lbl = np.concatenate([
                np.full(p, IGNORE, dtype=np.int32),
                rng.integers(0, vocab_size, size=c, dtype=np.int32),
            ])
        else:
            lbl = rng.integers(0, vocab_size, size=seq_len, dtype=np.int32)
        labels_chunks.append(lbl)
        pos_chunks.append(np.arange(seq_len, dtype=np.int32))

    inputs = np.concatenate(inputs_chunks)
    labels = np.concatenate(labels_chunks)
    position_ids = np.concatenate(pos_chunks)

    # 复刻 compute_aux_seq_tensors_scalars (flash_attention_prefixlm_v2.py:10)
    cu_seqlens = np.pad(np.cumsum(total_lens, dtype=np.int32), (1, 0))  # [0, ...]
    # 全部 pad 到 batch_max_length 长度
    def pad_to(arr, val=0):
        if arr.shape[0] >= batch_max_length:
            return arr[:batch_max_length]
        return np.pad(arr, (0, batch_max_length - arr.shape[0]), constant_values=val)

    aux = {
        "prefix_lens": pad_to(prefix_lens),
        "causal_lens": pad_to(causal_lens),
        "cu_seqlens": pad_to(cu_seqlens),
    }

    batch_np = {
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        **aux,
    }

    batch = {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}

    scalars = {
        "total_seqlen": int(total_lens.sum()),
        "numseqs": int(num_seqs),
        "max_seqlen_prefix": int(prefix_lens.max()),
        "max_seqlen_causal": int(causal_lens.max()),
        "max_seqlen_all": int(total_lens.max()),
    }
    return batch, scalars
