"""SM100 tensor-core index scoring for compressed sparse attention."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm100 import (
    IndexerForwardSm100,
)
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor


_compile_cache: dict[tuple[object, ...], object] = {}


def exact_bf16_index_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    qhead_per_kv_head: int,
    out: torch.Tensor,
    sm_scale: float,
    current_stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Launch the SM100 tensor-core BSHD indexer scorer.

    Computes S[b,q,t] = sm_scale * sum_h [ReLU(Q_h . K_t^T) * W_{b,q,h}]
    with a ratio-causal mask. Supports head_dim (index_dim) of 64 or 128.
    """

    if not (q.is_contiguous() and k.is_contiguous() and weights.is_contiguous()):
        raise ValueError("Tensor-core index-score inputs must be contiguous.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("Tensor-core index scoring currently requires BF16 Q and K.")
    if weights.dtype != torch.bfloat16 or out.dtype != torch.float32:
        raise TypeError("Tensor-core index weights must be BF16 and scores FP32.")

    batch, sequence, query_heads, head_dim = q.shape
    key_batch, num_blocks, key_heads, key_dim = k.shape
    if key_batch != batch or key_dim != head_dim or query_heads != qhead_per_kv_head * key_heads:
        raise ValueError("Incompatible tensor-core index-score shapes.")
    if weights.shape != (batch, sequence, query_heads):
        raise ValueError("Incompatible tensor-core index-weight shape.")
    if out.shape != (batch, sequence, num_blocks):
        raise ValueError("Incompatible tensor-core score output shape.")

    m_block_size = qhead_per_kv_head * 2
    n_block_size = 128
    kv_stage = 4
    head_dim_padded = ((head_dim + 15) // 16) * 16
    k_block_size = 64 if head_dim_padded % 64 == 0 else head_dim_padded
    compile_key = (
        q.dtype,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        k_block_size,
        kv_stage,
    )
    stream = resolve_stream(current_stream)
    if compile_key not in _compile_cache:
        kernel = IndexerForwardSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            ratio=ratio,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            kv_stage=kv_stage,
            compute_lse=False,
            is_compressed_logits=False,
        )
        denom_placeholder = torch.empty(
            batch, sequence, device=q.device, dtype=torch.float32
        )
        _compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q),
            to_cute_tensor(k),
            to_cute_tensor(weights),
            to_cute_tensor(out),
            to_cute_tensor(denom_placeholder),
            Float32(sm_scale),
            Int32(sequence),
            Int32(num_blocks),
            None,
            None,
            None,
            stream,
            options=compile_options(),
        )

    denom_placeholder = torch.empty(
        batch, sequence, device=q.device, dtype=torch.float32
    )
    out.fill_(float("-inf"))
    _compile_cache[compile_key](
        q,
        k,
        weights,
        out,
        denom_placeholder,
        Float32(sm_scale),
        Int32(sequence),
        Int32(num_blocks),
        None,
        None,
        None,
        stream,
    )
    return out


__all__ = ["exact_bf16_index_scores"]
