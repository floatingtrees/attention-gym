"""Deterministic key-owned shared-KV gradient for causal sliding-window attention.

This is an isolated SM100 prototype.  Each Triton program owns one contiguous
tile of KV rows and accumulates every reverse-window query and every query head
before storing once.  Consequently the kernel needs neither global atomics nor
head-sized partial-gradient workspaces.

The public launcher intentionally accepts the tensors already retained by the
CuTe SWA forward/backward path:

* ``query``, ``output``, and ``grad_output`` use BHSD layout.
* ``local_kv`` uses BSHD layout with one physically shared KV head.
* ``lse`` is the forward's KV-only log-sum-exp; the learned sink is folded into
  the normalizer in this kernel.
* ``query`` and ``grad_output`` are rotated while loading.  ``output`` is used
  only for ``sum(output * grad_output)``, which is invariant under that rotation.

No forward probability or S-by-S tensor is materialized.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_BLOCK_HEADS = 16
_BLOCK_KEYS = 16
_HEAD_DIM = 512


@triton.jit
def _load_rotated_bhsd(
    tensor_ptr,
    cos_ptr,
    sin_ptr,
    batch,
    position,
    heads,
    dimensions,
    row_mask,
    dimension_mask,
    stride_tb: tl.constexpr,
    stride_th: tl.constexpr,
    stride_ts: tl.constexpr,
    stride_td: tl.constexpr,
    stride_rs: tl.constexpr,
    stride_rd: tl.constexpr,
    D: tl.constexpr,
    ROPE_DIMS: tl.constexpr,
):
    """Load one position by a head tile and apply forward RoPE."""
    matrix_mask = row_mask[:, None] & dimension_mask[None, :]
    pointer = (
        tensor_ptr
        + batch * stride_tb
        + heads[:, None] * stride_th
        + position * stride_ts
        + dimensions[None, :] * stride_td
    )
    values = tl.load(pointer, mask=matrix_mask, other=0.0)

    rope_begin = D - ROPE_DIMS
    in_rope = dimensions >= rope_begin
    relative_dimension = dimensions - rope_begin
    is_even = relative_dimension % 2 == 0
    paired_dimensions = tl.where(is_even, dimensions + 1, dimensions - 1)
    paired_pointer = (
        tensor_ptr
        + batch * stride_tb
        + heads[:, None] * stride_th
        + position * stride_ts
        + paired_dimensions[None, :] * stride_td
    )
    paired = tl.load(
        paired_pointer,
        mask=matrix_mask & in_rope[None, :],
        other=0.0,
    )

    rope_pair = tl.maximum(relative_dimension // 2, 0)
    rope_mask = dimension_mask & in_rope
    cosine = tl.load(
        cos_ptr + position * stride_rs + rope_pair * stride_rd,
        mask=rope_mask,
        other=1.0,
    )
    sine = tl.load(
        sin_ptr + position * stride_rs + rope_pair * stride_rd,
        mask=rope_mask,
        other=0.0,
    )
    rotated = tl.where(
        is_even[None, :],
        values * cosine[None, :] - paired * sine[None, :],
        paired * sine[None, :] + values * cosine[None, :],
    )
    return tl.where(in_rope[None, :], rotated, values)


@triton.jit
def _local_key_owned_dkv_kernel(
    query_ptr,
    local_kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    sink_ptr,
    cos_ptr,
    sin_ptr,
    grad_local_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    stride_gob: tl.constexpr,
    stride_goh: tl.constexpr,
    stride_gos: tl.constexpr,
    stride_god: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_les: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_sink: tl.constexpr,
    stride_rs: tl.constexpr,
    stride_rd: tl.constexpr,
    stride_glb: tl.constexpr,
    stride_gls: tl.constexpr,
    stride_glh: tl.constexpr,
    stride_gld: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    ROPE_DIMS: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_HEAD_TILES: tl.constexpr,
    NUM_QUERY_POSITIONS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_tile = tl.program_id(0)
    batch = tl.program_id(1)

    keys = key_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dimensions = tl.arange(0, BLOCK_D)
    key_mask = keys < S
    dimension_mask = dimensions < D
    values = tl.load(
        local_kv_ptr
        + batch * stride_lb
        + keys[:, None] * stride_ls
        + 0 * stride_lh
        + dimensions[None, :] * stride_ld,
        mask=key_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    gradient = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    # A tile's earliest key first appears at key_tile * BLOCK_N.  Its latest
    # key can remain in a reverse window for WINDOW - 1 additional positions.
    first_query = key_tile * BLOCK_N
    head_offsets = tl.arange(0, BLOCK_H)
    for query_offset in tl.range(0, NUM_QUERY_POSITIONS):
        position = first_query + query_offset
        query_valid = position < S
        for head_tile in tl.range(0, NUM_HEAD_TILES):
            heads = head_tile * BLOCK_H + head_offsets
            row_mask = (heads < H) & query_valid

            query = _load_rotated_bhsd(
                query_ptr,
                cos_ptr,
                sin_ptr,
                batch,
                position,
                heads,
                dimensions,
                row_mask,
                dimension_mask,
                stride_qb,
                stride_qh,
                stride_qs,
                stride_qd,
                stride_rs,
                stride_rd,
                D,
                ROPE_DIMS,
            )
            grad_output_raw = tl.load(
                grad_output_ptr
                + batch * stride_gob
                + heads[:, None] * stride_goh
                + position * stride_gos
                + dimensions[None, :] * stride_god,
                mask=row_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            grad_output = _load_rotated_bhsd(
                grad_output_ptr,
                cos_ptr,
                sin_ptr,
                batch,
                position,
                heads,
                dimensions,
                row_mask,
                dimension_mask,
                stride_gob,
                stride_goh,
                stride_gos,
                stride_god,
                stride_rs,
                stride_rd,
                D,
                ROPE_DIMS,
            )
            output = tl.load(
                output_ptr
                + batch * stride_ob
                + heads[:, None] * stride_oh
                + position * stride_os
                + dimensions[None, :] * stride_od,
                mask=row_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            delta = tl.sum(
                grad_output_raw.to(tl.float32) * output.to(tl.float32),
                axis=1,
            )

            kv_lse = tl.load(
                lse_ptr
                + batch * stride_leb
                + position * stride_les
                + heads * stride_leh,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            sink = tl.load(
                sink_ptr + heads * stride_sink,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            normalizer_max = tl.maximum(kv_lse, sink)
            total_lse = normalizer_max + tl.log(
                tl.exp(kv_lse - normalizer_max) + tl.exp(sink - normalizer_max)
            )

            valid = (
                row_mask[:, None]
                & key_mask[None, :]
                & (keys[None, :] <= position)
                & (keys[None, :] >= position - WINDOW + 1)
            )
            scores = tl.dot(
                query,
                tl.trans(values),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            probabilities = tl.exp(scores * SCALE - total_lse[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            grad_probabilities = tl.dot(
                grad_output,
                tl.trans(values),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            grad_scores = probabilities * (grad_probabilities - delta[:, None])

            gradient += tl.dot(
                tl.trans(probabilities.to(tl.bfloat16)),
                grad_output,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            gradient += tl.dot(
                tl.trans((grad_scores * SCALE).to(tl.bfloat16)),
                query,
                input_precision="ieee",
                out_dtype=tl.float32,
            )

    tl.store(
        grad_local_ptr
        + batch * stride_glb
        + keys[:, None] * stride_gls
        + 0 * stride_glh
        + dimensions[None, :] * stride_gld,
        gradient,
        mask=key_mask[:, None] & dimension_mask[None, :],
    )


def _validate_local_inputs(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    attention_sink: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window: int,
    rope_dims: int,
) -> tuple[int, int, int, int]:
    if query.ndim != 4:
        raise ValueError("query must use BHSD layout.")
    batch, heads, sequence, dim = query.shape
    if dim != _HEAD_DIM or heads % _BLOCK_HEADS:
        raise ValueError(
            f"key-owned dKV requires D={_HEAD_DIM} and a multiple of {_BLOCK_HEADS} heads."
        )
    if local_kv.shape != (batch, sequence, 1, dim):
        raise ValueError("local_kv must have shape [B,S,1,D].")
    if output.shape != query.shape or grad_output.shape != query.shape:
        raise ValueError("output and grad_output must match query's BHSD shape.")
    if lse.shape != (batch, sequence, heads):
        raise ValueError("lse must contain the KV-only FP32 LSE in [B,S,H] layout.")
    if attention_sink.ndim != 1 or attention_sink.numel() != heads:
        raise ValueError("attention_sink must contain one value per query head.")
    if not 1 <= window <= sequence:
        raise ValueError("window must be in [1, sequence].")
    if rope_dims <= 0 or rope_dims > dim or rope_dims % 2:
        raise ValueError("rope_dims must be positive, even, and no larger than D.")
    if cos.shape != (sequence, rope_dims // 2) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have shape [S, rope_dims / 2].")

    tensors = (query, local_kv, output, grad_output, lse, attention_sink, cos, sin)
    if not query.is_cuda or any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all key-owned dKV inputs must be CUDA tensors on one device.")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (query, local_kv, output, grad_output, attention_sink)
    ):
        raise TypeError("query, KV, output, grad_output, and sink must be bfloat16.")
    if any(tensor.dtype != torch.float32 for tensor in (lse, cos, sin)):
        raise TypeError("lse, cos, and sin must be float32.")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("the innermost dimension of every input must be contiguous.")
    return batch, heads, sequence, dim


def local_key_owned_dkv(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    attention_sink: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window: int,
    rope_dims: int,
    *,
    grad_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the deterministic FP32 gradient of a physically shared local KV.

    ``lse`` must exclude the learned sink, matching the CuTe forward state.
    ``query``, ``output``, and ``grad_output`` are the unrotated/final BHSD
    tensors; forward RoPE is fused into the query and grad-output loads.
    """
    batch, heads, sequence, dim = _validate_local_inputs(
        query,
        local_kv,
        output,
        grad_output,
        lse,
        attention_sink,
        cos,
        sin,
        window,
        rope_dims,
    )
    if grad_local is None:
        grad_local = torch.empty(
            batch,
            sequence,
            1,
            dim,
            device=query.device,
            dtype=torch.float32,
        )
    elif (
        grad_local.shape != local_kv.shape
        or grad_local.device != query.device
        or grad_local.dtype != torch.float32
        or not grad_local.is_contiguous()
    ):
        raise ValueError("grad_local must be contiguous FP32 [B,S,1,D] on query's device.")

    num_query_positions = window + _BLOCK_KEYS - 1
    _local_key_owned_dkv_kernel[(triton.cdiv(sequence, _BLOCK_KEYS), batch)](
        query,
        local_kv,
        output,
        grad_output,
        lse,
        attention_sink,
        cos,
        sin,
        grad_local,
        *query.stride(),
        *local_kv.stride(),
        *output.stride(),
        *grad_output.stride(),
        *lse.stride(),
        attention_sink.stride(0),
        *cos.stride(),
        *grad_local.stride(),
        H=heads,
        S=sequence,
        D=dim,
        WINDOW=window,
        ROPE_DIMS=rope_dims,
        SCALE=1.0 / math.sqrt(dim),
        NUM_HEAD_TILES=triton.cdiv(heads, _BLOCK_HEADS),
        NUM_QUERY_POSITIONS=num_query_positions,
        BLOCK_H=_BLOCK_HEADS,
        BLOCK_N=_BLOCK_KEYS,
        BLOCK_D=_HEAD_DIM,
        num_warps=8,
        num_stages=1,
    )
    return grad_local


__all__ = ["local_key_owned_dkv"]
