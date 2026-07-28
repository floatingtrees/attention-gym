"""SWA-only launcher for the SM100 qv attention primitive."""

from __future__ import annotations

from collections import OrderedDict
import math

import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from ...compressed_sparse_attention.cute.fa4_local import (
    FlashAttentionMLAForwardSm100,
)


_COMPILE_CACHE_MAXSIZE = 32
_compile_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return tensor.dtype, tuple(tensor.shape), tuple(tensor.stride())


def _compile_local(
    query: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    sink: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window: int,
    head_offset: int,
    rope_dims: int,
) -> object:
    key = (
        _tensor_signature(query),
        _tensor_signature(value),
        _tensor_signature(output),
        _tensor_signature(sink),
        _tensor_signature(cos),
        window,
        head_offset,
        rope_dims,
    )
    compiled = _compile_cache.get(key)
    if compiled is not None:
        _compile_cache.move_to_end(key)
        return compiled

    heads = query.shape[2]
    kernel = FlashAttentionMLAForwardSm100(
        is_causal=False,
        is_local=True,
        use_cpasync_load_KV=False,
        topk_length=value.shape[1],
        is_topk_gather=False,
        pack_gqa=True,
        qhead_per_kvhead=heads,
        nheads_kv=1,
        is_varlen_q=False,
        disable_bitmask=False,
        has_qk=False,
        fuse_csa_epilogue=True,
        csa_head_offset=head_offset,
        csa_rope_dims=rope_dims,
    )
    compiled = cute.compile(
        kernel,
        None,
        to_cute_tensor(query),
        None,
        to_cute_tensor(value),
        to_cute_tensor(output),
        None,
        1.0 / math.sqrt(query.shape[-1]),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        window - 1,
        0,
        to_cute_tensor(sink),
        to_cute_tensor(cos, assumed_align=4),
        to_cute_tensor(sin, assumed_align=4),
        None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    _compile_cache.move_to_end(key)
    if len(_compile_cache) > _COMPILE_CACHE_MAXSIZE:
        _compile_cache.popitem(last=False)
    return compiled


def local_attention(
    query: torch.Tensor,
    value: torch.Tensor,
    sink: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    window: int,
    head_offset: int,
    rope_dims: int,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run local attention, sink normalization, and inverse RoPE in one launch."""
    if query.ndim != 4 or value.ndim != 4 or output.ndim != 4:
        raise ValueError("query, value, and output must use BSHD layout.")
    batch, sequence, heads, dim = query.shape
    if heads not in (64, 128) or dim != 512 or value.shape != (batch, sequence, 1, 512):
        raise ValueError(
            "local CuTe attention requires Q=[B,S,H,512] with H in {64,128} "
            "and V=[B,S,1,512]."
        )
    if output.shape != query.shape:
        raise ValueError("output must have the same logical BSHD shape as query.")
    if not 1 <= window <= sequence:
        raise ValueError("window must be in [1, sequence].")
    if not query.is_contiguous() or not value.is_contiguous():
        raise ValueError("query and value must be contiguous BSHD tensors.")
    if any(tensor.device != query.device for tensor in (value, output, sink, cos, sin)):
        raise ValueError("all local attention inputs must be on query's device.")
    if value.dtype != query.dtype or output.dtype != query.dtype or sink.dtype != query.dtype:
        raise ValueError("query, value, output, and sink must have the same dtype.")
    expected_rope_shape = (sequence, rope_dims // 2)
    if cos.shape != expected_rope_shape or sin.shape != expected_rope_shape:
        raise ValueError(f"cos and sin must have shape {expected_rope_shape}.")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("cos and sin must have dtype float32.")
    if sink.ndim != 1 or sink.numel() < head_offset + heads:
        raise ValueError("sink must cover this head tile.")

    compiled = _compile_local(
        query,
        value,
        output,
        sink,
        cos,
        sin,
        window,
        head_offset,
        rope_dims,
    )
    compiled(
        None,
        query,
        None,
        value,
        output,
        None,
        1.0 / math.sqrt(dim),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        window - 1,
        0,
        sink,
        cos,
        sin,
        None,
    )
    return output


__all__ = ["local_attention"]
