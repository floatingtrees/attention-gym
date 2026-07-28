"""SM100 CuTe DSL backend for shared-KV sliding-window attention.

The data path contains only the operations required by SWA: plain RoPE, shared-KV
RMS normalization, causal local attention, the learned sink, and inverse output RoPE.
The tensor-core attention primitive is shared with CSA's local-only path; no compression
or indexing work is launched here.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import torch

from ...compressed_sparse_attention.cute.kernels import (
    compile_local_norm,
    compile_query_rope,
    cute_dtype,
)
from .local_attention import local_attention


_HEAD_DIM = 512
_TESTED_CUDA_VERSION = "13.3"
_ROPE_CACHE_MAXSIZE = 8
_rope_table_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
] = OrderedDict()


def _require_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "The CuTe sliding window attention backend targets SM100 exclusively; "
            f"device {device} has compute capability {capability[0]}.{capability[1]}."
        )
    if torch.version.cuda != _TESTED_CUDA_VERSION:
        raise RuntimeError(
            "The CuTe sliding window attention backend is validated with CUDA "
            f"{_TESTED_CUDA_VERSION}; this PyTorch build uses CUDA {torch.version.cuda}."
        )


def _rope_tables(
    device_index: int,
    sequence_length: int,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference's cached FP32 base-10,000 cosine and sine tables."""
    key = (device_index, sequence_length, rope_dims)
    cached = _rope_table_cache.get(key)
    device = torch.device("cuda", device_index)
    if cached is None:
        with torch.cuda.device(device):
            pair_positions = torch.arange(
                0,
                rope_dims,
                2,
                device=device,
                dtype=torch.float32,
            )
            frequencies = 1.0 / (10_000.0 ** (pair_positions / rope_dims))
            positions = torch.arange(
                sequence_length,
                device=device,
                dtype=torch.float32,
            )
            angles = torch.outer(positions, frequencies)
            cos, sin = angles.cos(), angles.sin()
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        cached = (cos, sin, ready)
        _rope_table_cache[key] = cached
        if len(_rope_table_cache) > _ROPE_CACHE_MAXSIZE:
            _rope_table_cache.popitem(last=False)
    else:
        _rope_table_cache.move_to_end(key)

    cos, sin, ready = cached
    consumer_stream = torch.cuda.current_stream(device)
    consumer_stream.wait_event(ready)
    cos.record_stream(consumer_stream)
    sin.record_stream(consumer_stream)
    return cos, sin


def _validate_configuration(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    share_kv: bool,
) -> tuple[int, int, int]:
    if not Q.is_cuda:
        raise ValueError("The CuTe backend requires CUDA tensors.")
    _require_sm100(Q.device)
    if Q.dtype != torch.bfloat16:
        raise TypeError("The CuTe backend supports bfloat16 inputs only.")
    batch, heads, sequence, head_dim = Q.shape
    if sequence >= 2**31:
        raise ValueError("The CuTe backend requires S < 2**31.")
    if head_dim != _HEAD_DIM:
        raise ValueError("The SM100 CuTe specialization requires D=512.")
    if heads % 64:
        raise ValueError("The SM100 CuTe specialization requires H to be a multiple of 64.")
    if not share_kv:
        raise ValueError("The SM100 CuTe specialization requires share_kv=True.")
    if KV.shape[1] != 1:
        raise ValueError("KV must physically have one shared head for the CuTe backend.")
    for name, tensor in (
        ("Q", Q),
        ("KV", KV),
        ("KV_norm_weight", KV_norm_weight),
        ("attention_sink", attention_sink),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the CuTe backend.")
    return batch, heads, sequence


def _sliding_window_attention_forward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    """Run the full shared-KV SWA forward path on the current CUDA stream."""
    batch, heads, sequence = _validate_configuration(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        share_kv,
    )
    if sliding_window_size == 0:
        return torch.zeros_like(Q)

    window = min(sliding_window_size, sequence)
    dtype = cute_dtype(Q)
    device_index = Q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cos, sin = _rope_tables(device_index, sequence, rope_dims)

    local_kv = torch.empty(
        batch,
        sequence,
        1,
        _HEAD_DIM,
        device=Q.device,
        dtype=Q.dtype,
    )
    compile_local_norm(dtype, batch, sequence, _HEAD_DIM, rope_dims, 0)(
        KV,
        KV_norm_weight,
        cos,
        sin,
        local_kv,
    )

    output = torch.empty_like(Q)
    for head_offset in range(0, heads, 128):
        active_heads = min(128, heads - head_offset)
        tile_heads = 64 if active_heads <= 64 else 128
        query = torch.empty(
            batch,
            sequence,
            tile_heads,
            _HEAD_DIM,
            device=Q.device,
            dtype=Q.dtype,
        )
        compile_query_rope(
            dtype,
            batch,
            heads,
            tile_heads,
            active_heads,
            head_offset,
            sequence,
            _HEAD_DIM,
            rope_dims,
        )(Q, cos, sin, query, query)

        selected_output = output[
            :, head_offset : head_offset + active_heads
        ].permute(0, 2, 1, 3)
        local_attention(
            query,
            local_kv,
            attention_sink,
            cos,
            sin,
            window,
            head_offset,
            rope_dims,
            output=selected_output,
        )

    return output


@torch.library.custom_op(
    "attention_gym::_cute_sliding_window_attention_forward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_sliding_window_attention_forward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    with torch.cuda.device(Q.device):
        return _sliding_window_attention_forward(
            Q,
            KV,
            KV_norm_weight,
            attention_sink,
            sliding_window_size,
            rope_dims,
            share_kv,
        )


@_cute_sliding_window_attention_forward.register_fake
def _cute_sliding_window_attention_forward_fake(
    Q,
    KV,
    KV_norm_weight,
    attention_sink,
    sliding_window_size,
    rope_dims,
    share_kv,
):
    del (
        KV,
        KV_norm_weight,
        attention_sink,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    return torch.empty_like(Q)


def sliding_window_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    """Dispatch the opaque CuTe forward operator."""
    return _cute_sliding_window_attention_forward(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["sliding_window_attention"]
