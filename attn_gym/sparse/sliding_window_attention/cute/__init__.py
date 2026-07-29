"""SM100 CuTe DSL backend for shared-KV sliding-window attention.

The data path contains only the operations required by SWA: plain RoPE, shared-KV
RMS normalization, causal local attention, the learned sink, and inverse output RoPE.
The tensor-core attention primitive is shared with CSA's local-only path; no compression
or indexing work is launched here.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from cutlass import BFloat16, Int32

from ...compressed_sparse_attention.cute.backward import (
    compile_cast_gradient,
    compile_local_norm_backward,
    compile_pack_dsa_kv_sink,
    compile_prepare_dsa_backward,
    compile_unpack_dsa_gradients,
)
from ...compressed_sparse_attention.cute.kernels import (
    compile_local_norm,
    compile_query_rope,
    cute_dtype,
)
from .backward import compile_pack_dsa_local_indices
from .local_attention import local_attention

_HEAD_DIM = 512
_TESTED_CUDA_VERSION = "13.3"
_ROPE_CACHE_MAXSIZE = 8
_DSA_PACKED_WORKSPACE_BYTES = 1536 * 1024 * 1024
_rope_table_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
] = OrderedDict()


def _dsa_workspace_bytes(
    tokens: int,
    dim: int,
    heads: int,
    total_kv: int,
    index_width: int,
) -> int:
    """Estimate the live packed tensors and DSA workspaces for one SWA tile."""
    rounded_tokens = math.ceil(tokens / 8) * 8
    rounded_heads = math.ceil(heads / 64) * 64
    rounded_dim = math.ceil(dim / 8) * 8
    head_dependent = (
        tokens * heads * (4 * dim * 2 + 4)
        + rounded_tokens * rounded_heads * 2 * 4
        + heads * (4 + 4 + 2 + 4)
    )
    rounded_kv = math.ceil(total_kv / 8) * 8
    kv_dependent = total_kv * dim * (2 + 2 + 4) + rounded_kv * rounded_dim * 4
    sparse_metadata = tokens * (index_width * 4 + 4)
    return head_dependent + kv_dependent + sparse_metadata


def _dsa_tile_shape(
    tokens: int,
    dim: int,
    heads: int,
    total_kv: int,
    index_width: int,
) -> tuple[int, int]:
    """Choose head/token tiles that minimize launches under the workspace budget."""
    fixed_bytes = _dsa_workspace_bytes(0, dim, 0, total_kv, index_width)
    if fixed_bytes >= _DSA_PACKED_WORKSPACE_BYTES:
        raise RuntimeError(
            "The CuTe SWA backward fixed KV workspace exceeds the "
            f"{_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )

    candidates = {min(heads, candidate) for candidate in (128, 64, 32, 16, 8, 4, 2, 1)}
    best: tuple[int, int, int] | None = None
    for head_tile in candidates:
        if head_tile <= 0:
            continue
        low, high = 0, tokens
        while low < high:
            middle = (low + high + 1) // 2
            if (
                _dsa_workspace_bytes(
                    middle,
                    dim,
                    head_tile,
                    total_kv,
                    index_width,
                )
                <= _DSA_PACKED_WORKSPACE_BYTES
            ):
                low = middle
            else:
                high = middle - 1
        token_tile = low
        if token_tile < tokens:
            alignment = 128 if token_tile >= 128 else 4
            token_tile = token_tile // alignment * alignment
        if token_tile < 1:
            continue
        launches = math.ceil(heads / head_tile) * math.ceil(tokens / token_tile)
        candidate = (launches, -head_tile, -token_tile)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError(
            "The CuTe SWA backward workspace cannot fit one query/head tile within "
            f"the {_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )
    return -best[1], -best[2]


def _require_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "The CuTe sliding window attention backend targets SM100 exclusively; "
            f"device {device} has compute capability {capability[0]}.{capability[1]}."
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
    if heads <= 0:
        raise ValueError("The SM100 CuTe specialization requires H to be positive.")
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
    *,
    _return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run the full shared-KV SWA forward path on the current CUDA stream."""
    batch, heads, sequence = _validate_configuration(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        share_kv,
    )
    if sliding_window_size == 0:
        if _return_state:
            raise ValueError("Zero-window SWA has no backward attention state.")
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
    padded_sink = attention_sink
    if heads % 64:
        padded_heads = math.ceil(heads / 64) * 64
        padded_sink = torch.zeros(
            padded_heads,
            device=attention_sink.device,
            dtype=attention_sink.dtype,
        )
        padded_sink[:heads].copy_(attention_sink)
    combined_lse = (
        torch.empty(
            batch,
            sequence,
            heads,
            device=Q.device,
            dtype=torch.float32,
        )
        if _return_state
        else None
    )
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

        direct_output = active_heads == tile_heads
        selected_output = (
            output[:, head_offset : head_offset + active_heads].permute(0, 2, 1, 3)
            if direct_output
            else torch.empty_like(query)
        )
        selected_lse = (
            torch.empty(
                batch,
                sequence,
                tile_heads,
                device=Q.device,
                dtype=torch.float32,
            )
            if _return_state
            else None
        )
        local_attention(
            query,
            local_kv,
            padded_sink,
            cos,
            sin,
            window,
            head_offset,
            rope_dims,
            output=selected_output,
            lse=selected_lse,
        )
        if not direct_output:
            output[:, head_offset : head_offset + active_heads].copy_(
                selected_output[:, :, :active_heads].permute(0, 2, 1, 3)
            )
        if _return_state:
            assert combined_lse is not None and selected_lse is not None
            combined_lse[:, :, head_offset : head_offset + active_heads].copy_(
                selected_lse[:, :, :active_heads]
            )

    if _return_state:
        assert combined_lse is not None
        return output, (local_kv, cos, sin, combined_lse)
    return output


def _sliding_window_attention_backward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    dout: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the shared-KV SWA path with local-only CuTe DSL kernels."""
    batch, heads, sequence = _validate_configuration(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        share_kv,
    )
    window = min(sliding_window_size, sequence)
    if window == 0:
        return (
            torch.zeros_like(Q),
            torch.zeros_like(KV),
            torch.zeros_like(KV_norm_weight),
            torch.zeros_like(attention_sink),
        )

    output, state = _sliding_window_attention_forward(
        Q.detach(),
        KV.detach(),
        KV_norm_weight.detach(),
        attention_sink.detach(),
        sliding_window_size,
        rope_dims,
        share_kv,
        _return_state=True,
    )
    local_kv, cos, sin, combined_lse = state
    dtype = cute_dtype(Q)
    dsa_dtype = BFloat16
    tokens = batch * sequence
    total_kv = tokens
    index_width = math.ceil(window / 64) * 64

    compressed_sentinel = torch.empty(
        batch,
        1,
        1,
        _HEAD_DIM,
        device=Q.device,
        dtype=Q.dtype,
    )
    kv_packed = torch.empty(
        total_kv,
        _HEAD_DIM,
        device=Q.device,
        dtype=torch.bfloat16,
    )
    sink_fp32 = torch.empty(
        heads,
        device=Q.device,
        dtype=torch.float32,
    )
    compile_pack_dsa_kv_sink(
        dtype,
        dsa_dtype,
        batch,
        sequence,
        _HEAD_DIM,
        0,
        sequence,
        heads,
    )(
        compressed_sentinel,
        local_kv,
        attention_sink,
        kv_packed,
        sink_fp32,
    )

    head_chunk, token_chunk = _dsa_tile_shape(
        tokens,
        _HEAD_DIM,
        heads,
        total_kv,
        index_width,
    )
    indices = torch.empty(
        token_chunk,
        index_width,
        device=Q.device,
        dtype=torch.int32,
    )
    topk_lengths = torch.empty(
        token_chunk,
        device=Q.device,
        dtype=torch.int32,
    )
    pack_indices = compile_pack_dsa_local_indices(
        sequence,
        window,
        index_width,
        token_chunk,
    )

    dQ = torch.empty_like(Q)
    dlocal = torch.zeros_like(local_kv, dtype=torch.float32)
    dcompressed_sentinel = torch.empty_like(
        compressed_sentinel,
        dtype=torch.float32,
    )
    d_sink_accumulator = torch.zeros(
        heads,
        device=Q.device,
        dtype=torch.float32,
    )
    dout = dout.contiguous()
    from ...compressed_sparse_attention.cute.dsa_backward_sm100 import (
        sparse_attention_backward_wrapper,
    )

    for token_offset in range(0, tokens, token_chunk):
        packed_tokens = min(token_chunk, tokens - token_offset)
        selected_indices = indices[:packed_tokens]
        selected_lengths = topk_lengths[:packed_tokens]
        pack_indices(
            indices,
            topk_lengths,
            Int32(token_offset),
            Int32(packed_tokens),
        )
        for head_offset in range(0, heads, head_chunk):
            packed_heads = min(head_chunk, heads - head_offset)
            selected_sink = sink_fp32[head_offset : head_offset + packed_heads]
            q_packed = torch.empty(
                packed_tokens,
                packed_heads,
                _HEAD_DIM,
                device=Q.device,
                dtype=torch.bfloat16,
            )
            output_packed = torch.empty_like(q_packed)
            dout_packed = torch.empty_like(q_packed)
            lse_packed = torch.empty(
                packed_tokens,
                packed_heads,
                device=Q.device,
                dtype=torch.float32,
            )
            compile_prepare_dsa_backward(
                dtype,
                dsa_dtype,
                batch,
                heads,
                packed_heads,
                sequence,
                packed_tokens,
                _HEAD_DIM,
                rope_dims,
            )(
                Q,
                output,
                dout,
                combined_lse,
                cos,
                sin,
                q_packed,
                output_packed,
                dout_packed,
                lse_packed,
                Int32(head_offset),
                Int32(token_offset),
            )
            result = sparse_attention_backward_wrapper(
                q_packed,
                kv_packed,
                output_packed,
                dout_packed,
                lse_packed,
                selected_sink,
                selected_indices,
                softmax_scale=1.0 / math.sqrt(_HEAD_DIM),
                topk_length=selected_lengths,
            )
            compile_unpack_dsa_gradients(
                dtype,
                dsa_dtype,
                batch,
                heads,
                packed_heads,
                sequence,
                packed_tokens,
                _HEAD_DIM,
                rope_dims,
                0,
                sequence,
            )(
                result["dq"],
                result["dkv"],
                cos,
                sin,
                dQ,
                dlocal,
                dcompressed_sentinel,
                Int32(head_offset),
                Int32(token_offset),
            )
            d_sink_accumulator[head_offset : head_offset + packed_heads].add_(result["d_sink"])

    dKV = torch.empty_like(KV)
    dKV_norm_weight_fp32 = torch.zeros_like(KV_norm_weight, dtype=torch.float32)
    compile_local_norm_backward(dtype, batch, sequence, _HEAD_DIM, rope_dims)(
        KV,
        KV_norm_weight,
        cos,
        sin,
        dlocal,
        dKV,
        dKV_norm_weight_fp32,
    )
    dKV_norm_weight = torch.empty_like(KV_norm_weight)
    compile_cast_gradient(dtype, KV_norm_weight.numel())(
        dKV_norm_weight_fp32,
        dKV_norm_weight,
    )

    d_attention_sink = torch.empty_like(attention_sink)
    compile_cast_gradient(dtype, heads)(
        d_sink_accumulator,
        d_attention_sink,
    )
    return dQ, dKV, dKV_norm_weight, d_attention_sink


@torch.library.custom_op(
    "attention_gym::_cute_sliding_window_attention_backward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_sliding_window_attention_backward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    dout: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.cuda.device(Q.device):
        return _sliding_window_attention_backward(
            Q,
            KV,
            KV_norm_weight,
            attention_sink,
            dout,
            sliding_window_size,
            rope_dims,
            share_kv,
        )


@_cute_sliding_window_attention_backward.register_fake
def _cute_sliding_window_attention_backward_fake(
    Q,
    KV,
    KV_norm_weight,
    attention_sink,
    dout,
    sliding_window_size,
    rope_dims,
    share_kv,
):
    del dout, sliding_window_size, rope_dims, share_kv
    return tuple(torch.empty_like(tensor) for tensor in (Q, KV, KV_norm_weight, attention_sink))


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


def _cute_sliding_window_attention_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:4])
    ctx.sliding_window_size, ctx.rope_dims, ctx.share_kv = inputs[4:]


def _cute_sliding_window_attention_autograd_backward(ctx, dout):
    grads = _cute_sliding_window_attention_backward(
        *ctx.saved_tensors,
        dout,
        ctx.sliding_window_size,
        ctx.rope_dims,
        ctx.share_kv,
    )
    return (*grads, None, None, None)


_cute_sliding_window_attention_forward.register_autograd(
    _cute_sliding_window_attention_autograd_backward,
    setup_context=_cute_sliding_window_attention_setup_context,
)


def sliding_window_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    """Dispatch the opaque CuTe forward/backward operators."""
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
