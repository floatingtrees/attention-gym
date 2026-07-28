"""SM100 CuTe DSL backend for shared-KV heavily compressed attention.

The path is specialized for the DeepSeek-V4 HCA shape (H=64, D=512).  It reuses
the vendored FA4 QV-only tensor-core loops from CSA, but has no indexer or top-k:
the compressed gather is the deterministic prefix of completed blocks.
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
from ...compressed_sparse_attention.cute.local_attention import compressed_attention
from .backward import (
    compile_compression_backward,
    compile_pack_dsa_indices,
)
from .kernels import compile_causal_gather, compile_compression

_HEADS = 64
_HEAD_DIM = 512
_TESTED_CUDA_VERSION = "13.3"
_DIFFERENTIABLE_INPUTS = 8
_DSA_PACKED_WORKSPACE_BYTES = 1536 * 1024 * 1024
_ROPE_CACHE_MAXSIZE = 8
_rope_table_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
] = OrderedDict()
_query_streams: dict[int, torch.cuda.Stream] = {}
_compressed_streams: dict[int, torch.cuda.Stream] = {}


def _dsa_workspace_bytes(
    tokens: int,
    dim: int,
    heads: int,
    total_kv: int,
    index_width: int = 0,
) -> int:
    """Estimate packed HCA tensors and vendored DSA workspaces for one tile."""
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
    """Choose the fewest HCA head/token tiles within the packed-memory budget."""
    fixed_bytes = _dsa_workspace_bytes(0, dim, 0, total_kv, index_width)
    if fixed_bytes >= _DSA_PACKED_WORKSPACE_BYTES:
        raise RuntimeError(
            "The CuTe HCA backward fixed KV workspace exceeds the "
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
            "The CuTe HCA backward workspace cannot fit one query/head tile "
            f"within the {_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )
    return -best[1], -best[2]


def _require_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "The CuTe heavily compressed attention backend targets SM100 exclusively; "
            f"device {device} has compute capability {capability[0]}.{capability[1]}."
        )
    if torch.version.cuda != _TESTED_CUDA_VERSION:
        raise RuntimeError(
            "The CuTe heavily compressed attention backend is validated with CUDA "
            f"{_TESTED_CUDA_VERSION}; this PyTorch build uses CUDA {torch.version.cuda}."
        )


def _validate_configuration(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    share_kv: bool,
) -> tuple[int, int]:
    if not Q.is_cuda:
        raise ValueError("The CuTe backend requires CUDA tensors.")
    _require_sm100(Q.device)
    if Q.dtype != torch.bfloat16:
        raise TypeError("The CuTe backend supports bfloat16 inputs only.")
    batch, heads, sequence, dim = Q.shape
    if heads != _HEADS or dim != _HEAD_DIM:
        raise ValueError(f"The SM100 CuTe specialization requires H={_HEADS} and D={_HEAD_DIM}.")
    if sequence >= 2**31:
        raise ValueError("The CuTe backend requires S < 2**31.")
    if sequence < compression_rate:
        raise ValueError("The CuTe backend requires S >= compression_rate.")
    if sliding_window_size <= 0:
        raise ValueError("The CuTe backend requires a positive sliding_window_size.")
    if not share_kv:
        raise ValueError("The SM100 CuTe specialization requires share_kv=True.")
    for name, tensor in (("KV", KV), ("C", C), ("Z", Z)):
        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must physically have one KV head for the SM100 CuTe backend."
            )
    for name, tensor in (
        ("Q", Q),
        ("KV", KV),
        ("C", C),
        ("Z", Z),
        ("B", B),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the CuTe backend.")
    return batch, sequence


def _stream(
    streams: dict[int, torch.cuda.Stream],
    device_index: int,
) -> torch.cuda.Stream:
    result = streams.get(device_index)
    if result is None:
        result = torch.cuda.Stream(device=device_index)
        streams[device_index] = result
    return result


def _rope_tables(
    device_index: int,
    sequence: int,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached FP32 YaRN tables with explicit cross-stream synchronization."""
    key = (device_index, sequence, rope_dims)
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
            frequencies = 1.0 / (160_000.0 ** (pair_positions / rope_dims))
            correction_scale = 2 * math.log(160_000.0)
            low = math.floor(
                rope_dims * math.log(65_536 / (32.0 * 2 * math.pi)) / correction_scale
            )
            high = math.ceil(rope_dims * math.log(65_536 / (1.0 * 2 * math.pi)) / correction_scale)
            low = max(low, 0)
            high = min(high, rope_dims - 1)
            if low == high:
                high += 0.001
            ramp = (
                torch.arange(
                    rope_dims // 2,
                    device=device,
                    dtype=torch.float32,
                )
                - low
            ) / (high - low)
            smooth = 1 - ramp.clamp(0, 1)
            frequencies = frequencies / 16.0 * (1 - smooth) + frequencies * smooth
            positions = torch.arange(sequence, device=device, dtype=torch.float32)
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
    consumer = torch.cuda.current_stream(device)
    consumer.wait_event(ready)
    cos.record_stream(consumer)
    sin.record_stream(consumer)
    return cos, sin


def _heavily_compressed_attention_forward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    _return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run the full shared-KV HCA prefill path."""
    batch, sequence = _validate_configuration(
        Q,
        KV,
        C,
        Z,
        B,
        KV_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        sliding_window_size,
        share_kv,
    )
    window = min(sliding_window_size, sequence)
    attention_sequence = sequence + sequence % 2
    if attention_sequence != sequence:
        attention_query = torch.nn.functional.pad(Q, (0, 0, 0, 1))
        attention_kv = torch.nn.functional.pad(KV, (0, 0, 0, 1))
    else:
        attention_query = Q
        attention_kv = KV
    num_blocks = sequence // compression_rate
    gather_length = math.ceil((num_blocks + min(window + 1, sequence)) / 128) * 128
    gather_rows = math.ceil(sequence * _HEADS / 128)
    dtype = cute_dtype(Q)
    device_index = Q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cos, sin = _rope_tables(device_index, attention_sequence, rope_dims)

    query = torch.empty(
        batch,
        attention_sequence,
        _HEADS,
        _HEAD_DIM,
        device=Q.device,
        dtype=Q.dtype,
    )
    local_kv = torch.empty(
        batch,
        attention_sequence,
        1,
        _HEAD_DIM,
        device=Q.device,
        dtype=Q.dtype,
    )
    compressed_kv = torch.empty(
        batch,
        num_blocks,
        1,
        _HEAD_DIM,
        device=Q.device,
        dtype=Q.dtype,
    )
    gather = torch.empty(
        batch,
        gather_rows,
        gather_length,
        device=Q.device,
        dtype=torch.int32,
    )
    output = torch.empty_like(attention_query)
    output_bshd = output.permute(0, 2, 1, 3)
    combined_lse = None
    if _return_state:
        combined_lse = torch.empty(
            batch,
            attention_sequence,
            _HEADS,
            device=Q.device,
            dtype=torch.float32,
        )

    current_stream = torch.cuda.current_stream(Q.device)
    query_stream = _stream(_query_streams, device_index)
    compressed_stream = _stream(_compressed_streams, device_index)
    query_stream.wait_stream(current_stream)
    compressed_stream.wait_stream(current_stream)
    cos.record_stream(query_stream)
    sin.record_stream(query_stream)
    cos.record_stream(compressed_stream)
    sin.record_stream(compressed_stream)

    with torch.cuda.stream(query_stream):
        compile_query_rope(
            dtype,
            batch,
            _HEADS,
            _HEADS,
            _HEADS,
            0,
            attention_sequence,
            _HEAD_DIM,
            rope_dims,
            0,
        )(attention_query, cos, sin, query, query)
        query.record_stream(query_stream)

    with torch.cuda.stream(compressed_stream):
        compile_compression(
            dtype,
            batch,
            sequence,
            _HEAD_DIM,
            compression_rate,
            rope_dims,
        )(
            C,
            Z,
            B,
            compressed_kv_norm_weight,
            cos[:sequence],
            sin[:sequence],
            compressed_kv,
        )
        compile_causal_gather(
            batch,
            sequence,
            _HEADS,
            num_blocks,
            compression_rate,
            window,
            gather_length,
        )(gather)
        compressed_kv.record_stream(compressed_stream)
        gather.record_stream(compressed_stream)

    compile_local_norm(
        dtype,
        batch,
        attention_sequence,
        _HEAD_DIM,
        rope_dims,
        0,
    )(attention_kv, KV_norm_weight, cos, sin, local_kv)
    current_stream.wait_stream(query_stream)
    current_stream.wait_stream(compressed_stream)
    compressed_attention(
        query,
        compressed_kv,
        gather,
        local_value=local_kv,
        output=output_bshd,
        lse=combined_lse,
        sink=attention_sink,
        cos=cos,
        sin=sin,
        rope_dims=rope_dims,
        csa_topk=num_blocks,
        csa_window=window,
        csa_rate=compression_rate,
        store_lse=_return_state,
    )
    if attention_sequence != sequence:
        result = output[:, :, :sequence].contiguous()
        local_state = local_kv[:, :sequence].contiguous()
        lse_state = combined_lse[:, :sequence].contiguous() if combined_lse is not None else None
    else:
        result = output
        local_state = local_kv
        lse_state = combined_lse
    if _return_state:
        assert lse_state is not None
        return result, (
            local_state,
            compressed_kv,
            cos[:sequence],
            sin[:sequence],
            lse_state,
        )
    return result


def _heavily_compressed_attention_backward(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    dout: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Differentiate the fixed completed-block HCA data path."""
    dout = dout.contiguous()
    recompute_tensors = tuple(
        tensor.detach()
        for tensor in (
            Q,
            KV,
            C,
            Z,
            B,
            KV_norm_weight,
            compressed_kv_norm_weight,
            attention_sink,
        )
    )
    output, state = _heavily_compressed_attention_forward(
        *recompute_tensors,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
        _return_state=True,
    )
    local_kv, compressed_kv, cos, sin, combined_lse = state
    batch, heads, sequence, dim = Q.shape
    blocks = sequence // compression_rate
    window = min(sliding_window_size, sequence)
    dtype = cute_dtype(Q)
    dsa_dtype = BFloat16
    dQ = torch.empty_like(Q)
    dlocal = torch.zeros_like(local_kv, dtype=torch.float32)
    dcompressed = torch.zeros_like(compressed_kv, dtype=torch.float32)

    from ...compressed_sparse_attention.cute.dsa_backward_sm100 import (
        sparse_attention_backward_wrapper,
    )

    tokens = batch * sequence
    index_width = math.ceil((blocks + window) / 64) * 64
    total_kv = batch * (blocks + sequence)
    kv_packed = torch.empty(
        total_kv,
        dim,
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
        dim,
        blocks,
        sequence,
        heads,
    )(
        compressed_kv,
        local_kv,
        attention_sink,
        kv_packed,
        sink_fp32,
    )

    head_chunk, token_chunk = _dsa_tile_shape(
        tokens,
        dim,
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
    d_sink_accumulator = torch.zeros(
        heads,
        device=Q.device,
        dtype=torch.float32,
    )
    pack_indices = compile_pack_dsa_indices(
        batch,
        sequence,
        blocks,
        compression_rate,
        window,
        index_width,
        token_chunk,
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
                dim,
                device=Q.device,
                dtype=torch.bfloat16,
            )
            out_packed = torch.empty_like(q_packed)
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
                dim,
                rope_dims,
            )(
                Q,
                output,
                dout,
                combined_lse,
                cos,
                sin,
                q_packed,
                out_packed,
                dout_packed,
                lse_packed,
                Int32(head_offset),
                Int32(token_offset),
            )
            result = sparse_attention_backward_wrapper(
                q_packed,
                kv_packed,
                out_packed,
                dout_packed,
                lse_packed,
                selected_sink,
                selected_indices,
                softmax_scale=1.0 / math.sqrt(dim),
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
                dim,
                rope_dims,
                blocks,
                sequence,
            )(
                result["dq"],
                result["dkv"],
                cos,
                sin,
                dQ,
                dlocal,
                dcompressed,
                Int32(head_offset),
                Int32(token_offset),
            )
            d_sink_accumulator[head_offset : head_offset + packed_heads].add_(result["d_sink"])
            del (
                q_packed,
                out_packed,
                dout_packed,
                lse_packed,
                result,
            )

    d_attention_sink = torch.empty_like(attention_sink)
    compile_cast_gradient(dtype, heads)(
        d_sink_accumulator,
        d_attention_sink,
    )
    del (
        kv_packed,
        indices,
        topk_lengths,
        sink_fp32,
        d_sink_accumulator,
        output,
        combined_lse,
    )

    dKV = torch.empty_like(KV)
    dKV_weight_fp32 = torch.zeros_like(KV_norm_weight, dtype=torch.float32)
    compile_local_norm_backward(dtype, batch, sequence, dim, rope_dims)(
        KV,
        KV_norm_weight,
        cos,
        sin,
        dlocal,
        dKV,
        dKV_weight_fp32,
    )
    dKV_weight = torch.empty_like(KV_norm_weight)
    compile_cast_gradient(dtype, KV_norm_weight.numel())(
        dKV_weight_fp32,
        dKV_weight,
    )

    # The final partial compression block is never causally visible. Start at
    # zero so its input rows receive the same zero gradient as the eager oracle.
    dC = torch.zeros_like(C)
    dZ = torch.zeros_like(Z)
    dB_fp32 = torch.zeros_like(B, dtype=torch.float32)
    dcompressed_weight_fp32 = torch.zeros_like(
        compressed_kv_norm_weight,
        dtype=torch.float32,
    )
    compile_compression_backward(
        dtype,
        batch,
        sequence,
        dim,
        compression_rate,
        rope_dims,
    )(
        C,
        Z,
        B,
        compressed_kv_norm_weight,
        cos,
        sin,
        dcompressed,
        dC,
        dZ,
        dB_fp32,
        dcompressed_weight_fp32,
    )
    dB = torch.empty_like(B)
    dcompressed_weight = torch.empty_like(compressed_kv_norm_weight)
    compile_cast_gradient(dtype, B.numel())(
        dB_fp32.view(-1),
        dB.view(-1),
    )
    compile_cast_gradient(dtype, compressed_kv_norm_weight.numel())(
        dcompressed_weight_fp32,
        dcompressed_weight,
    )
    return (
        dQ,
        dKV,
        dC,
        dZ,
        dB,
        dKV_weight,
        dcompressed_weight,
        d_attention_sink,
    )


@torch.library.custom_op(
    "attention_gym::_cute_heavily_compressed_attention_backward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_heavily_compressed_attention_backward_op(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    dout: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with torch.cuda.device(Q.device):
        return _heavily_compressed_attention_backward(
            Q,
            KV,
            C,
            Z,
            B,
            KV_norm_weight,
            compressed_kv_norm_weight,
            attention_sink,
            dout,
            compression_rate,
            sliding_window_size,
            rope_dims,
            share_kv,
        )


@_cute_heavily_compressed_attention_backward_op.register_fake
def _cute_heavily_compressed_attention_backward_fake(
    Q,
    KV,
    C,
    Z,
    B,
    KV_norm_weight,
    compressed_kv_norm_weight,
    attention_sink,
    dout,
    compression_rate,
    sliding_window_size,
    rope_dims,
    share_kv,
):
    del (
        dout,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    return tuple(
        torch.empty_like(tensor)
        for tensor in (
            Q,
            KV,
            C,
            Z,
            B,
            KV_norm_weight,
            compressed_kv_norm_weight,
            attention_sink,
        )
    )


@torch.library.custom_op(
    "attention_gym::_cute_heavily_compressed_attention_forward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_heavily_compressed_attention_forward_op(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    with torch.cuda.device(Q.device):
        return _heavily_compressed_attention_forward(
            Q,
            KV,
            C,
            Z,
            B,
            KV_norm_weight,
            compressed_kv_norm_weight,
            attention_sink,
            compression_rate,
            sliding_window_size,
            rope_dims,
            share_kv,
        )


@_cute_heavily_compressed_attention_forward_op.register_fake
def _cute_heavily_compressed_attention_forward_fake(
    Q,
    KV,
    C,
    Z,
    B,
    KV_norm_weight,
    compressed_kv_norm_weight,
    attention_sink,
    compression_rate,
    sliding_window_size,
    rope_dims,
    share_kv,
):
    del (
        KV,
        C,
        Z,
        B,
        KV_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    return torch.empty_like(Q)


def _cute_heavily_compressed_attention_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:_DIFFERENTIABLE_INPUTS])
    (
        ctx.compression_rate,
        ctx.sliding_window_size,
        ctx.rope_dims,
        ctx.share_kv,
    ) = inputs[_DIFFERENTIABLE_INPUTS:]


def _cute_heavily_compressed_attention_autograd_backward(ctx, dout):
    grads = _cute_heavily_compressed_attention_backward_op(
        *ctx.saved_tensors,
        dout,
        ctx.compression_rate,
        ctx.sliding_window_size,
        ctx.rope_dims,
        ctx.share_kv,
    )
    return (*grads, None, None, None, None)


_cute_heavily_compressed_attention_forward_op.register_autograd(
    _cute_heavily_compressed_attention_autograd_backward,
    setup_context=_cute_heavily_compressed_attention_setup_context,
)


def heavily_compressed_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    return _cute_heavily_compressed_attention_forward_op(
        Q,
        KV,
        C,
        Z,
        B,
        KV_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["heavily_compressed_attention"]
