"""SM100 CuTe DSL backend for shared-KV heavily compressed attention.

The path is specialized for the DeepSeek-V4 HCA shape (H=64, D=512).  It reuses
the vendored FA4 QV-only tensor-core loops from CSA, but has no indexer or top-k:
the compressed gather is the deterministic prefix of completed blocks.
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
from ...compressed_sparse_attention.cute.local_attention import compressed_attention
from .kernels import compile_causal_gather, compile_compression


_HEADS = 64
_HEAD_DIM = 512
_TESTED_CUDA_VERSION = "13.3"
_ROPE_CACHE_MAXSIZE = 8
_rope_table_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
] = OrderedDict()
_query_streams: dict[int, torch.cuda.Stream] = {}
_compressed_streams: dict[int, torch.cuda.Stream] = {}


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
        raise ValueError(
            f"The SM100 CuTe specialization requires H={_HEADS} and D={_HEAD_DIM}."
        )
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
                rope_dims
                * math.log(65_536 / (32.0 * 2 * math.pi))
                / correction_scale
            )
            high = math.ceil(
                rope_dims
                * math.log(65_536 / (1.0 * 2 * math.pi))
                / correction_scale
            )
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
) -> torch.Tensor:
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
        sink=attention_sink,
        cos=cos,
        sin=sin,
        rope_dims=rope_dims,
        csa_topk=num_blocks,
        csa_window=window,
        csa_rate=compression_rate,
        store_lse=False,
    )
    if attention_sequence != sequence:
        return output[:, :, :sequence].contiguous()
    return output


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
