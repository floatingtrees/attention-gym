"""SWA/HCA-owned launcher exposing the vendored DSA FP32 dKV workspace.

The generic DSA launcher converts its FP32 atomic accumulator to BF16 before
returning.  HCA and SWA invoke DSA repeatedly over query/head slabs, so adding
those returned BF16 tensors to an outer FP32 accumulator introduces one
quantization boundary per slab.  This experimental launcher returns the
pre-conversion FP32 workspace as a tensor view, allowing all slabs to be
combined in FP32 and cast only after the structured attention gradient is
complete.

The subclass retains the vendored tensor-core kernels but makes their final dKV
workspace conversion write FP32. Callers unpack dQ independently and accumulate
the returned dKV tensor without an intermediate BF16 conversion.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from .dsa_backward_sm100 import (
    FlashAttentionDSABackwardSm100,
    _TORCH_TO_CUTE_DTYPE,
    _to_cute_tensor,
)


class _FlashAttentionDSABackwardFp32Dkv(FlashAttentionDSABackwardSm100):
    """Retain the vendored kernel and replace only its final dKV conversion."""

    @cute.kernel
    def convert(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: cutlass.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        _, seq_block_idx, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy

        if seq_id < seqlen:
            cur_mdKV_acc_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            tile_mdKV_acc_row = cute.flat_divide(cur_mdKV_acc_row, (64,))
            tile_mdKV_acc_row = cute.flat_divide(tile_mdKV_acc_row, (32,))

            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    value = tile_mdKV_acc_row[tidx, j, i]
                    dimension = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dimension] = value

            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    value = tile_mdKV_acc_row[tidx, j, num_128_tiles]
                    k = tidx // 2 + j * 16
                    dimension = (
                        self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    )
                    cur_mdKV_row[dimension] = value


_compile_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()


def sparse_attention_backward_fp32_dkv(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float | None = None,
    topk_length: torch.Tensor | None = None,
    dq: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run vendored DSA and expose its FP32 dKV accumulator before conversion."""
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise RuntimeError("The vendored DSA backward targets SM100 exclusively.")
    total_q, num_heads, head_dim = q.shape
    total_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("The vendored DSA backward requires FP16 or BF16 inputs.")
    if not (q.dtype == kv.dtype == out.dtype == dout.dtype):
        raise TypeError("Q, KV, output, and output gradient must have the same dtype.")
    if lse.dtype != torch.float32 or attn_sink.dtype != torch.float32:
        raise TypeError("LSE and attention sink must be FP32.")
    if topk_idxs.dtype != torch.int32:
        raise TypeError("Sparse indices must be INT32.")

    q, kv, out, dout, lse = (tensor.contiguous() for tensor in (q, kv, out, dout, lse))
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if dq is None:
        dq = torch.empty_like(q)
    dkv_fp32 = torch.empty(
        total_kv,
        head_dim,
        device=kv.device,
        dtype=torch.float32,
    )
    d_sink = torch.zeros_like(attn_sink)

    block_tile = 64
    batch_size = 1
    workspace_lse_odo = torch.zeros(
        *FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
            total_q,
            head_dim,
            num_heads,
            batch_size,
            cutlass.Float32,
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    workspace_dkv = torch.zeros(
        *FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
            total_kv,
            head_dim,
            batch_size,
            cutlass.Float32,
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    problem_shape = (total_q, total_kv, head_dim, (num_heads, batch_size))
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    dtype = _TORCH_TO_CUTE_DTYPE[q.dtype]
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        num_heads,
        block_tile,
        topk_length is not None,
    )
    compiled = _compile_cache.get(compile_key)
    if compiled is None:
        kernel = _FlashAttentionDSABackwardFp32Dkv(head_dim, head_dim_v, block_tile)
        compiled = cute.compile(
            kernel,
            problem_shape,
            _to_cute_tensor(q, divisibility=head_dim),
            _to_cute_tensor(kv, divisibility=head_dim),
            _to_cute_tensor(out, divisibility=head_dim_v),
            _to_cute_tensor(dout, divisibility=head_dim_v),
            _to_cute_tensor(lse, assumed_align=4),
            _to_cute_tensor(attn_sink),
            _to_cute_tensor(topk_idxs),
            _to_cute_tensor(topk_length) if topk_length is not None else None,
            _to_cute_tensor(dq, divisibility=head_dim),
            _to_cute_tensor(dkv_fp32, divisibility=head_dim),
            _to_cute_tensor(d_sink),
            _to_cute_tensor(workspace_lse_odo),
            _to_cute_tensor(workspace_dkv),
            softmax_scale,
            stream,
            options="--enable-tvm-ffi",
        )
        _compile_cache[compile_key] = compiled
        if len(_compile_cache) > 64:
            _compile_cache.popitem(last=False)
    else:
        _compile_cache.move_to_end(compile_key)

    compiled(
        problem_shape,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length,
        dq,
        dkv_fp32,
        d_sink,
        workspace_lse_odo,
        workspace_dkv,
        softmax_scale,
        stream,
    )

    return {
        "dq": dq,
        "dkv_fp32": dkv_fp32,
        "d_sink": d_sink,
    }


__all__ = ["sparse_attention_backward_fp32_dkv"]
