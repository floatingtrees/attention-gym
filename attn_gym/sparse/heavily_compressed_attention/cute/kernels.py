"""HCA-specific CuTe DSL kernels for the SM100 forward path."""

from __future__ import annotations

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor
from quack.reduce import row_reduce


_RMS_EPS = 1.1920928955078125e-7


class CompressionNormRope:
    """Fuse single-branch R-token compression, RMSNorm, and YaRN tail RoPE."""

    def __init__(
        self,
        dtype,
        batch: int,
        sequence: int,
        dim: int,
        rate: int,
        rope: int,
    ):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.rate = rate
        self.rope = rope
        # The reference materializes a trailing partial block, but its causal mask
        # makes that block unreachable. Do not spend work or storage on it.
        self.num_blocks = sequence // rate
        self.num_threads = math.gcd(dim // 2, 128)
        self.values_per_thread = dim // self.num_threads
        assert self.num_blocks > 0
        assert dim % self.num_threads == 0
        assert self.values_per_thread % 2 == 0

    @cute.jit
    def __call__(
        self,
        c: cute.Tensor,
        z: cute.Tensor,
        bias: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(c, z, bias, norm_weight, cos, sin, out).launch(
            grid=[self.batch * self.num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        c: cute.Tensor,
        z: cute.Tensor,
        bias: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block_row, _, _ = cute.arch.block_idx()
        batch_idx = block_row // self.num_blocks
        block_idx = block_row - batch_idx * self.num_blocks
        d_base = tid * self.values_per_thread

        compressed = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        local_sq = Float32(0.0)
        for j in cutlass.range_constexpr(self.values_per_thread):
            d = d_base + j
            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                position = block_idx * self.rate + r
                logit = (
                    z[batch_idx, 0, position, d].to(Float32)
                    + bias[r, d].to(Float32)
                ).to(self.dtype).to(Float32)
                maximum = cute.arch.fmax(maximum, logit)

            denominator = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                position = block_idx * self.rate + r
                logit = (
                    z[batch_idx, 0, position, d].to(Float32)
                    + bias[r, d].to(Float32)
                ).to(self.dtype).to(Float32)
                denominator += cute.math.exp(logit - maximum, fastmath=False)

            value = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                position = block_idx * self.rate + r
                logit = (
                    z[batch_idx, 0, position, d].to(Float32)
                    + bias[r, d].to(Float32)
                ).to(self.dtype).to(Float32)
                probability = (
                    cute.math.exp(logit - maximum, fastmath=False) / denominator
                ).to(self.dtype).to(Float32)
                product = (
                    c[batch_idx, 0, position, d].to(Float32) * probability
                ).to(self.dtype).to(Float32)
                value += product

            # Compression is materialized in the input dtype before RMSNorm.
            value = value.to(self.dtype).to(Float32)
            compressed[j] = value
            local_sq += value * value

        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
            byte_alignment=16,
        )
        sum_sq = row_reduce(
            local_sq,
            cute.ReductionOp.ADD,
            self.num_threads,
            reduction,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)
        rope_position = block_idx * self.rate

        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            x0 = (
                compressed[j0] * rstd * norm_weight[d0].to(Float32)
            ).to(self.dtype).to(Float32)
            x1 = (
                compressed[j0 + 1] * rstd * norm_weight[d0 + 1].to(Float32)
            ).to(self.dtype).to(Float32)
            y0, y1 = x0, x1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                cosine = cos[rope_position, rope_pair].to(Float32)
                sine = sin[rope_position, rope_pair].to(Float32)
                y0 = x0 * cosine - x1 * sine
                y1 = x0 * sine + x1 * cosine
            out[batch_idx, block_idx, 0, d0] = y0.to(self.dtype)
            out[batch_idx, block_idx, 0, d0 + 1] = y1.to(self.dtype)


class CausalGather:
    """Build one deterministic compressed-plus-local gather per packed M tile."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        heads: int,
        num_blocks: int,
        rate: int,
        window: int,
        gather_length: int,
    ):
        self.batch = batch
        self.sequence = sequence
        self.heads = heads
        self.num_blocks = num_blocks
        self.rate = rate
        self.window = min(window, sequence)
        self.gather_length = gather_length
        self.positions_per_cluster = 128 // heads
        self.gather_rows = (
            sequence + self.positions_per_cluster - 1
        ) // self.positions_per_cluster
        self.num_threads = 128
        assert heads in (64, 128) and 128 % heads == 0

    @cute.jit
    def __call__(self, gather: cute.Tensor, stream: cuda.CUstream):
        self.kernel(gather).launch(
            grid=[cute.ceil_div(self.batch * self.gather_rows, 4), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, gather: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        row = block * 4 + warp
        if row < self.batch * self.gather_rows:
            batch_idx = row // self.gather_rows
            packed_row = row - batch_idx * self.gather_rows
            position = min(
                self.sequence - 1,
                (packed_row + 1) * self.positions_per_cluster - 1,
            )
            completed = min(self.num_blocks, (position + 1) // self.rate)
            local_count = min(self.window + self.positions_per_cluster - 1, position + 1)
            for item in cutlass.range_constexpr(
                self.gather_length // cute.arch.WARP_SIZE
            ):
                slot = lane + item * cute.arch.WARP_SIZE
                value = Int32(-1)
                if slot < completed:
                    value = Int32(slot)
                else:
                    local_slot = slot - self.num_blocks
                    if local_slot >= 0 and local_slot < local_count:
                        key = position - local_count + 1 + local_slot
                        value = Int32(self.num_blocks + key)
                gather[batch_idx, packed_row, slot] = value


def _fake(dtype, shape):
    return make_fake_tensor(dtype, shape, math.gcd(8, shape[-1]))


@lru_cache
def compile_compression(dtype, batch, sequence, dim, rate, rope):
    blocks = sequence // rate
    return cute.compile(
        CompressionNormRope(dtype, batch, sequence, dim, rate, rope),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (rate, dim)),
        _fake(dtype, (dim,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, blocks, 1, dim)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_causal_gather(
    batch,
    sequence,
    heads,
    num_blocks,
    rate,
    window,
    gather_length,
):
    gather_rows = math.ceil(sequence / (128 // heads))
    return cute.compile(
        CausalGather(
            batch,
            sequence,
            heads,
            num_blocks,
            rate,
            window,
            gather_length,
        ),
        _fake(Int32, (batch, gather_rows, gather_length)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


__all__ = ["compile_causal_gather", "compile_compression"]
