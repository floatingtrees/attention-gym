"""HCA-specific CuTe DSL backward kernels."""

from __future__ import annotations

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, cute
from quack.compile_utils import make_fake_tensor
from quack.reduce import row_reduce

_RMS_EPS = 1.1920928955078125e-7


class PackDsaIndices:
    """Pack deterministic HCA compressed and local indices for one token slab."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        blocks: int,
        rate: int,
        window: int,
        width: int,
        token_capacity: int,
    ):
        self.batch = batch
        self.sequence = sequence
        self.blocks = blocks
        self.rate = rate
        self.window = min(window, sequence)
        self.width = width
        self.token_capacity = token_capacity
        self.local_length = sequence if self.window > 0 else 0
        self.kv_per_batch = blocks + self.local_length

    @cute.jit
    def __call__(
        self,
        indices: cute.Tensor,
        lengths: cute.Tensor,
        token_offset: Int32,
        active_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            indices,
            lengths,
            token_offset,
            active_tokens,
        ).launch(
            grid=[cute.ceil_div(active_tokens * self.width, 256), 1, 1],
            block=[256, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        indices: cute.Tensor,
        lengths: cute.Tensor,
        token_offset: Int32,
        active_tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        linear = bid * 256 + tid
        total = active_tokens * self.width
        if linear < total:
            slot = linear % self.width
            slab_token = linear // self.width
            token = token_offset + slab_token
            position = token % self.sequence
            batch_idx = token // self.sequence
            compressed_count = (position + 1) // self.rate
            local_count = min(self.window, position + 1)
            valid_length = compressed_count + local_count
            index = Int32(-1)
            base = batch_idx * self.kv_per_batch
            if slot < compressed_count:
                index = Int32(base + slot)
            elif slot < valid_length:
                local_slot = slot - compressed_count
                key = position - local_count + 1 + local_slot
                index = Int32(base + self.blocks + key)
            indices[slab_token, slot] = index
            if slot == 0:
                lengths[slab_token] = Int32(valid_length)


class CompressionBackward:
    """Differentiate HCA compression, RMSNorm, and compressed-position RoPE."""

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
        self.blocks = sequence // rate
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads
        assert self.blocks > 0
        assert dim == 512 and self.values_per_thread == 4

    @cute.jit
    def __call__(
        self,
        c: cute.Tensor,
        z: cute.Tensor,
        bias: cute.Tensor,
        weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        dy: cute.Tensor,
        dc: cute.Tensor,
        dz: cute.Tensor,
        dbias: cute.Tensor,
        dweight: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            c,
            z,
            bias,
            weight,
            cos,
            sin,
            dy,
            dc,
            dz,
            dbias,
            dweight,
        ).launch(
            grid=[self.batch * self.blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        c: cute.Tensor,
        z: cute.Tensor,
        bias: cute.Tensor,
        weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        dy: cute.Tensor,
        dc: cute.Tensor,
        dz: cute.Tensor,
        dbias: cute.Tensor,
        dweight: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        batch_idx = row // self.blocks
        block = row - batch_idx * self.blocks
        d_base = tid * self.values_per_thread
        raw = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        grad = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        local_sq = Float32(0.0)

        # Match the BF16 boundaries in the fused forward compressor.
        for j in cutlass.range_constexpr(self.values_per_thread):
            d = d_base + j
            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                maximum = cute.arch.fmax(maximum, logit)

            denominator = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                denominator += cute.math.exp(logit - maximum, fastmath=False)

            value = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                probability = (
                    (cute.math.exp(logit - maximum, fastmath=False) / denominator)
                    .to(self.dtype)
                    .to(Float32)
                )
                product = (
                    (c[batch_idx, 0, position, d].to(Float32) * probability)
                    .to(self.dtype)
                    .to(Float32)
                )
                value += product
            value = value.to(self.dtype).to(Float32)
            raw[j] = value
            local_sq += value * value

        # Undo the output RoPE before differentiating RMSNorm.
        rope_position = block * self.rate
        local_dot = Float32(0.0)
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j = pair * 2
            d = d_base + j
            g0 = dy[batch_idx, block, 0, d]
            g1 = dy[batch_idx, block, 0, d + 1]
            if d >= self.dim - self.rope:
                rope_pair = (d - (self.dim - self.rope)) // 2
                cosine = cos[rope_position, rope_pair]
                sine = sin[rope_position, rope_pair]
                u0 = g0 * cosine + g1 * sine
                u1 = -g0 * sine + g1 * cosine
                g0, g1 = u0, u1
            grad[j], grad[j + 1] = g0, g1
            local_dot += g0 * weight[d].to(Float32) * raw[j]
            local_dot += g1 * weight[d + 1].to(Float32) * raw[j + 1]

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
        dot = row_reduce(
            local_dot,
            cute.ReductionOp.ADD,
            self.num_threads,
            reduction,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)
        correction = dot * rstd * rstd / self.dim

        for j in cutlass.range_constexpr(self.values_per_thread):
            d = d_base + j
            draw = rstd * (grad[j] * weight[d].to(Float32) - raw[j] * correction)
            cute.arch.atomic_add(
                dweight.iterator + d,
                grad[j] * raw[j] * rstd,
            )

            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                maximum = cute.arch.fmax(maximum, logit)

            denominator = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                denominator += cute.math.exp(logit - maximum, fastmath=False)

            for r in cutlass.range(self.rate, unroll=1):
                position = block * self.rate + r
                logit = (
                    (z[batch_idx, 0, position, d].to(Float32) + bias[r, d].to(Float32))
                    .to(self.dtype)
                    .to(Float32)
                )
                probability = (
                    (cute.math.exp(logit - maximum, fastmath=False) / denominator)
                    .to(self.dtype)
                    .to(Float32)
                )
                c_value = c[batch_idx, 0, position, d].to(Float32)
                dc[batch_idx, 0, position, d] = (probability * draw).to(self.dtype)
                z_gradient = probability * (c_value - raw[j]) * draw
                dz[batch_idx, 0, position, d] = z_gradient.to(self.dtype)
                cute.arch.atomic_add(
                    dbias.iterator + r * self.dim + d,
                    z_gradient,
                )


def _fake(dtype, shape):
    return make_fake_tensor(dtype, shape, math.gcd(8, shape[-1]))


@lru_cache
def compile_pack_dsa_indices(
    batch,
    sequence,
    blocks,
    rate,
    window,
    width,
    token_capacity,
):
    return cute.compile(
        PackDsaIndices(
            batch,
            sequence,
            blocks,
            rate,
            window,
            width,
            token_capacity,
        ),
        _fake(Int32, (token_capacity, width)),
        _fake(Int32, (token_capacity,)),
        Int32(0),
        Int32(token_capacity),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_compression_backward(dtype, batch, sequence, dim, rate, rope):
    blocks = sequence // rate
    return cute.compile(
        CompressionBackward(dtype, batch, sequence, dim, rate, rope),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (rate, dim)),
        _fake(dtype, (dim,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (batch, blocks, 1, dim)),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(Float32, (rate, dim)),
        _fake(Float32, (dim,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


__all__ = ["compile_compression_backward", "compile_pack_dsa_indices"]
