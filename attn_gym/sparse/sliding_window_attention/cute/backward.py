"""SWA-specific CuTe DSL backward metadata kernels."""

from __future__ import annotations

from functools import lru_cache
import math

import cuda.bindings.driver as cuda
from cutlass import Int32, cute
from quack.compile_utils import make_fake_tensor


class PackDsaLocalIndices:
    """Pack one causal sliding-window index slab for the tensor-core DSA backward."""

    def __init__(
        self,
        sequence: int,
        window: int,
        width: int,
        token_capacity: int,
    ):
        self.sequence = sequence
        self.window = min(window, sequence)
        self.width = width
        self.token_capacity = token_capacity

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
            local_count = min(self.window, position + 1)
            index = Int32(-1)
            if slot < local_count:
                key = position - local_count + 1 + slot
                index = Int32(batch_idx * self.sequence + key)
            indices[slab_token, slot] = index
            if slot == 0:
                lengths[slab_token] = Int32(local_count)


def _fake(dtype, shape):
    return make_fake_tensor(dtype, shape, math.gcd(8, shape[-1]))


@lru_cache
def compile_pack_dsa_local_indices(
    sequence: int,
    window: int,
    width: int,
    token_capacity: int,
):
    return cute.compile(
        PackDsaLocalIndices(
            sequence,
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


__all__ = ["compile_pack_dsa_local_indices"]
