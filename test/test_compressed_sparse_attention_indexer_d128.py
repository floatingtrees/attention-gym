"""Tests for the tensor-core indexer path with index_dim=128 (DSv4 Flash config)."""

import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.compressed_sparse_attention.api import compressed_sparse_attention

pytest.importorskip("flash_attn.cute.interface")

MAX_ABS_ERROR = 3.5e-2


def _indexer_reference(q, k, w, ratio, sm_scale):
    """Reference: S[b,q,t] = sm_scale * sum_h [ReLU(Q_h . K_t^T) * W_{b,q,h}]"""
    batch, seq, heads, dim = q.shape
    _, num_blocks, _, _ = k.shape

    k_exp = k.expand(batch, num_blocks, heads, dim)
    qk = torch.einsum("bshd,bnhd->bshn", q.float(), k_exp.float())
    qk_relu = torch.relu(qk)
    w_exp = w.float().unsqueeze(-1)
    score = (qk_relu * w_exp).sum(dim=2) * sm_scale

    for b in range(batch):
        for s in range(seq):
            col_limit = (s + 1) // ratio
            score[b, s, col_limit:] = float("-inf")

    return score


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the tensor-core indexer targets SM100 exclusively",
)
class TestIndexerDim128:
    """Tensor-core indexer correctness with index_dim=128."""

    def test_basic_correctness(self):
        from attn_gym.sparse.compressed_sparse_attention.cute.index_scores import (
            exact_bf16_index_scores,
        )

        batch, seq, index_heads, index_dim = 1, 64, 64, 128
        compression_rate = 4
        num_blocks = seq // compression_rate
        sm_scale = 1.0 / math.sqrt(index_dim * index_heads)

        torch.manual_seed(42)
        device = "cuda"
        q = torch.randn(batch, seq, index_heads, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        k = torch.randn(batch, num_blocks, 1, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(batch, seq, index_heads, device=device, dtype=torch.bfloat16) * 0.1
        out = torch.empty(batch, seq, num_blocks, device=device, dtype=torch.float32)

        result = exact_bf16_index_scores(
            q, k, w,
            ratio=compression_rate,
            qhead_per_kv_head=index_heads,
            out=out,
            sm_scale=sm_scale,
        )
        ref = _indexer_reference(q, k, w, compression_rate, sm_scale)

        valid = ref != float("-inf")
        assert (result[~valid] == float("-inf")).all(), "Causal mask mismatch"
        error = (result[valid] - ref[valid]).abs().max().item()
        assert error < 1e-4, f"Max error {error} exceeds tolerance"

    def test_larger_sequence(self):
        from attn_gym.sparse.compressed_sparse_attention.cute.index_scores import (
            exact_bf16_index_scores,
        )

        batch, seq, index_heads, index_dim = 2, 512, 64, 128
        compression_rate = 4
        num_blocks = seq // compression_rate
        sm_scale = 1.0 / math.sqrt(index_dim * index_heads)

        torch.manual_seed(99)
        device = "cuda"
        q = torch.randn(batch, seq, index_heads, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        k = torch.randn(batch, num_blocks, 1, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(batch, seq, index_heads, device=device, dtype=torch.bfloat16) * 0.1
        out = torch.empty(batch, seq, num_blocks, device=device, dtype=torch.float32)

        result = exact_bf16_index_scores(
            q, k, w,
            ratio=compression_rate,
            qhead_per_kv_head=index_heads,
            out=out,
            sm_scale=sm_scale,
        )
        ref = _indexer_reference(q, k, w, compression_rate, sm_scale)

        valid = ref != float("-inf")
        assert (result[~valid] == float("-inf")).all(), "Causal mask mismatch"
        error = (result[valid] - ref[valid]).abs().max().item()
        assert error < 1e-3, f"Max error {error} exceeds tolerance"

    def test_backward_compat_dim64(self):
        """Verify index_dim=64 still works with the updated kernel."""
        from attn_gym.sparse.compressed_sparse_attention.cute.index_scores import (
            exact_bf16_index_scores,
        )

        batch, seq, index_heads, index_dim = 1, 128, 64, 64
        compression_rate = 32
        num_blocks = seq // compression_rate
        sm_scale = 1.0 / math.sqrt(index_dim * index_heads)

        torch.manual_seed(7)
        device = "cuda"
        q = torch.randn(batch, seq, index_heads, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        k = torch.randn(batch, num_blocks, 1, index_dim, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(batch, seq, index_heads, device=device, dtype=torch.bfloat16) * 0.1
        out = torch.empty(batch, seq, num_blocks, device=device, dtype=torch.float32)

        result = exact_bf16_index_scores(
            q, k, w,
            ratio=compression_rate,
            qhead_per_kv_head=index_heads,
            out=out,
            sm_scale=sm_scale,
        )
        ref = _indexer_reference(q, k, w, compression_rate, sm_scale)

        valid = ref != float("-inf")
        assert (result[~valid] == float("-inf")).all()
        error = (result[valid] - ref[valid]).abs().max().item()
        assert error < 1e-4, f"Max error {error} exceeds tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
class TestCSAEndToEndDSv4Flash:
    """End-to-end CSA with DSv4 Flash parameters (index_dim=128)."""

    def _make_inputs(self, batch=1, seq=256, seed=123):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        generator = torch.Generator(device=device).manual_seed(seed)
        heads, head_dim = 128, 512
        index_heads, index_dim = 64, 128
        compression_rate, topk, window, rope_dims = 4, 64, 128, 64

        def randn(*shape, scale=0.2):
            return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * scale

        def query(*shape):
            return F.normalize(randn(*shape), dim=-1)

        return (
            query(batch, heads, seq, head_dim),
            query(batch, index_heads, seq, index_dim),
            randn(batch, 1, seq, head_dim),
            randn(batch, 1, seq, head_dim),
            randn(batch, 1, seq, head_dim),
            randn(batch, 1, seq, head_dim),
            randn(batch, 1, seq, head_dim),
            randn(compression_rate, head_dim),
            randn(compression_rate, head_dim),
            randn(batch, seq, index_heads),
            randn(batch, 1, seq, index_dim),
            randn(batch, 1, seq, index_dim),
            randn(batch, 1, seq, index_dim),
            randn(batch, 1, seq, index_dim),
            randn(compression_rate, index_dim),
            randn(compression_rate, index_dim),
            1.0 + randn(head_dim, scale=0.05),
            1.0 + randn(index_dim, scale=0.05),
            1.0 + randn(head_dim, scale=0.05),
            randn(heads),
            compression_rate,
            topk,
            window,
            rope_dims,
            True,
        )

    def test_dsv4_flash_matches_reference(self):
        inputs = self._make_inputs(batch=1, seq=256)
        with torch.inference_mode():
            expected = compressed_sparse_attention(*inputs, backend="eager")
            actual = compressed_sparse_attention(*inputs, backend="cute")

        error = (actual.float() - expected.float()).abs().max().item()
        assert error <= MAX_ABS_ERROR, f"Max error {error} > {MAX_ABS_ERROR}"

    def test_dsv4_flash_batch4(self):
        inputs = self._make_inputs(batch=4, seq=256)
        with torch.inference_mode():
            expected = compressed_sparse_attention(*inputs, backend="eager")
            actual = compressed_sparse_attention(*inputs, backend="cute")

        error = (actual.float() - expected.float()).abs().max().item()
        assert error <= MAX_ABS_ERROR, f"Max error {error} > {MAX_ABS_ERROR}"

    def test_dsv4_flash_long_sequence(self):
        inputs = self._make_inputs(batch=1, seq=1024, seed=456)
        with torch.inference_mode():
            expected = compressed_sparse_attention(*inputs, backend="eager")
            actual = compressed_sparse_attention(*inputs, backend="cute")

        error = (actual.float() - expected.float()).abs().max().item()
        # Longer sequences accumulate more BF16 rounding noise in the
        # multi-head attention reduce. Use a slightly relaxed bound.
        long_seq_tolerance = 4e-2
        assert error <= long_seq_tolerance, f"Max error {error} > {long_seq_tolerance}"
