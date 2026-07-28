import importlib

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.heavily_compressed_attention.api import (
    heavily_compressed_attention,
)

pytest.importorskip("flash_attn.cute.interface")
hca_cute = importlib.import_module("attn_gym.sparse.heavily_compressed_attention.cute")

MAX_ABS_ERROR = 3e-2


def make_inputs(
    *,
    batch=1,
    heads=64,
    sequence=256,
    dim=512,
    compression_rate=128,
    window=128,
    rope_dims=64,
    seed=123,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=0.2):
        return (
            torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    return (
        F.normalize(randn(batch, heads, sequence, dim), dim=-1),
        randn(batch, 1, sequence, dim),
        randn(batch, 1, sequence, dim),
        randn(batch, 1, sequence, dim),
        randn(compression_rate, dim),
        1.0 + randn(dim, scale=0.05),
        1.0 + randn(dim, scale=0.05),
        randn(heads),
        compression_rate,
        window,
        rope_dims,
        True,
    )


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend requires SM100",
)


@requires_sm100
@pytest.mark.parametrize("sequence", [256, 385])
def test_cute_matches_reference_with_absolute_error_bound(sequence):
    inputs = make_inputs(sequence=sequence)
    with torch.inference_mode():
        expected = heavily_compressed_attention(*inputs, backend="eager")
        actual = heavily_compressed_attention(*inputs, backend="cute")

    error = (actual.float() - expected.float()).abs()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert error.max().item() <= MAX_ABS_ERROR


def assert_backward_matches_reference(base):
    def differentiable_copy():
        return tuple(
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor)
            else value
            for value in base
        )

    reference_inputs = differentiable_copy()
    cute_inputs = differentiable_copy()
    expected = heavily_compressed_attention(*reference_inputs, backend="eager")
    actual = heavily_compressed_attention(*cute_inputs, backend="cute")

    saved = actual.grad_fn.saved_tensors
    input_pointers = {value.data_ptr() for value in cute_inputs[:8]}
    assert len(saved) == 8
    assert all(value.data_ptr() in input_pointers for value in saved)

    generator = torch.Generator(device=actual.device).manual_seed(456)
    grad_output = (
        torch.randn(
            actual.shape,
            device=actual.device,
            dtype=actual.dtype,
            generator=generator,
        )
        * 0.01
    )
    expected.backward(grad_output)
    actual.backward(grad_output)

    for index, (cute_input, reference_input) in enumerate(
        zip(cute_inputs[:8], reference_inputs[:8])
    ):
        assert cute_input.grad is not None
        assert reference_input.grad is not None
        error = (cute_input.grad.float() - reference_input.grad.float()).abs().max()
        assert error.item() <= MAX_ABS_ERROR, f"input {index} max error {error.item()}"


@requires_sm100
def test_cute_backward_matches_reference():
    assert_backward_matches_reference(
        make_inputs(
            sequence=35,
            compression_rate=16,
            window=16,
        )
    )


@requires_sm100
def test_cute_backward_handles_cross_batch_token_slabs(monkeypatch):
    batch = 2
    sequence = 35
    monkeypatch.setattr(
        hca_cute,
        "_DSA_PACKED_WORKSPACE_BYTES",
        10 * 1024**2,
    )
    head_chunk, token_chunk = hca_cute._dsa_tile_shape(
        batch * sequence,
        512,
        64,
        batch * (sequence // 16 + sequence),
        64,
    )
    assert head_chunk == 64
    assert sequence < token_chunk < batch * sequence

    assert_backward_matches_reference(
        make_inputs(
            batch=batch,
            sequence=sequence,
            compression_rate=16,
            window=16,
        )
    )


@requires_sm100
def test_cute_rejects_non_dv4_head_shape():
    inputs = list(make_inputs(heads=32))
    with pytest.raises(ValueError, match=r"H=64 and D=512"):
        heavily_compressed_attention(*inputs, backend="cute")


def test_useful_pair_formula_matches_dv4_benchmark():
    sequence, window, rate = 4096, 128, 128
    local_pairs = sum(min(window, query + 1) for query in range(sequence))
    compressed_pairs = sum((query + 1) // rate for query in range(sequence))
    assert local_pairs == 516_160
    assert compressed_pairs == 63_520
    flops = 4 * 8 * 64 * 512 * (local_pairs + compressed_pairs)
    assert flops == 607_838_535_680
