import importlib

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.heavily_compressed_attention.api import heavily_compressed_attention

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


PATHOLOGICAL_SHAPES = [
    pytest.param(
        {
            "heads": 1,
            "sequence": 1,
            "compression_rate": 1,
            "window": 1,
            "rope_dims": 2,
        },
        id="minimum-valid-shape",
    ),
    pytest.param(
        {
            "heads": 127,
            "sequence": 31,
            "compression_rate": 32,
            "window": 64,
            "rope_dims": 2,
        },
        id="head-tile-minus-one-and-local-only",
    ),
    pytest.param(
        {
            "heads": 128,
            "sequence": 32,
            "compression_rate": 32,
            "window": 32,
            "rope_dims": 32,
        },
        id="head-and-compression-tile-exact",
    ),
    pytest.param(
        {
            "heads": 129,
            "sequence": 33,
            "compression_rate": 32,
            "window": 33,
            "rope_dims": 64,
        },
        id="head-and-compression-tile-plus-one",
    ),
    pytest.param(
        {
            "heads": 128,
            "sequence": 65,
            "compression_rate": 17,
            "window": 33,
            "rope_dims": 320,
        },
        id="odd-sequence-and-wide-rope",
    ),
    pytest.param(
        {
            "heads": 1,
            "sequence": 130,
            "compression_rate": 2,
            "window": 64,
            "rope_dims": 64,
        },
        id="split-local-and-compressed",
    ),
    pytest.param(
        {
            "batch": 3,
            "heads": 5,
            "sequence": 17,
            "compression_rate": 17,
            "window": 99,
            "rope_dims": 512,
        },
        id="batch-full-rope-and-clamped-window",
    ),
    pytest.param(
        {"sequence": 17, "compression_rate": 32, "window": 0},
        id="sink-only",
    ),
    pytest.param(
        {"sequence": 17, "compression_rate": 8, "window": 0},
        id="compressed-only",
    ),
]


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


@requires_sm100
@pytest.mark.parametrize("heads", [63, 64, 65])
def test_cute_matches_reference_around_h64_fast_path_with_odd_sequence(heads):
    inputs = make_inputs(
        heads=heads,
        sequence=129,
        compression_rate=128,
        window=128,
    )
    with torch.inference_mode():
        expected = heavily_compressed_attention(*inputs, backend="eager")
        actual = heavily_compressed_attention(*inputs, backend="cute")

    error = (actual.float() - expected.float()).abs()
    assert actual.shape == expected.shape
    assert error.max().item() <= MAX_ABS_ERROR


@requires_sm100
@pytest.mark.parametrize("overrides", PATHOLOGICAL_SHAPES)
def test_cute_generalized_shapes_match_reference(overrides):
    inputs = make_inputs(**overrides)
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
    assert_backward_matches_reference(make_inputs(sequence=35, compression_rate=16, window=16))


@requires_sm100
def test_cute_backward_handles_cross_batch_token_slabs(monkeypatch):
    batch = 2
    sequence = 35
    monkeypatch.setattr(hca_cute, "_DSA_PACKED_WORKSPACE_BYTES", 10 * 1024**2)
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
@pytest.mark.parametrize("heads", [127, 129])
def test_cute_backward_handles_pathological_head_counts(heads):
    assert_backward_matches_reference(
        make_inputs(
            heads=heads,
            sequence=17,
            compression_rate=8,
            window=9,
        )
    )


@requires_sm100
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"sequence": 17, "compression_rate": 32, "window": 9},
            id="local-only",
        ),
        pytest.param(
            {"batch": 2, "sequence": 17, "compression_rate": 8, "window": 0},
            id="batched-compressed-only",
        ),
        pytest.param(
            {"sequence": 17, "compression_rate": 32, "window": 0},
            id="sink-only",
        ),
        pytest.param(
            {"heads": 1, "sequence": 130, "compression_rate": 2, "window": 64},
            id="split-local-and-compressed",
        ),
    ],
)
def test_cute_backward_handles_attention_modes(overrides):
    assert_backward_matches_reference(make_inputs(**overrides))


def _all_tensor_dtype(values, dtype):
    return tuple(
        value.to(dtype=dtype) if isinstance(value, torch.Tensor) else value for value in values
    )


def _expanded_kv_inputs(values, *, share_kv):
    transformed = list(values)
    heads = transformed[0].shape[1]
    for index in (1, 2, 3):
        transformed[index] = transformed[index].expand(-1, heads, -1, -1).contiguous()
    transformed[-1] = share_kv
    return tuple(transformed)


def _expanded_input(values, index):
    transformed = list(values)
    transformed[index] = transformed[index].expand(
        -1,
        transformed[0].shape[1],
        -1,
        -1,
    )
    return tuple(transformed)


def _noncontiguous_query(values):
    transformed = list(values)
    transformed[0] = transformed[0].transpose(1, 2).contiguous().transpose(1, 2)
    assert not transformed[0].is_contiguous()
    return tuple(transformed)


@requires_sm100
@pytest.mark.parametrize(
    ("make_case", "match"),
    [
        pytest.param(
            lambda: make_inputs(sequence=17, dim=256, compression_rate=8),
            r"D=512",
            id="head-dim",
        ),
        pytest.param(
            lambda: _all_tensor_dtype(
                make_inputs(sequence=17, compression_rate=8),
                torch.float16,
            ),
            "bfloat16 inputs only",
            id="dtype",
        ),
        pytest.param(
            lambda: _expanded_input(
                make_inputs(sequence=17, compression_rate=8),
                1,
            ),
            "KV must physically have one KV head",
            id="physical-kv-head",
        ),
        pytest.param(
            lambda: _expanded_kv_inputs(
                make_inputs(sequence=17, compression_rate=8),
                share_kv=False,
            ),
            "share_kv=True",
            id="share-kv",
        ),
        pytest.param(
            lambda: _noncontiguous_query(
                make_inputs(sequence=17, compression_rate=8),
            ),
            "Q must be contiguous",
            id="contiguous",
        ),
    ],
)
def test_cute_rejects_unsupported_configuration(make_case, match):
    with pytest.raises((TypeError, ValueError), match=match):
        heavily_compressed_attention(*make_case(), backend="cute")


@requires_sm100
def test_cute_fullgraph_compile_forward_backward_matches_reference():
    base = make_inputs(heads=5, sequence=17, compression_rate=8, window=9)

    def differentiable_copy(values):
        return tuple(
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor)
            else value
            for value in values
        )

    reference_inputs = differentiable_copy(base)
    compiled_inputs = differentiable_copy(base)
    expected = heavily_compressed_attention(*reference_inputs, backend="eager")

    def run(*tensor_args):
        return heavily_compressed_attention(
            *tensor_args,
            base[8],
            base[9],
            base[10],
            base[11],
            backend="cute",
        )

    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    actual = compiled(*compiled_inputs[:8])
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

    assert (actual.float() - expected.float()).abs().max().item() <= MAX_ABS_ERROR
    for index, (compiled_input, reference_input) in enumerate(
        zip(compiled_inputs[:8], reference_inputs[:8])
    ):
        assert compiled_input.grad is not None
        assert reference_input.grad is not None
        error = (compiled_input.grad.float() - reference_input.grad.float()).abs().max()
        assert error.item() <= MAX_ABS_ERROR, f"input {index} max error {error.item()}"

    fresh = make_inputs(
        heads=5,
        sequence=17,
        compression_rate=8,
        window=9,
        seed=789,
    )
    fresh_reference_inputs = differentiable_copy(fresh)
    fresh_compiled_inputs = differentiable_copy(fresh)
    fresh_expected = heavily_compressed_attention(
        *fresh_reference_inputs,
        backend="eager",
    )
    fresh_actual = compiled(*fresh_compiled_inputs[:8])
    fresh_error = (fresh_actual.float() - fresh_expected.float()).abs().max()
    assert fresh_error.item() <= MAX_ABS_ERROR

    fresh_generator = torch.Generator(device=fresh_actual.device).manual_seed(790)
    fresh_grad_output = (
        torch.randn(
            fresh_actual.shape,
            device=fresh_actual.device,
            dtype=fresh_actual.dtype,
            generator=fresh_generator,
        )
        * 0.01
    )
    fresh_expected.backward(fresh_grad_output)
    fresh_actual.backward(fresh_grad_output)
    for index, (compiled_input, reference_input) in enumerate(
        zip(fresh_compiled_inputs[:8], fresh_reference_inputs[:8])
    ):
        assert compiled_input.grad is not None
        assert reference_input.grad is not None
        error = (compiled_input.grad.float() - reference_input.grad.float()).abs().max()
        assert error.item() <= MAX_ABS_ERROR, f"fresh input {index} max error {error.item()}"


def test_h64_paired_fast_path_supports_more_than_64_compressed_blocks():
    sequence = 65 * 128 + 1
    assert hca_cute._use_paired_h64_attention(
        batch=1,
        sequence=sequence,
        blocks=sequence // 128,
        window=128,
    )


def test_dsa_tile_shape_respects_workspace_budget():
    tokens = 8192
    dim = 512
    heads = 128
    total_kv = 8256
    index_width = 192

    head_tile, token_tile = hca_cute._dsa_tile_shape(
        tokens,
        dim,
        heads,
        total_kv,
        index_width,
    )

    assert 0 < head_tile <= heads
    assert 0 < token_tile < tokens
    assert (
        hca_cute._dsa_workspace_bytes(
            token_tile,
            dim,
            head_tile,
            total_kv,
            index_width,
        )
        <= hca_cute._DSA_PACKED_WORKSPACE_BYTES
    )


def test_cute_rejects_non_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))
    with pytest.raises(RuntimeError, match="SM100 exclusively"):
        hca_cute._require_sm100(torch.device("cuda"))


def test_useful_pair_formula_matches_dv4_benchmark():
    sequence, window, rate = 4096, 128, 128
    local_pairs = sum(min(window, query + 1) for query in range(sequence))
    compressed_pairs = sum((query + 1) // rate for query in range(sequence))
    assert local_pairs == 516_160
    assert compressed_pairs == 63_520
    flops = 4 * 8 * 64 * 512 * (local_pairs + compressed_pairs)
    assert flops == 607_838_535_680
