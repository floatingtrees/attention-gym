import argparse
import importlib

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.sliding_window_attention.api import sliding_window_attention

pytest.importorskip("flash_attn.cute.interface")
swa_cute = importlib.import_module("attn_gym.sparse.sliding_window_attention.cute")

MAX_ABS_ERROR = 3e-2

SWA_PATHOLOGICAL_SHAPES = [
    pytest.param(
        {"heads": 1, "sequence_length": 1, "window": 1, "rope_dims": 2},
        id="minimum-valid-shape",
    ),
    pytest.param(
        {
            "batch": 3,
            "heads": 5,
            "sequence_length": 17,
            "window": 99,
            "rope_dims": 512,
        },
        id="batch3-full-rope-and-clamped-window",
    ),
    pytest.param(
        {"heads": 127, "sequence_length": 31, "window": 64, "rope_dims": 2},
        id="head-tile-minus-one",
    ),
    pytest.param(
        {"heads": 128, "sequence_length": 32, "window": 32, "rope_dims": 32},
        id="head-tile-exact",
    ),
    pytest.param(
        {"heads": 129, "sequence_length": 33, "window": 33, "rope_dims": 64},
        id="head-tile-plus-one",
    ),
    pytest.param(
        {"heads": 128, "sequence_length": 65, "window": 33, "rope_dims": 320},
        id="wide-rope-crosses-mma-splits",
    ),
    pytest.param(
        {"sequence_length": 127, "window": 64},
        id="sequence-tile-minus-one",
    ),
    pytest.param(
        {"sequence_length": 128, "window": 64},
        id="sequence-tile-exact",
    ),
]


def make_inputs(args: argparse.Namespace):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * scale
        )

    query = F.normalize(
        randn(args.batch, args.heads, args.sequence_length, args.head_dim),
        dim=-1,
    )
    return (
        query,
        randn(args.batch, 1, args.sequence_length, args.head_dim),
        1.0 + randn(args.head_dim, scale=0.05),
        randn(args.heads),
        args.window,
        args.rope_dims,
        True,
    )


def _inputs(**overrides):
    configuration = {
        "batch": 1,
        "heads": 64,
        "sequence_length": 256,
        "head_dim": 512,
        "window": 128,
        "rope_dims": 64,
        "seed": 123,
    }
    configuration.update(overrides)
    return make_inputs(argparse.Namespace(**configuration))


requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0)),
    reason="the CuTe backend requires an SM100 CUDA GPU",
)


@requires_sm100
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="h64-standard-window"),
        pytest.param(
            {"sequence_length": 129, "window": 999, "rope_dims": 512},
            id="clamped-window-full-rope",
        ),
        pytest.param({"sequence_length": 65, "window": 1}, id="self-only"),
        *SWA_PATHOLOGICAL_SHAPES,
    ],
)
def test_cute_matches_reference(overrides):
    inputs = _inputs(**overrides)
    with torch.inference_mode():
        expected = sliding_window_attention(*inputs, backend="eager")
        actual = sliding_window_attention(*inputs, backend="cute")

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    error = (actual.float() - expected.float()).abs()
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
    expected = sliding_window_attention(*reference_inputs, backend="eager")
    actual = sliding_window_attention(*cute_inputs, backend="cute")

    saved = actual.grad_fn.saved_tensors
    input_pointers = {value.data_ptr() for value in cute_inputs[:4]}
    assert len(saved) == 4
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
        zip(cute_inputs[:4], reference_inputs[:4])
    ):
        assert cute_input.grad is not None
        assert reference_input.grad is not None
        error = (cute_input.grad.float() - reference_input.grad.float()).abs().max()
        assert error.item() <= MAX_ABS_ERROR, f"input {index} max error {error.item()}"


@requires_sm100
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"heads": 1, "sequence_length": 1, "window": 1, "rope_dims": 2},
            id="minimum-valid-shape",
        ),
        pytest.param(
            {"heads": 5, "sequence_length": 17, "window": 99, "rope_dims": 512},
            id="partial-head-tile-and-clamped-window",
        ),
        pytest.param(
            {"batch": 2, "heads": 5, "sequence_length": 17, "window": 9},
            id="batched-partial-head-tile",
        ),
        pytest.param(
            {"heads": 127, "sequence_length": 17, "window": 9},
            id="head-tile-minus-one",
        ),
        pytest.param(
            {"heads": 129, "sequence_length": 17, "window": 9},
            id="head-tile-plus-one",
        ),
        pytest.param(
            {"heads": 128, "sequence_length": 17, "window": 0},
            id="h128-zero-window",
        ),
    ],
)
def test_cute_backward_matches_reference(overrides):
    assert_backward_matches_reference(_inputs(**overrides))


@requires_sm100
def test_cute_backward_handles_cross_batch_token_slabs(monkeypatch):
    batch = 2
    sequence = 35
    monkeypatch.setattr(swa_cute, "_DSA_PACKED_WORKSPACE_BYTES", 10 * 1024**2)
    head_chunk, token_chunk = swa_cute._dsa_tile_shape(
        batch * sequence,
        512,
        64,
        batch * sequence,
        64,
    )
    assert head_chunk == 64
    assert sequence < token_chunk < batch * sequence
    assert_backward_matches_reference(
        _inputs(
            batch=batch,
            heads=64,
            sequence_length=sequence,
            window=16,
        )
    )


@requires_sm100
def test_cute_fullgraph_compile_forward_backward_matches_reference():
    base = _inputs(heads=5, sequence_length=17, window=9)

    def differentiable_copy(inputs):
        return tuple(
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor)
            else value
            for value in inputs
        )

    reference_inputs = differentiable_copy(base)
    compiled_inputs = differentiable_copy(base)

    def run(query, kv, kv_norm_weight, attention_sink):
        return sliding_window_attention(
            query,
            kv,
            kv_norm_weight,
            attention_sink,
            9,
            64,
            True,
            backend="cute",
        )

    expected = sliding_window_attention(*reference_inputs, backend="eager")
    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    actual = compiled(*compiled_inputs[:4])

    output_error = (actual.float() - expected.float()).abs().max()
    assert output_error.item() <= MAX_ABS_ERROR

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

    for index, (compiled_input, reference_input) in enumerate(
        zip(compiled_inputs[:4], reference_inputs[:4])
    ):
        assert compiled_input.grad is not None
        assert reference_input.grad is not None
        error = (compiled_input.grad.float() - reference_input.grad.float()).abs().max()
        assert error.item() <= MAX_ABS_ERROR, f"input {index} max error {error.item()}"

    fresh = _inputs(heads=5, sequence_length=17, window=9, seed=789)
    fresh_reference_inputs = differentiable_copy(fresh)
    fresh_compiled_inputs = differentiable_copy(fresh)
    fresh_expected = sliding_window_attention(
        *fresh_reference_inputs,
        backend="eager",
    )
    fresh_actual = compiled(*fresh_compiled_inputs[:4])
    fresh_error = (fresh_actual.float() - fresh_expected.float()).abs().max()
    assert fresh_error.item() <= MAX_ABS_ERROR


@requires_sm100
def test_zero_window_is_exactly_zero():
    inputs = _inputs(sequence_length=17, window=0)
    with torch.inference_mode():
        actual = sliding_window_attention(*inputs, backend="cute")

    assert torch.count_nonzero(actual).item() == 0


@requires_sm100
@pytest.mark.parametrize(
    ("transform", "match"),
    [
        (
            lambda values: values[:1] + (values[1].expand(-1, 64, -1, -1),) + values[2:],
            "one shared head",
        ),
        (lambda values: values[:6] + (False,), "share_kv=True"),
        (
            lambda values: tuple(
                value.to(torch.float16) if isinstance(value, torch.Tensor) else value
                for value in values
            ),
            "bfloat16 inputs only",
        ),
    ],
)
def test_cute_rejects_unsupported_configuration(transform, match):
    inputs = _inputs(sequence_length=17)
    transformed = transform(inputs)

    with pytest.raises((TypeError, ValueError), match=match):
        sliding_window_attention(*transformed, backend="cute")


def test_dsa_tile_shape_respects_workspace_budget():
    tokens = 8192
    dim = 512
    heads = 128
    total_kv = tokens
    index_width = 128

    head_tile, token_tile = swa_cute._dsa_tile_shape(
        tokens,
        dim,
        heads,
        total_kv,
        index_width,
    )

    assert 0 < head_tile <= heads
    assert 0 < token_tile < tokens
    assert (
        swa_cute._dsa_workspace_bytes(
            token_tile,
            dim,
            head_tile,
            total_kv,
            index_width,
        )
        <= swa_cute._DSA_PACKED_WORKSPACE_BYTES
    )


def test_cute_rejects_non_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))
    with pytest.raises(RuntimeError, match="SM100 exclusively"):
        swa_cute._require_sm100(torch.device("cuda"))
