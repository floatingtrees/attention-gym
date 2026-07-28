import argparse

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.sliding_window_attention.api import sliding_window_attention


pytest.importorskip("flash_attn.cute.interface")

MAX_ABS_ERROR = 3e-2


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
    configuration = dict(
        batch=1,
        heads=64,
        sequence_length=256,
        head_dim=512,
        window=128,
        rope_dims=64,
        seed=123,
    )
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
            dict(batch=2, heads=128, sequence_length=128, window=65),
            id="batch2-h128-partial-final-block",
        ),
        pytest.param(
            dict(sequence_length=129, window=999, rope_dims=512),
            id="clamped-window-full-rope",
        ),
        pytest.param(dict(sequence_length=65, window=1), id="self-only"),
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
        (lambda values: values[:-1], "H to be a multiple of 64"),
        (
            lambda values: values[:1]
            + (values[1].expand(-1, 64, -1, -1),)
            + values[2:],
            "one shared head",
        ),
        (lambda values: values[:6] + (False,), "share_kv=True"),
    ],
)
def test_cute_rejects_unsupported_configuration(transform, match):
    inputs = _inputs(sequence_length=17)
    if match == "H to be a multiple of 64":
        query = inputs[0][:, :-1].contiguous()
        sink = inputs[3][:-1].contiguous()
        transformed = (query, inputs[1], inputs[2], sink, *inputs[4:])
    else:
        transformed = transform(inputs)

    with pytest.raises(ValueError, match=match):
        sliding_window_attention(*transformed, backend="cute")
