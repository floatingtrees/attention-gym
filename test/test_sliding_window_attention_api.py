import importlib

import pytest
import torch


api = importlib.import_module("attn_gym.sparse.sliding_window_attention.api")
swa_package = importlib.import_module("attn_gym.sparse.sliding_window_attention")
sparse_package = importlib.import_module("attn_gym.sparse")


def make_arguments(*, share_kv=True):
    batch, heads, sequence, dim = 2, 4, 7, 8
    kv_heads = 1 if share_kv else heads
    return (
        torch.randn(batch, heads, sequence, dim),
        torch.randn(batch, kv_heads, sequence, dim),
        torch.randn(dim),
        torch.randn(heads),
        5,
        4,
        share_kv,
    )


def fail_loader():
    raise AssertionError("unexpected backend loader call")


def test_sliding_window_attention_is_publicly_exported():
    assert swa_package.sliding_window_attention is api.sliding_window_attention
    assert sparse_package.sliding_window_attention is api.sliding_window_attention
    assert "sliding_window_attention" in swa_package.__all__
    assert "sliding_window_attention" in sparse_package.__all__


@pytest.mark.parametrize(
    "backend_kwargs",
    [{}, {"backend": "eager"}, {"backend": "triton"}],
    ids=["default", "explicit-eager", "triton-fallback"],
)
def test_eager_dispatch(monkeypatch, backend_kwargs):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", lambda: implementation)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)

    assert api.sliding_window_attention(*arguments, **backend_kwargs) is expected
    assert calls == [arguments]


def test_cute_dispatch(monkeypatch):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", implementation)

    assert api.sliding_window_attention(*arguments, backend="cute") is expected
    assert calls == [arguments]


def test_cute_dispatch_reaches_registered_op_during_fullgraph_trace(monkeypatch):
    if api._cute_implementation is None:
        pytest.skip(f"CuTe backend is unavailable: {api._cute_initialization_error}")

    captured_graphs = []

    def capture_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return lambda *args: (torch.empty_like(args[0]),)

    def run(*args):
        return api.sliding_window_attention(*args, backend="cute")

    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_validate_cute_dependencies", fail_loader)
    torch._dynamo.reset()
    compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
    result = compiled(*make_arguments())

    assert result.shape == (2, 4, 7, 8)
    assert len(captured_graphs) == 1
    assert any(
        node.target
        is torch.ops.attention_gym._cute_sliding_window_attention_forward.default
        for node in captured_graphs[0].graph.nodes
    )


def test_cute_backend_rejects_autograd_inputs(monkeypatch):
    arguments = list(make_arguments())
    arguments[0].requires_grad_(True)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)

    with pytest.raises(RuntimeError, match="forward-only"):
        api.sliding_window_attention(*arguments, backend="cute")


@pytest.mark.parametrize("backend", ["eager", "triton", "cute"])
def test_shape_validation_precedes_backend_loading(monkeypatch, backend):
    arguments = list(make_arguments())
    arguments[2] = torch.randn(7)
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)

    with pytest.raises(ValueError, match="KV_norm_weight must have shape"):
        api.sliding_window_attention(*arguments, backend=backend)


@pytest.mark.parametrize(
    ("argument", "value", "error", "match"),
    [
        (4, -1, ValueError, "sliding_window_size must be non-negative"),
        (4, True, TypeError, "sliding_window_size must be a Python int"),
        (5, 3, ValueError, "rope_dims must be positive, even"),
        (5, 10, ValueError, "rope_dims must be positive, even"),
        (6, 1, TypeError, "share_kv must be a Python bool"),
    ],
)
def test_scalar_validation(monkeypatch, argument, value, error, match):
    arguments = list(make_arguments())
    arguments[argument] = value
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)

    with pytest.raises(error, match=match):
        api.sliding_window_attention(*arguments)


def test_dtype_validation_precedes_backend_loading(monkeypatch):
    arguments = list(make_arguments())
    arguments[3] = arguments[3].double()
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)

    with pytest.raises(TypeError, match="attention_sink must have dtype"):
        api.sliding_window_attention(*arguments)


def test_invalid_backend_and_mode_are_rejected(monkeypatch):
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)

    with pytest.raises(ValueError, match="backend"):
        api.sliding_window_attention(*make_arguments(), backend="cuda")
    with pytest.raises(ValueError, match="mode"):
        api.sliding_window_attention(*make_arguments(), mode="streaming")
    with pytest.raises(ValueError, match="Recurrent"):
        api.sliding_window_attention(*make_arguments(), mode="recurrent")


def test_cute_initialization_failure_is_deferred(monkeypatch):
    error = ValueError("broken optional backend")
    monkeypatch.setattr(api, "_cute_implementation", None)
    monkeypatch.setattr(api, "_cute_initialization_error", error)

    with pytest.raises(RuntimeError, match="broken optional backend"):
        api.sliding_window_attention(*make_arguments(), backend="cute")
