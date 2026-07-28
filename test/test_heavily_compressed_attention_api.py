import importlib

import pytest
import torch

api = importlib.import_module("attn_gym.sparse.heavily_compressed_attention.api")
hca_package = importlib.import_module("attn_gym.sparse.heavily_compressed_attention")
sparse_package = importlib.import_module("attn_gym.sparse")


def make_arguments(*, share_kv=False):
    batch, heads, sequence, dim, rate = 2, 3, 5, 8, 2
    kv_heads = 1 if share_kv else heads
    return (
        torch.randn(batch, heads, sequence, dim),
        torch.randn(batch, kv_heads, sequence, dim),
        torch.randn(batch, kv_heads, sequence, dim),
        torch.randn(batch, kv_heads, sequence, dim),
        torch.randn(rate, dim),
        torch.randn(dim),
        torch.randn(dim),
        torch.randn(heads),
        rate,
        5,
        4,
        share_kv,
    )


def fail_loader():
    raise AssertionError("unexpected backend loader call")


def test_heavily_compressed_attention_is_publicly_exported():
    assert hca_package.heavily_compressed_attention is api.heavily_compressed_attention
    assert sparse_package.heavily_compressed_attention is api.heavily_compressed_attention
    assert "heavily_compressed_attention" in hca_package.__all__
    assert "heavily_compressed_attention" in sparse_package.__all__


@pytest.mark.parametrize("backend_kwargs", [{}, {"backend": "eager"}])
def test_eager_dispatch(monkeypatch, backend_kwargs):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", lambda: implementation)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    result = api.heavily_compressed_attention(*arguments, **backend_kwargs)

    assert result is expected
    assert calls == [arguments]


def test_cute_dispatch(monkeypatch):
    arguments = make_arguments(share_kv=True)
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", implementation)

    result = api.heavily_compressed_attention(*arguments, backend="cute")
    assert result is expected
    assert calls == [arguments]


def test_cute_dispatch_reaches_registered_op_without_loading_during_trace(
    monkeypatch,
):
    if api._cute_implementation is None:
        pytest.skip(f"CuTe backend is unavailable: {api._cute_initialization_error}")

    captured_graphs = []

    def capture_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return lambda *args: (torch.empty_like(args[0]),)

    def run(*args):
        return api.heavily_compressed_attention(*args, backend="cute")

    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_validate_cute_dependencies", fail_loader)
    torch._dynamo.reset()
    compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
    result = compiled(*make_arguments(share_kv=True))

    assert result.shape == (2, 3, 5, 8)
    assert result.dtype == torch.float32
    assert len(captured_graphs) == 1
    assert any(
        node.target is torch.ops.attention_gym._cute_heavily_compressed_attention_forward.default
        for node in captured_graphs[0].graph.nodes
    )


def test_cute_autograd_reaches_registered_backward_during_aot_trace(monkeypatch):
    if api._cute_implementation is None:
        pytest.skip(f"CuTe backend is unavailable: {api._cute_initialization_error}")

    from functorch.compile import aot_function, make_boxed_func
    from torch._subclasses.fake_tensor import FakeTensorMode

    captured_backward_graphs = []

    def forward_compiler(graph_module, _example_inputs):
        return make_boxed_func(graph_module.forward)

    def backward_compiler(graph_module, _example_inputs):
        captured_backward_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    def run(*tensor_args):
        return api.heavily_compressed_attention(
            *tensor_args,
            2,
            5,
            4,
            True,
            backend="cute",
        )

    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_validate_cute_dependencies", fail_loader)
    compiled = aot_function(
        run,
        fw_compiler=forward_compiler,
        bw_compiler=backward_compiler,
    )
    with FakeTensorMode():
        tensor_args = tuple(
            tensor.requires_grad_() for tensor in make_arguments(share_kv=True)[:8]
        )
        compiled(*tensor_args).sum().backward()

    assert len(captured_backward_graphs) == 1
    assert any(
        node.target is torch.ops.attention_gym._cute_heavily_compressed_attention_backward.default
        for node in captured_backward_graphs[0].graph.nodes
    )


def test_triton_spelling_preserves_eager_fallback(monkeypatch):
    arguments = make_arguments()
    expected = object()
    monkeypatch.setattr(api, "_load_eager_implementation", lambda: lambda *_args: expected)

    assert api.heavily_compressed_attention(*arguments, backend="triton") is expected


def test_invalid_backend_is_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)
    with pytest.raises(ValueError, match="(?i)backend"):
        api.heavily_compressed_attention(*make_arguments(), backend="cuda")


def test_recurrent_mode_is_rejected():
    with pytest.raises(ValueError, match="(?i)recurrent"):
        api.heavily_compressed_attention(*make_arguments(), mode="recurrent")


@pytest.mark.parametrize("backend", ["eager", "triton", "cute"])
def test_shape_validation_precedes_backend_loading(monkeypatch, backend):
    arguments = list(make_arguments())
    arguments[4] = torch.randn(3, 8)
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)
    with pytest.raises(ValueError, match="B must have shape"):
        api.heavily_compressed_attention(*arguments, backend=backend)


def test_scalar_validation_rejects_bool_as_integer(monkeypatch):
    arguments = list(make_arguments())
    arguments[8] = True
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    with pytest.raises(TypeError, match="compression_rate must be a Python int"):
        api.heavily_compressed_attention(*arguments)


def test_shared_kv_contract_accepts_shared_or_expanded_heads(monkeypatch):
    arguments = list(make_arguments(share_kv=True))
    arguments[1] = arguments[1].expand(-1, 3, -1, -1)
    expected = object()
    monkeypatch.setattr(api, "_load_eager_implementation", lambda: lambda *_args: expected)
    assert api.heavily_compressed_attention(*arguments) is expected


def test_incompatible_cute_dependency_version_has_clear_error(monkeypatch):
    versions = {
        distribution: expected
        for distribution, _module, expected in api._CUTE_RUNTIME_DEPENDENCIES
    }
    versions["nvidia-cutlass-dsl"] = "0.0.0"
    monkeypatch.setattr(api.importlib, "import_module", lambda _module: object())
    monkeypatch.setattr(api.metadata, "version", versions.__getitem__)

    with pytest.raises(
        RuntimeError,
        match=r"nvidia-cutlass-dsl==4\.5\.2 is required; found 0\.0\.0",
    ):
        api._validate_cute_dependencies()
