"""Public API and backend dispatch for sliding window attention."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from importlib import metadata
from typing import Literal

import torch

Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]

_CUTE_RUNTIME_DEPENDENCIES = (
    ("cuda-python", "cuda.bindings.driver", "13.3.1"),
    ("nvidia-cutlass-dsl", "cutlass.cute", "4.5.2"),
    ("flash-attn-4", "flash_attn.cute.interface", "4.0.0b17"),
    ("quack-kernels", "quack", "0.5.0"),
)

_cute_implementation: Callable[..., torch.Tensor] | None = None
_cute_initialization_error: Exception | None = None


def _validate_inputs(
    Q: object,
    KV: object,
    KV_norm_weight: object,
    attention_sink: object,
    sliding_window_size: object,
    rope_dims: object,
    share_kv: object,
) -> None:
    """Validate the backend-independent SWA contract."""
    tensors = (
        ("Q", Q),
        ("KV", KV),
        ("KV_norm_weight", KV_norm_weight),
        ("attention_sink", attention_sink),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    for name, value in (
        ("sliding_window_size", sliding_window_size),
        ("rope_dims", rope_dims),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int, got {type(value).__name__}.")
    if type(share_kv) is not bool:
        raise TypeError(f"share_kv must be a Python bool, got {type(share_kv).__name__}.")

    assert isinstance(Q, torch.Tensor)
    assert isinstance(KV, torch.Tensor)
    assert isinstance(KV_norm_weight, torch.Tensor)
    assert isinstance(attention_sink, torch.Tensor)
    assert isinstance(sliding_window_size, int)
    assert isinstance(rope_dims, int)
    assert isinstance(share_kv, bool)

    if Q.ndim != 4:
        raise ValueError("Q must have shape [batch, heads, sequence, head_dim].")
    batch, heads, sequence, head_dim = Q.shape
    if min(batch, heads, sequence, head_dim) <= 0:
        raise ValueError("Q dimensions must all be positive.")
    if not Q.is_floating_point():
        raise TypeError("Sliding window attention inputs must have a floating-point dtype.")
    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be non-negative.")
    if rope_dims <= 0 or rope_dims % 2 or rope_dims > head_dim:
        raise ValueError("rope_dims must be positive, even, and no larger than head_dim.")

    expected_kv_heads = (1, heads) if share_kv else (heads,)
    if (
        KV.ndim != 4
        or KV.shape[0] != batch
        or KV.shape[1] not in expected_kv_heads
        or KV.shape[2:] != (sequence, head_dim)
    ):
        expected_heads = "1 or heads" if share_kv else "heads"
        raise ValueError(f"KV must have shape [batch, {expected_heads}, sequence, head_dim].")
    if tuple(KV_norm_weight.shape) != (head_dim,):
        raise ValueError(
            f"KV_norm_weight must have shape {(head_dim,)}, got {tuple(KV_norm_weight.shape)}."
        )
    if tuple(attention_sink.shape) != (heads,):
        raise ValueError(
            f"attention_sink must have shape {(heads,)}, got {tuple(attention_sink.shape)}."
        )

    for name, tensor in tensors[1:]:
        assert isinstance(tensor, torch.Tensor)
        if tensor.device != Q.device:
            raise ValueError(f"{name} must be on {Q.device}, got {tensor.device}.")
        if tensor.dtype != Q.dtype:
            raise TypeError(f"{name} must have dtype {Q.dtype}, got {tensor.dtype}.")


def _load_eager_implementation() -> Callable[..., torch.Tensor]:
    from .reference import SWA

    return SWA


def _validate_cute_dependencies() -> None:
    """Validate the pinned CuTe stack during optional backend initialization."""
    problems = []
    for distribution, module, expected_version in _CUTE_RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(module.partition(".")[0])
            importlib.import_module(module)
        except (ImportError, RuntimeError) as error:
            problems.append(f"{distribution}=={expected_version} is unavailable ({error})")
            continue
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            problems.append(f"{distribution}=={expected_version} is not installed")
            continue
        if installed_version != expected_version:
            problems.append(
                f"{distribution}=={expected_version} is required; found {installed_version}"
            )
    if problems:
        raise RuntimeError(
            "The CuTe DSL backend dependency set is incompatible. "
            f"Install attn_gym[cute]. Details: {'; '.join(problems)}."
        )


def _load_cute_implementation() -> Callable[..., torch.Tensor]:
    _validate_cute_dependencies()
    module_name = f"{__package__}.cute"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module == module_name or missing_module.split(".")[0] in (
            "flash_attn",
            "cutlass",
            "cuda",
            "quack",
        ):
            raise RuntimeError(
                "The CuTe DSL backend for sliding window attention is unavailable; "
                "install nvidia-cutlass-dsl, flash-attn-4, and quack-kernels."
            ) from error
        raise
    return module.sliding_window_attention


def _initialize_cute_backend() -> None:
    """Resolve the optional backend before Dynamo traces public calls."""
    global _cute_implementation, _cute_initialization_error
    if _cute_implementation is not None or _cute_initialization_error is not None:
        return
    try:
        _cute_implementation = _load_cute_implementation()
    except Exception as error:
        _cute_initialization_error = error


_initialize_cute_backend()


def sliding_window_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    mode: Mode = "auto",
    backend: Backend = "eager",
) -> torch.Tensor:
    """Apply DeepSeek-V4 causal sliding-window attention.

    ``auto`` and ``chunked`` are prefill aliases. The optimized CuTe backend targets
    SM100 with BF16, D=512, and one physically shared KV head.
    """
    if backend not in ("eager", "triton", "cute"):
        raise ValueError(
            f"Unsupported sliding window attention backend {backend!r}; "
            "expected 'eager', 'triton', or 'cute'."
        )
    if mode not in ("auto", "chunked", "recurrent"):
        raise ValueError(
            f"Unsupported sliding window attention mode {mode!r}; "
            "expected 'auto', 'chunked', or 'recurrent'."
        )
    if mode == "recurrent":
        raise ValueError("Recurrent mode is currently unsupported.")
    if backend == "cute" and share_kv is False:
        raise ValueError("The CuTe backend requires share_kv=True.")

    _validate_inputs(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        sliding_window_size,
        rope_dims,
        share_kv,
    )

    if backend in ("eager", "triton"):
        implementation = _load_eager_implementation()
    else:
        if _cute_implementation is None:
            assert _cute_initialization_error is not None
            raise RuntimeError(str(_cute_initialization_error)) from _cute_initialization_error
        implementation = _cute_implementation

    return implementation(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["sliding_window_attention"]
