"""Public API and backend dispatch for heavily compressed attention."""

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
    tensors: tuple[tuple[str, object], ...],
    compression_rate: object,
    sliding_window_size: object,
    rope_dims: object,
    share_kv: object,
) -> None:
    """Validate the backend-independent HCA tensor contract."""
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    for name, value in {
        "compression_rate": compression_rate,
        "sliding_window_size": sliding_window_size,
        "rope_dims": rope_dims,
    }.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int, got {type(value).__name__}.")
    if type(share_kv) is not bool:
        raise TypeError(f"share_kv must be a Python bool, got {type(share_kv).__name__}.")

    by_name = dict(tensors)
    query = by_name["Q"]
    assert isinstance(query, torch.Tensor)
    if query.ndim != 4:
        raise ValueError("Q must have shape [batch, heads, sequence, head_dim].")
    batch, heads, sequence, dim = query.shape
    if min(batch, heads, sequence, dim) <= 0:
        raise ValueError("Q dimensions must all be positive.")
    if not query.is_floating_point():
        raise TypeError("Heavily compressed attention inputs must be floating point.")

    assert isinstance(compression_rate, int)
    assert isinstance(sliding_window_size, int)
    assert isinstance(rope_dims, int)
    assert isinstance(share_kv, bool)
    if compression_rate <= 0:
        raise ValueError("compression_rate must be positive.")
    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be non-negative.")
    if rope_dims <= 0 or rope_dims % 2 or rope_dims > dim:
        raise ValueError(
            "rope_dims must be positive, even, and no larger than head_dim."
        )

    for name, tensor in tensors:
        assert isinstance(tensor, torch.Tensor)
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {tensor.device}.")
        if tensor.dtype != query.dtype:
            raise TypeError(f"{name} must have dtype {query.dtype}, got {tensor.dtype}.")

    expected_kv_heads = (1, heads) if share_kv else (heads,)
    for name in ("KV", "C", "Z"):
        tensor = by_name[name]
        assert isinstance(tensor, torch.Tensor)
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[1] not in expected_kv_heads
            or tensor.shape[2:] != (sequence, dim)
        ):
            expected_heads = "1 or heads" if share_kv else "heads"
            raise ValueError(
                f"{name} must have shape "
                f"[batch, {expected_heads}, sequence, head_dim]."
            )

    expected_shapes = {
        "B": (compression_rate, dim),
        "KV_norm_weight": (dim,),
        "compressed_kv_norm_weight": (dim,),
        "attention_sink": (heads,),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = by_name[name]
        assert isinstance(tensor, torch.Tensor)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )


def _load_eager_implementation() -> Callable[..., torch.Tensor]:
    from .reference import HCA

    return HCA


def _load_triton_implementation() -> Callable[..., torch.Tensor]:
    # Preserve the original public contract: Triton is an accepted spelling but
    # falls back to the eager oracle until a dedicated implementation is added.
    return _load_eager_implementation()


def _validate_cute_dependencies() -> None:
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
    module_name = f"{__package__}.cute"
    _validate_cute_dependencies()
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
                "The CuTe backend for heavily compressed attention is not available; "
                "install nvidia-cutlass-dsl, flash-attn-4, and quack-kernels."
            ) from error
        raise
    return module.heavily_compressed_attention


def _initialize_cute_backend() -> None:
    global _cute_implementation, _cute_initialization_error
    if _cute_implementation is not None or _cute_initialization_error is not None:
        return
    try:
        _cute_implementation = _load_cute_implementation()
    except Exception as error:
        # Optional kernels must not make the base package unimportable.
        _cute_initialization_error = error


_initialize_cute_backend()


def heavily_compressed_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    C: torch.Tensor,
    Z: torch.Tensor,
    B: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    mode: Mode = "auto",
    backend: Backend = "eager",
) -> torch.Tensor:
    """Apply DeepSeek-V4 heavily compressed attention.

    Shapes use ``B=batch``, ``H=query heads``, ``S=sequence``, ``D=head dim``,
    and ``R=compression_rate``:

    - ``Q``: ``[B,H,S,D]``
    - ``KV``, ``C``, ``Z``: ``[B,H_KV,S,D]``, with ``H_KV=1`` for shared KV
    - compressor bias: ``[R,D]``
    - normalization weights: ``[D]``; learned sink: ``[H]``

    ``auto`` and ``chunked`` are prefill aliases. Recurrent decoding is not yet
    implemented. The SM100 CuTe specialization accepts BF16 shared KV with
    ``H=64`` and ``D=512``.
    """
    if backend not in ("eager", "triton", "cute"):
        raise ValueError(
            f"Unsupported heavily compressed attention backend {backend!r}; "
            "expected 'eager', 'triton', or 'cute'."
        )
    if mode not in ("auto", "chunked", "recurrent"):
        raise ValueError(
            f"Unsupported heavily compressed attention mode {mode!r}; "
            "expected 'auto', 'chunked', or 'recurrent'."
        )
    if mode == "recurrent":
        raise ValueError("Recurrent mode is currently unsupported.")

    tensors = (
        ("Q", Q),
        ("KV", KV),
        ("C", C),
        ("Z", Z),
        ("B", B),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    )
    _validate_inputs(
        tensors,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
    )

    if backend == "eager":
        implementation = _load_eager_implementation()
    elif backend == "triton":
        implementation = _load_triton_implementation()
    else:
        if _cute_implementation is None:
            assert _cute_initialization_error is not None
            raise RuntimeError(str(_cute_initialization_error)) from _cute_initialization_error
        implementation = _cute_implementation

    return implementation(
        Q,
        KV,
        C,
        Z,
        B,
        KV_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["heavily_compressed_attention"]
