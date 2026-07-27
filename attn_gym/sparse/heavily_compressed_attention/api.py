"""Public API for heavily compressed attention."""

from typing import Literal

import torch

Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]


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

    ``mode`` and ``backend`` mirror the compressed sparse attention API. HCA currently
    has only an eager reference implementation, so every supported setting routes to it.
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

    from .reference import HCA

    return HCA(
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
