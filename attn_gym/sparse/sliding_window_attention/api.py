"""Public API for sliding window attention."""

from typing import Literal

import torch

Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]


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
    """Apply DeepSeek-V4 sliding window attention.

    ``mode`` and ``backend`` mirror the compressed sparse attention API. SWA currently
    has only an eager reference implementation, so every supported setting routes to it.
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

    from .reference import SWA

    return SWA(
        Q,
        KV,
        KV_norm_weight,
        attention_sink,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["sliding_window_attention"]
