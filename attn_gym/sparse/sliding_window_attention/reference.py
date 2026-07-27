"""Torch-only sliding window attention reference implementation."""

import math

import torch
import torch.nn.functional as F


def make_sliding_window_mask(query_length, window_size, device, dtype):
    query_positions = torch.arange(query_length, device=device)[:, None]
    key_positions = torch.arange(query_length, device=device)[None, :]
    valid = (
        (key_positions <= query_positions)
        & (key_positions >= query_positions - window_size + 1)
    )
    return torch.zeros(
        (query_length, query_length),
        device=device,
        dtype=dtype,
    ).masked_fill(~valid, float("-inf"))


def sink_softmax(x, sink, dim):
    sink = sink[None, :, None, None]
    maximums = torch.max(x, dim=dim, keepdim=True).values
    maximums = torch.maximum(maximums, sink)
    x = x - maximums
    sink = sink - maximums
    x = torch.exp(x)
    return x / (torch.sum(x, dim, keepdim=True) + torch.exp(sink))


def apply_rope(
    x: torch.Tensor,
    positions=None,
    base: float = 10_000.0,
    original_seq_len: int = 0,
    factor: float = 16.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    position_offset: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    sequence_length = x.shape[-2]
    rotary_dim = x.shape[-1]

    if positions is None:
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        positions = positions.to(device=x.device, dtype=torch.float32)

    frequencies = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                device=x.device,
                dtype=torch.float32,
            )
            / rotary_dim
        )
    )

    if original_seq_len > 0:

        def correction_dimension(num_rotations):
            return (
                rotary_dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dimension(beta_fast)), 0)
        high = min(math.ceil(correction_dimension(beta_slow)), rotary_dim - 1)
        if low == high:
            high += 0.001

        ramp = (
            torch.arange(
                rotary_dim // 2,
                device=x.device,
                dtype=torch.float32,
            )
            - low
        ) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth

    angles = torch.outer(positions, frequencies)
    frequencies_complex = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        frequencies_complex = frequencies_complex.conj()

    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], rotary_dim // 2, 2)
    )
    frequencies_complex = frequencies_complex.view(
        *([1] * (x.ndim - 2)),
        sequence_length,
        rotary_dim // 2,
    )
    rotated = torch.view_as_real(x_complex * frequencies_complex).flatten(-2)
    return rotated.to(x.dtype)


def SWA(
    Q,
    KV,
    KV_norm_weight,
    attention_sink,
    sliding_window_size,
    rope_dims: int,
    share_kv: bool,
):
    device = Q.device
    dtype = Q.dtype
    b, h, s, head_dim = Q.shape
    if share_kv:
        KV = KV.expand(-1, h, -1, -1)

    Q = torch.cat(
        [Q[:, :, :, :-rope_dims], apply_rope(Q[:, :, :, -rope_dims:])], dim=-1
    )
    KV = F.rms_norm(KV, (KV.shape[-1],), weight=KV_norm_weight)
    KV = torch.cat(
        [KV[:, :, :, :-rope_dims], apply_rope(KV[:, :, :, -rope_dims:])], dim=-1
    )

    SWA_mask = make_sliding_window_mask(
        s, sliding_window_size, device, dtype
    ).unsqueeze(0)
    SWA_mask = SWA_mask.expand(b, -1, -1)

    attention_kv = KV
    attention_mask = SWA_mask.unsqueeze(1)
    scale = head_dim**0.5

    P = sink_softmax(
        torch.matmul(Q, torch.permute(attention_kv, (0, 1, 3, 2))) / scale
        + attention_mask,
        attention_sink,
        dim=-1,
    )
    attn_output = P @ attention_kv
    return torch.cat(
        [
            attn_output[..., :-rope_dims],
            apply_rope(attn_output[..., -rope_dims:], inverse=True),
        ],
        dim=-1,
    )
