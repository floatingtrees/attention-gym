"""Benchmark the SM100 CuTe SWA backend against the eager reference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import triton

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attn_gym.sparse.sliding_window_attention.api import sliding_window_attention


MAX_ABS_ERROR = 3e-2


def make_inputs(args: argparse.Namespace):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * scale
        )

    return (
        F.normalize(
            randn(args.batch, args.heads, args.sequence_length, args.head_dim),
            dim=-1,
        ),
        randn(args.batch, 1, args.sequence_length, args.head_dim),
        1.0 + randn(args.head_dim, scale=0.05),
        randn(args.heads),
        args.window,
        args.rope_dims,
        args.share_kv,
    )


def useful_flops(args: argparse.Namespace) -> int:
    """Return only useful causal-window QK and PV FLOPs."""
    pairs = sum(
        min(args.window, query_position + 1)
        for query_position in range(args.sequence_length)
    )
    return 4 * args.batch * args.heads * args.head_dim * pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")

    inputs = make_inputs(args)
    eager = lambda: sliding_window_attention(*inputs, backend="eager")
    optimized = lambda: sliding_window_attention(*inputs, backend="cute")

    with torch.inference_mode():
        expected = eager()
        actual = optimized()
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        max_abs_error = error.max().item()
        mean_abs_error = error.mean().item()
        if max_abs_error > MAX_ABS_ERROR:
            raise AssertionError(
                "CuTe output does not meet the reference tolerance; "
                f"max absolute error is {max_abs_error:.7g}, "
                f"required <= {MAX_ABS_ERROR:g}."
            )
        eager_ms = triton.testing.do_bench(
            eager,
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )
        cute_ms = triton.testing.do_bench(
            optimized,
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )

    flops = useful_flops(args)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} dtype={args.dtype}"
    )
    print(
        f"window={args.window} rope_dims={args.rope_dims} share_kv={args.share_kv}"
    )
    print(f"useful QK+PV FLOPs: {flops / 1e9:.6f} GF")
    print(f"error: max_abs={max_abs_error:.7g} mean_abs={mean_abs_error:.7g}")
    print(f"eager end-to-end: {eager_ms:.4f} ms")
    print(f"CuTe end-to-end:  {cute_ms:.4f} ms")
    print(f"speedup: {eager_ms / cute_ms:.2f}x")
    print(f"CuTe useful sparse throughput: {flops / (cute_ms * 1e9):.2f} TFLOP/s")


if __name__ == "__main__":
    main()
