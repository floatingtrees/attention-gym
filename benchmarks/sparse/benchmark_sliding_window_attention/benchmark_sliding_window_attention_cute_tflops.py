"""Canonical DV4 end-to-end useful-TFLOP/s benchmark for the SM100 CuTe SWA backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import triton

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attn_gym.sparse.sliding_window_attention.api import sliding_window_attention

if __package__:
    from .benchmark_sliding_window_attention_cute import make_inputs, useful_flops
else:
    from benchmark_sliding_window_attention_cute import make_inputs, useful_flops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--minimum-tflops", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")

    inputs = make_inputs(args)
    optimized = lambda: sliding_window_attention(*inputs, backend="cute")
    with torch.inference_mode():
        optimized()
        torch.cuda.synchronize()
        cute_ms = triton.testing.do_bench(
            optimized,
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )

    pairs = sum(
        min(args.window, query_position + 1)
        for query_position in range(args.sequence_length)
    )
    flops = useful_flops(args)
    useful_tflops = flops / (cute_ms * 1e9)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} dtype={args.dtype}"
    )
    print(
        f"window={args.window} rope_dims={args.rope_dims} share_kv={args.share_kv}"
    )
    print(f"useful causal-window pairs per batch/head: {pairs:,}")
    print(f"useful QK+PV FLOPs: {flops / 1e9:.6f} GF")
    print(f"CuTe end-to-end: {cute_ms:.4f} ms")
    print(f"CuTe useful sparse throughput: {useful_tflops:.2f} TFLOP/s")
    if useful_tflops < args.minimum_tflops:
        raise AssertionError(
            f"CuTe useful throughput is {useful_tflops:.2f} TFLOP/s; "
            f"required >= {args.minimum_tflops:.2f} TFLOP/s."
        )


if __name__ == "__main__":
    main()
