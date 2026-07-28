"""Benchmark the full SM100 CuTe HCA operator against its eager reference."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import triton

from attn_gym.sparse.heavily_compressed_attention.api import (
    heavily_compressed_attention,
)


MAX_ABS_ERROR = 3e-2


def make_inputs(args: argparse.Namespace, *, batch: int | None = None):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(args.seed)
    batch = args.batch if batch is None else batch

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    return (
        F.normalize(
            randn(batch, args.heads, args.sequence_length, args.head_dim),
            dim=-1,
        ),
        randn(batch, 1, args.sequence_length, args.head_dim),
        randn(batch, 1, args.sequence_length, args.head_dim),
        randn(batch, 1, args.sequence_length, args.head_dim),
        randn(args.compression_rate, args.head_dim),
        1.0 + randn(args.head_dim, scale=0.05),
        1.0 + randn(args.head_dim, scale=0.05),
        randn(args.heads),
        args.compression_rate,
        args.window,
        args.rope_dims,
        True,
    )


def useful_pairs(args: argparse.Namespace) -> tuple[int, int]:
    """Count only reference-valid local and completed-compressed Q/KV pairs."""
    local = sum(
        min(args.window, query_position + 1)
        for query_position in range(args.sequence_length)
    )
    compressed = sum(
        (query_position + 1) // args.compression_rate
        for query_position in range(args.sequence_length)
    )
    return local, compressed


def useful_flops(args: argparse.Namespace) -> int:
    local, compressed = useful_pairs(args)
    return (
        4
        * args.batch
        * args.heads
        * args.head_dim
        * (local + compressed)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--compression-rate", type=int, default=128)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--correctness-batch",
        type=int,
        default=1,
        help="Batch used for the eager correctness check; zero disables it.",
    )
    parser.add_argument(
        "--minimum-tflops",
        type=float,
        default=500.0,
        help="Fail if end-to-end useful throughput is below this target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")
    if args.heads != 64 or args.head_dim != 512:
        raise ValueError("The HCA CuTe specialization requires --heads=64 --head-dim=512.")

    if args.correctness_batch:
        correctness_inputs = make_inputs(args, batch=args.correctness_batch)
        with torch.inference_mode():
            expected = heavily_compressed_attention(
                *correctness_inputs,
                backend="eager",
            )
            actual = heavily_compressed_attention(
                *correctness_inputs,
                backend="cute",
            )
            torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        max_abs_error = error.max().item()
        mean_abs_error = error.mean().item()
        if max_abs_error > MAX_ABS_ERROR:
            raise AssertionError(
                "CuTe HCA does not match the eager reference with "
                f"atol={MAX_ABS_ERROR:g}, rtol=0; max absolute error is "
                f"{max_abs_error:.7g}."
            )
        del correctness_inputs, expected, actual, error
    else:
        max_abs_error = float("nan")
        mean_abs_error = float("nan")

    inputs = make_inputs(args)
    optimized = lambda: heavily_compressed_attention(*inputs, backend="cute")
    with torch.inference_mode():
        optimized()
        torch.cuda.synchronize()
        cute_ms = triton.testing.do_bench(
            optimized,
            warmup=args.warmup,
            rep=args.rep,
            return_mode="median",
        )

    local_pairs, compressed_pairs = useful_pairs(args)
    flops = useful_flops(args)
    throughput = flops / (cute_ms * 1e9)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} dtype=bfloat16"
    )
    print(
        f"sparsity: compression={args.compression_rate} window={args.window} "
        "share_kv=True"
    )
    print(
        f"useful pairs per batch: local={local_pairs:,} "
        f"completed-compressed={compressed_pairs:,} "
        f"total={local_pairs + compressed_pairs:,}"
    )
    print(f"useful QK+PV FLOPs: {flops / 1e9:.6f} GF")
    print(f"error: max_abs={max_abs_error:.7g} mean_abs={mean_abs_error:.7g}")
    print(f"CuTe end-to-end: {cute_ms:.4f} ms")
    print(f"CuTe useful HCA throughput: {throughput:.2f} TFLOP/s")
    if throughput < args.minimum_tflops:
        raise AssertionError(
            f"Useful HCA throughput {throughput:.2f} TFLOP/s is below "
            f"the {args.minimum_tflops:.2f} TFLOP/s target."
        )


if __name__ == "__main__":
    main()
