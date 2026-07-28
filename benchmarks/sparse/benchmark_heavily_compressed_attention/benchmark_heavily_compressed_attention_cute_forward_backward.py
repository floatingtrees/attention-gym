"""Benchmark forward and backward wall time of the SM100 CuTe HCA backend.

The custom backward checkpoints attention activations and recomputes them during
the timed backward call.  Forward, backward (including recompute), and their sum
are therefore reported explicitly in milliseconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

if not __package__:
    benchmark_directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(benchmark_directory))
    sys.path.insert(0, str(benchmark_directory.parents[2]))

from attn_gym.sparse.heavily_compressed_attention.api import (
    heavily_compressed_attention,
)

if __package__:
    from .benchmark_heavily_compressed_attention_cute import (
        make_inputs,
        useful_flops,
    )
else:
    from benchmark_heavily_compressed_attention_cute import (
        make_inputs,
        useful_flops,
    )


DIFFERENTIABLE_INPUTS = tuple(range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--compression-rate", type=int, default=128)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=500, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")
    if args.heads != 64 or args.head_dim != 512:
        raise ValueError("The HCA CuTe specialization requires --heads=64 --head-dim=512.")

    raw_inputs = make_inputs(args)
    inputs = tuple(
        value.detach().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(raw_inputs)
    )
    gradient_targets = tuple(inputs[index] for index in DIFFERENTIABLE_INPUTS)

    def forward() -> torch.Tensor:
        return heavily_compressed_attention(*inputs, backend="cute")

    output = forward()
    if output.grad_fn is None:
        raise RuntimeError("The CuTe output has no autograd graph.")
    grad_output = torch.randn_like(output)

    def backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            output,
            gradient_targets,
            grad_outputs=grad_output,
            retain_graph=True,
        )

    backward()
    torch.cuda.synchronize()

    forward_ms = triton.testing.do_bench(
        forward,
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )
    backward_ms = triton.testing.do_bench(
        backward,
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )

    forward_attention_flops = useful_flops(args)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} dtype={args.dtype}"
    )
    print(
        f"sparsity: compression={args.compression_rate} window={args.window} "
        f"rope_dims={args.rope_dims} share_kv=True"
    )
    print(f"forward useful attention FLOPs: {forward_attention_flops / 1e9:.3f} GF")
    print(f"forward wall clock: {forward_ms:.4f} ms")
    print(f"backward wall clock (checkpoint recompute included): {backward_ms:.4f} ms")
    print(f"forward + backward wall clock: {forward_ms + backward_ms:.4f} ms")


if __name__ == "__main__":
    main()
