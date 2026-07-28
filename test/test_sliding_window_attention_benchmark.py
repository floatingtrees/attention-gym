import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK_DIRECTORY = (
    Path(__file__).parents[1] / "benchmarks" / "sparse" / "benchmark_sliding_window_attention"
)
BENCHMARK_SCRIPTS = (
    "benchmark_sliding_window_attention_cute.py",
    "benchmark_sliding_window_attention_cute_forward_backward.py",
    "benchmark_sliding_window_attention_cute_tflops.py",
)


@pytest.mark.parametrize("script_name", BENCHMARK_SCRIPTS)
def test_benchmark_supports_direct_execution(script_name):
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_DIRECTORY / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_useful_flops_counts_only_valid_causal_window_pairs():
    script = BENCHMARK_DIRECTORY / BENCHMARK_SCRIPTS[0]
    spec = importlib.util.spec_from_file_location("swa_benchmark", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        batch=2,
        heads=3,
        sequence_length=5,
        head_dim=7,
        window=3,
    )

    expected_pairs = 1 + 2 + 3 + 3 + 3
    assert module.useful_flops(args) == 4 * 2 * 3 * 7 * expected_pairs


def test_useful_flops_matches_canonical_dv4_shape():
    script = BENCHMARK_DIRECTORY / BENCHMARK_SCRIPTS[0]
    spec = importlib.util.spec_from_file_location("swa_benchmark_dv4", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        batch=8,
        heads=64,
        sequence_length=4096,
        head_dim=512,
        window=128,
    )

    assert sum(min(128, position + 1) for position in range(4096)) == 516_160
    assert module.useful_flops(args) == 541_232_988_160


def test_forward_backward_benchmark_uses_canonical_swa_shape(monkeypatch):
    script = BENCHMARK_DIRECTORY / BENCHMARK_SCRIPTS[1]
    spec = importlib.util.spec_from_file_location("swa_forward_backward_benchmark", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script)])

    args = module.parse_args()

    assert (
        args.batch,
        args.heads,
        args.sequence_length,
        args.head_dim,
        args.window,
        args.rope_dims,
    ) == (1, 64, 4096, 512, 128, 64)
    assert args.share_kv is True
    assert module.DIFFERENTIABLE_INPUTS == tuple(range(4))
