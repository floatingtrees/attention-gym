import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK_DIRECTORY = (
    Path(__file__).parents[1] / "benchmarks" / "sparse" / "benchmark_heavily_compressed_attention"
)
BENCHMARK_SCRIPTS = (
    "benchmark_heavily_compressed_attention_cute.py",
    "benchmark_heavily_compressed_attention_cute_forward_backward.py",
)


@pytest.mark.parametrize("script_name", BENCHMARK_SCRIPTS)
def test_hca_benchmark_supports_direct_execution(script_name):
    result = subprocess.run(
        [sys.executable, str(BENCHMARK_DIRECTORY / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_forward_backward_benchmark_uses_canonical_hca_shape(monkeypatch):
    script = BENCHMARK_DIRECTORY / BENCHMARK_SCRIPTS[1]
    spec = importlib.util.spec_from_file_location("hca_forward_backward_benchmark", script)
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
        args.compression_rate,
        args.window,
        args.rope_dims,
    ) == (8, 64, 4096, 512, 128, 128, 64)
    assert module.DIFFERENTIABLE_INPUTS == tuple(range(8))
