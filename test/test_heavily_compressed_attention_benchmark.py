import subprocess
import sys
from pathlib import Path


def test_hca_benchmark_supports_direct_execution():
    script = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "sparse"
        / "benchmark_heavily_compressed_attention"
        / "benchmark_heavily_compressed_attention_cute.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
