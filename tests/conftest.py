"""Allow tests to import Deca's existing top-level src modules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


import pytest


@pytest.fixture(autouse=True)
def short_gpu_timing(monkeypatch):
    # Correctness tests exercise real CUDA-event timing without long benchmark
    # warmups or statistical sampling. The CLI benchmarks retain their defaults.
    from kernels import attn_bench
    bench = attn_bench.bench_cuda
    monkeypatch.setattr(attn_bench, "bench_cuda", lambda fn: bench(
        fn, warmup_s=0.002, measure_s=0.002, min_iters=3
    ))
