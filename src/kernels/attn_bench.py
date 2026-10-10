"""
Shared timing / reference helpers for attention kernels.

Both the TVM kernels and the torch SDPA baseline are timed with the same
torch CUDA-event harness on the same tensors so that the comparison is fair.
"""

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# FlashAttention-2 paper setup: hidden = 2048, batch * seqlen = 16k.
FA_SEQLENS = [512, 1024, 2048, 4096, 8192, 16384]
FA_HEADDIM_HEADS = [(64, 32), (128, 16)]


def standard_shapes():
    """Returns a list of (B, H, S, D)."""
    shapes = []
    for D, H in FA_HEADDIM_HEADS:
        for S in FA_SEQLENS:
            shapes.append((16384 // S, H, S, D))
    return shapes


def attn_flops(B: int, H: int, S_q: int, S_kv: int, D: int, causal: bool = False) -> float:
    flops = 4.0 * B * H * S_q * S_kv * D
    return flops / 2 if causal else flops


def bench_cuda(
    fn: Callable[[], None],
    warmup_s: float = 0.3,
    measure_s: float = 0.3,
    min_iters: int = 20,
) -> float:
    """Median time of `fn` in seconds, measured with CUDA events.

    Warmup and measurement are time based: short warmups leave the GPU at a
    lower clock and make the numbers noisy (notably on H100).
    """
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        fn()
        torch.cuda.synchronize()
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    fn()
    probe_end.record()
    probe_end.synchronize()
    per_call = max(probe_start.elapsed_time(probe_end) / 1e3, 1e-6)
    iters = max(min_iters, int(measure_s / per_call))
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    return times[len(times) // 2]


def make_qkv(B: int, H: int, S_q: int, S_kv: int, D: int, dtype=torch.float16, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, S_q, D, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(B, H, S_kv, D, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(B, H, S_kv, D, device="cuda", dtype=dtype, generator=g)
    return q, k, v


def sdpa_flash(q, k, v, causal: bool = False, scale: float | None = None):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


def bench_sdpa_flash(B, H, S_q, S_kv, D, causal=False, dtype=torch.float16):
    """Returns (seconds, TFLOPS) of torch SDPA restricted to the flash backend."""
    q, k, v = make_qkv(B, H, S_q, S_kv, D, dtype)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        t = bench_cuda(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
    return t, attn_flops(B, H, S_q, S_kv, D, causal) / t / 1e12


def bench_interleaved(fns: dict, rounds: int = 3) -> dict:
    """Times several callables alternately; returns the median per name (seconds)."""
    samples = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            samples[name].append(bench_cuda(fn))
    return {name: sorted(ts)[len(ts) // 2] for name, ts in samples.items()}


def default_scale(D: int) -> float:
    return 1.0 / math.sqrt(D)
