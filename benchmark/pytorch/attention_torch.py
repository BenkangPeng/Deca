"""
Baseline: torch SDPA restricted to the FlashAttention backend.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python benchmark/pytorch/attention_torch.py [--shape B H S D] [--causal]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from kernels.attn_bench import bench_sdpa_flash, standard_shapes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--shape", nargs=4, type=int, default=None, metavar=("B", "H", "S", "D"))
parser.add_argument("--causal", action="store_true", default=False)
args = parser.parse_args()

shapes = [tuple(args.shape)] if args.shape else standard_shapes()
gpu = torch.cuda.get_device_name(0)
log_path = os.path.join(os.path.dirname(__file__), "attention_torch_performance.log")

with open(log_path, "a") as f:
    for B, H, S, D in shapes:
        t, tflops = bench_sdpa_flash(B, H, S, S, D, causal=args.causal)
        line = (
            f"[{gpu}] SDPA-flash B={B}, H={H}, S={S}, D={D}, causal={args.causal}: "
            f"{t * 1e3:.3f} ms, {tflops:.2f} TFLOPS"
        )
        print(line, flush=True)
        f.write(line + "\n")
    f.write("-" * 30 + "\n")
