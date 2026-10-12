"""
Baseline: eager torch.softmax / F.rms_norm over the row-wise shape grid.

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python benchmark/pytorch/rowwise_torch.py --op softmax --dtype fp16 [--shape M N]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from kernels.rowwise_bench import ROWWISE_OPS, bench_torch, standard_shapes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--op", choices=ROWWISE_OPS, default="softmax")
parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
parser.add_argument("--shape", nargs=2, type=int, default=None, metavar=("M", "N"))
args = parser.parse_args()

shapes = [tuple(args.shape)] if args.shape else standard_shapes(args.op)
gpu = torch.cuda.get_device_name(0)
log_path = os.path.join(os.path.dirname(__file__), "rowwise_torch_performance.log")

with open(log_path, "a") as f:
    for M, N in shapes:
        t, gbps = bench_torch(args.op, M, N, args.dtype)
        line = (
            f"[{gpu}] torch {args.op} {args.dtype} M={M}, N={N}: "
            f"{t * 1e3:.4f} ms, {gbps:.1f} GB/s"
        )
        print(line, flush=True)
        f.write(line + "\n")
    f.write("-" * 30 + "\n")
