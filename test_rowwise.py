"""
Auto-tune the row-wise Softmax / RMSNorm forward kernels and compare them with
eager torch (`torch.softmax`, `F.rms_norm`).

Usage (conda env `tvm`, TVM 0.27 with CUDA):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python test_rowwise.py --op softmax --shape 16384 4096
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python test_rowwise.py --op rmsnorm --dtype bf16 --sweep
"""

# Compile workers are spawned processes that re-import this module, so only
# light imports live at module level: importing `Solver` initializes CUDA.
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["softmax", "rmsnorm"], default="softmax")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--shape", nargs=2, type=int, default=[16384, 4096], metavar=("M", "N"))
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="tune the standard row-wise benchmark grid")
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument("--top_n", type=int, default=24,
                        help="number of ranked configs to compile and measure (0 = all)")
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1),
                        help="parallel compile workers")
    parser.add_argument("--finalists", type=int, default=3,
                        help="best configs re-timed against torch in the final round")
    parser.add_argument("--rounds", type=int, default=5,
                        help="interleaved timing rounds of the final comparison")
    return parser.parse_args()


def build_all(op, shape, dtype, eps, configs, arch_str, jobs):
    """Compiles configs in parallel; returns {cfg: path to exported library}."""
    from kernels.rowwise_tir import build_rowwise_to_file

    tmpdir = tempfile.mkdtemp(prefix="rw_tune_")
    path_of = {cfg: os.path.join(tmpdir, f"cfg{i}.so") for i, cfg in enumerate(configs)}
    paths, crashed = {}, []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = {
            pool.submit(
                build_rowwise_to_file, op, shape, dtype, cfg, eps, arch_str, path_of[cfg]
            ): cfg
            for cfg in configs
        }
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                paths[cfg] = fut.result()
            except BrokenProcessPool:
                crashed.append(cfg)
            except Exception as e:  # noqa: BLE001
                print(f"{cfg} failed to build: {str(e)[-200:]}", flush=True)
    # A dead worker breaks the whole pool; rebuild the affected configs serially.
    for cfg in crashed:
        try:
            paths[cfg] = build_rowwise_to_file(op, shape, dtype, cfg, eps, arch_str, path_of[cfg])
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} failed to build: {str(e)[-200:]}", flush=True)
    return paths


def tune(args, shape, arch, arch_str, log):
    import torch
    import tvm
    from Solver import solve_rowwise
    from cost_model.CostModel import rank_rowwise_configs
    from kernels.rowwise_bench import bench_interleaved
    from operators import RMSNorm, Softmax
    from RuntimeMod import RuntimeMod

    M, N = shape
    start = time.time()
    op = Softmax(M, N, args.dtype) if args.op == "softmax" else RMSNorm(M, N, args.dtype, args.eps)

    configs = solve_rowwise(arch, args.op, M, N)
    ranked = rank_rowwise_configs(configs, arch, args.op, M, N, top_n=args.top_n or None)
    print(f"Find {len(configs)} {op.name} configs, measuring {len(ranked)}", flush=True)

    paths = build_all(args.op, shape, op.dtype, op.eps, ranked, arch_str, args.jobs)
    compile_time = time.time() - start

    res = []
    for cfg in ranked:
        if cfg not in paths:
            continue
        rt = RuntimeMod(op, arch, False, cfg)
        try:
            rt.load_kernel(paths[cfg])
            rt.time_evaluator()
            print(rt.record, flush=True)
            res.append((rt.throughput, cfg, rt.execution_time, rt.max_error))
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} is not a valid config: {str(e)[-200:]}", flush=True)
    end = time.time()
    if not res:
        raise RuntimeError(f"no valid {op.name} config for {shape}")

    res.sort(key=lambda x: x[0], reverse=True)
    finalists = res[: args.finalists]
    err_of = {cfg: err for _, cfg, _, err in res}

    # Final numbers: the finalists and torch timed alternately on the same tensors,
    # so that clock drift and interference on a shared GPU hit all of them alike.
    inputs = op.make_inputs()
    out = torch.empty_like(inputs[0])
    tvm_args = [tvm.runtime.from_dlpack(x) for x in op.kernel_args(inputs, out)]
    fns = {"torch": lambda: op.reference(inputs)}
    for _, cfg, _, _ in finalists:
        func = tvm.runtime.load_module(paths[cfg])["main"]
        fns[cfg] = lambda func=func: func(*tvm_args)
    times = bench_interleaved(fns, rounds=args.rounds)
    torch_t = times.pop("torch")
    best_cfg, best_t = min(times.items(), key=lambda x: x[1])
    best_gbps = op.num_bytes / best_t / 1e9
    torch_gbps = op.num_bytes / torch_t / 1e9
    ratio = torch_t / best_t
    peak = 100 * best_gbps / arch.dram_bw

    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{arch.name}]\n"
        f" DeCA {op.describe()}, config: {best_cfg} "
        f"(cost-model rank {ranked.index(best_cfg) + 1}/{len(ranked)})\n"
        f" throughput: {best_gbps:.1f} GB/s ({peak:.0f}% of DRAM peak), "
        f"runtime: {best_t * 1e3:.4f} ms, max err: {err_of[best_cfg]:.2e}\n"
        f" torch: {torch_gbps:.1f} GB/s, runtime: {torch_t * 1e3:.4f} ms, ratio: {ratio:.3f}\n"
        f" compile time: {compile_time:.1f} s, total tuning time: {end - start:.1f} s\n"
    )
    print("✅✅✅ " + msg, flush=True)
    log.write(msg + "\n")
    log.flush()
    return best_cfg, best_gbps, torch_gbps


def main():
    import tvm
    from kernels.rowwise_bench import standard_shapes
    from src.utils.helper import GPUDetector

    args = parse_args()
    arch = GPUDetector.create_gpu_instance()
    major, minor = tvm.cuda(0).compute_version.split(".")
    arch_str = f"sm_{major}{minor}"

    shapes = standard_shapes(args.op) if args.sweep else [tuple(args.shape)]
    log_path = os.path.join(ROOT, "deca_performance.log")
    summary = []
    with open(log_path, "a") as log:
        for shape in shapes:
            try:
                cfg, gbps, torch_gbps = tune(args, shape, arch, arch_str, log)
            except RuntimeError as e:
                # e.g. out of memory on a shared GPU; keep sweeping the other shapes
                if not args.sweep:
                    raise
                print(f"skipping {shape}: {e}", flush=True)
                continue
            summary.append((shape, cfg, gbps, torch_gbps))

    if len(summary) > 1:
        print(f"\nSummary of {args.op} ({args.dtype}) on {arch.name}:")
        print(f"{'M':>6} {'N':>7} | {'DeCA':>7} {'torch':>7} {'%peak':>5} {'ratio':>6} | config")
        for (M, N), cfg, gbps, torch_gbps in summary:
            print(
                f"{M:>6} {N:>7} | {gbps:>7.0f} {torch_gbps:>7.0f} "
                f"{100 * gbps / arch.dram_bw:>5.0f} {gbps / torch_gbps:>6.3f} | {cfg}"
            )


if __name__ == "__main__":
    main()
