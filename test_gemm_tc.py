"""
Auto-tune the FP16 Tensor Core GEMM (C = A @ B, fp32 accumulation) and compare
it with torch.matmul (cuBLAS).

Usage (conda env `tvm-build-venv`, PYTHONPATH from .env):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python test_gemm_tc.py --shape 4096 4096 4096
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python test_gemm_tc.py --sweep
"""

# Compile workers are spawned processes that re-import this module, so only
# light imports live at module level: importing `Solver` initializes CUDA.
import argparse
import csv
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[4096, 4096, 4096],
        metavar=("M", "N", "K"),
    )
    # For experiment to tune the square 2K-16K benchmark grid
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="tune the square 2K-16K benchmark grid",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=24,
        help="number of ranked configs to compile and measure (0 = all)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="parallel compile workers",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=3,
        help="best configs re-timed against torch in the final round",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="interleaved timing rounds of the final comparison",
    )
    return parser.parse_args(argv)


def build_all(shape, configs, arch_str, jobs):
    """Compiles configs in parallel; returns {cfg: path to exported library}."""
    from kernels.gemm_tir import build_gemm_tc_to_file

    tmpdir = tempfile.mkdtemp(prefix="gemm_tune_")
    path_of = {cfg: os.path.join(tmpdir, f"cfg{i}.so") for i, cfg in enumerate(configs)}
    paths, crashed = {}, []
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(build_gemm_tc_to_file, shape, cfg, arch_str, path_of[cfg]): cfg
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
            paths[cfg] = build_gemm_tc_to_file(shape, cfg, arch_str, path_of[cfg])
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} failed to build: {str(e)[-200:]}", flush=True)
    return paths


def tune(args, shape, arch, arch_str, log):
    import torch
    import tvm
    from Solver import solve_gemm_tc
    from cost_model.CostModel import rank_gemm_configs
    from kernels.gemm_bench import bench_interleaved, torch_ref
    from operators import GEMMTC
    from RuntimeMod import RuntimeMod

    M, N, K = shape
    start = time.time()
    op = GEMMTC(M, N, K)

    configs = solve_gemm_tc(arch, M, N, K)
    ranked = rank_gemm_configs(configs, arch, M, N, K, top_n=args.top_n or None)
    print(f"Find {len(configs)} {op.name} configs, measuring {len(ranked)}", flush=True)

    paths = build_all(shape, ranked, arch_str, args.jobs)
    compile_time = time.time() - start

    res = []
    for cfg in ranked:
        if cfg not in paths:
            continue
        rt = RuntimeMod(op, arch, True, cfg)
        try:
            rt.load_kernel(paths[cfg])
            rt.time_evaluator()
            print(rt.record, flush=True)
            res.append((rt.throughput, cfg, rt.execution_time, rt.max_error))
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} is not a valid config: {str(e)[-200:]}", flush=True)
        torch.cuda.empty_cache()
    end = time.time()
    if not res:
        raise RuntimeError(f"no valid {op.name} config for {shape}")

    res.sort(key=lambda x: x[0], reverse=True)
    finalists = res[: args.finalists]
    err_of = {cfg: err for _, cfg, _, err in res}

    # Final numbers: the finalists and torch timed alternately on the same tensors,
    # so that clock drift and interference on a shared GPU hit all of them alike.
    inputs = op.make_inputs()
    out = op.make_output(inputs)
    tvm_args = [tvm.nd.from_dlpack(x) for x in op.kernel_args(inputs, out)]
    fns = {"torch": lambda: torch_ref(inputs)}
    for _, cfg, _, _ in finalists:
        func = tvm.runtime.load_module(paths[cfg])["main"]
        fns[cfg] = lambda func=func: func(*tvm_args)
    times = bench_interleaved(fns, rounds=args.rounds)
    torch_t = times.pop("torch")
    best_cfg, best_t = min(times.items(), key=lambda x: x[1])
    best_tflops = op.num_flops / best_t / 1e12
    torch_tflops = op.num_flops / torch_t / 1e12
    ratio = torch_t / best_t
    peak = 100 * best_tflops * 1e3 / arch.peak_mma_sync_flops

    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{arch.name}]\n"
        f" DeCA {op.describe()}, config: {best_cfg} "
        f"(cost-model rank {ranked.index(best_cfg) + 1}/{len(ranked)})\n"
        f" throughput: {best_tflops:.1f} TFLOPS ({peak:.0f}% of fp32-acc TC peak), "
        f"runtime: {best_t * 1e3:.4f} ms, max err: {err_of[best_cfg]:.2e}\n"
        f" torch: {torch_tflops:.1f} TFLOPS, runtime: {torch_t * 1e3:.4f} ms, ratio: {ratio:.3f}\n"
        f" compile time: {compile_time:.1f} s, total tuning time: {end - start:.1f} s\n"
    )
    print("✅✅✅ " + msg, flush=True)
    log.write(msg + "\n")
    log.flush()
    return best_cfg, best_tflops, torch_tflops


def write_sweep_csv(summary, path):
    """Writes DeCA and torch TFLOPS for the 2K-16K sweep."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "DeCA_TFLOPS", "torch_TFLOPS", "ratio"])
        for (M, N, K), _, tflops, torch_tflops in summary:
            ratio = tflops / torch_tflops if torch_tflops else float("nan")
            writer.writerow(
                [
                    M,
                    N,
                    K,
                    f"{tflops:.4f}",
                    f"{torch_tflops:.4f}",
                    f"{ratio:.4f}",
                ]
            )


def main(argv=None):
    import tvm
    from kernels.gemm_bench import standard_shapes
    from src.utils.helper import GPUDetector

    args = parse_args(argv)
    arch = GPUDetector.create_gpu_instance()
    major, minor = tvm.cuda(0).compute_version.split(".")
    arch_str = f"sm_{major}{minor}"

    shapes = standard_shapes() if args.sweep else [tuple(args.shape)]
    log_path = os.path.join(ROOT, "deca_performance.log")
    summary = []
    with open(log_path, "a") as log:
        for shape in shapes:
            try:
                cfg, tflops, torch_tflops = tune(args, shape, arch, arch_str, log)
            except RuntimeError as e:
                # e.g. out of memory on a shared GPU; keep sweeping the other shapes
                if not args.sweep:
                    raise
                print(f"skipping {shape}: {e}", flush=True)
                continue
            summary.append((shape, cfg, tflops, torch_tflops))

    if len(summary) > 1:
        print(f"\nSummary of FP16 Tensor Core GEMM on {arch.name}:")
        print(
            f"{'M':>6} {'N':>6} {'K':>6} | {'DeCA':>6} {'torch':>6} {'ratio':>6} | config"
        )
        for (M, N, K), cfg, tflops, torch_tflops in summary:
            print(
                f"{M:>6} {N:>6} {K:>6} | {tflops:>6.1f} {torch_tflops:>6.1f} "
                f"{tflops / torch_tflops:>6.3f} | {cfg}"
            )

    if args.sweep:
        csv_path = os.path.join(ROOT, "gemm_tc_2K_to_16K.csv")
        write_sweep_csv(summary, csv_path)
        print(f"wrote {len(summary)} shapes to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
