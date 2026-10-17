"""
Auto-tune the FlashAttention forward kernel and compare it with torch SDPA
(flash backend).

Usage (conda env `tvm`, TVM 0.27 with CUDA):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python test_flash_attn.py --shape 4 16 4096 128
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python test_flash_attn.py --sweep [--causal]
"""

# Compile workers are spawned processes that re-import this module, so only
# light imports live at module level: importing `Solver` initializes CUDA.
import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
import multiprocessing as mp

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs="*", type=int, default=[4, 16, 4096, 128],
                        help="B H S D (S_q = S_kv) or B H S_q S_kv D")
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="tune the standard FlashAttention-2 benchmark grid")
    parser.add_argument("--causal", action="store_true", default=False,
                        help="top-left aligned causal mask (S_q == S_kv)")
    parser.add_argument("--top_n", type=int, default=16,
                        help="number of ranked configs to compile and measure (0 = all)")
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1),
                        help="parallel compile workers")
    parser.add_argument("--finalists", type=int, default=3,
                        help="best configs re-timed against SDPA in the final round")
    parser.add_argument("--rounds", type=int, default=5,
                        help="interleaved timing rounds of the final comparison")
    return parser.parse_args()


def parse_shape(s):
    if len(s) == 4:
        B, H, S, D = s
        return (B, H, S, S, D)
    assert len(s) == 5, "--shape expects B H S D or B H S_q S_kv D"
    return tuple(s)


def build_all(shape, configs, softmax_scale, causal, arch_str, jobs):
    """Compiles configs in parallel; returns {cfg: path to exported library}."""
    from kernels.flash_attn_tir import build_flash_attn_to_file

    tmpdir = tempfile.mkdtemp(prefix="fa_tune_")
    path_of = {cfg: os.path.join(tmpdir, f"cfg{i}.so") for i, cfg in enumerate(configs)}
    paths, crashed = {}, []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
        futures = {
            pool.submit(build_flash_attn_to_file, shape, cfg, softmax_scale, arch_str, path_of[cfg], causal
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
            paths[cfg] = build_flash_attn_to_file(
                shape, cfg, softmax_scale, arch_str, path_of[cfg], causal
            )
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} failed to build: {str(e)[-200:]}", flush=True)
    return paths


def tune(args, shape, arch, arch_str, log):
    import torch
    import tvm
    from Solver import solve_flash_attn
    from cost_model.CostModel import rank_fa_configs
    from kernels.attn_bench import bench_interleaved, make_qkv, sdpa_flash
    from operators.FlashAttention import FlashAttention
    from RuntimeMod import RuntimeMod

    B, H, S_q, S_kv, D = shape
    start = time.time()
    op = FlashAttention(B, H, S_q, S_kv, D, causal=args.causal)

    configs = solve_flash_attn(arch, B, H, S_q, S_kv, D)
    ranked = rank_fa_configs(configs, arch, shape, causal=args.causal, top_n=args.top_n or None)
    print(f"Find {len(configs)} FlashAttention configs, measuring {len(ranked)}", flush=True)

    paths = build_all(shape, ranked, op.softmax_scale, args.causal, arch_str, args.jobs)
    compile_time = time.time() - start

    res = []
    for cfg in ranked:
        if cfg not in paths:
            continue
        rt = RuntimeMod(op, arch, True, cfg)
        try:
            rt.load_flash_attn(paths[cfg])
            rt.time_evaluator()
            print(rt.record, flush=True)
            res.append((rt.throughput, cfg, rt.execution_time, rt.max_error))
        except Exception as e:  # noqa: BLE001
            print(f"{cfg} is not a valid config: {str(e)[-200:]}", flush=True)
    end = time.time()
    if not res:
        raise RuntimeError(f"no valid FlashAttention config for {shape}")

    res.sort(key=lambda x: x[0], reverse=True)
    finalists = res[: args.finalists]
    err_of = {cfg: err for _, cfg, _, err in res}

    # Final numbers: the finalists and SDPA timed alternately on the same tensors,
    # so that clock drift and interference on a shared GPU hit all of them alike.
    q, k, v = make_qkv(B, H, S_q, S_kv, D)
    o = torch.empty_like(q)
    tvm_args = [tvm.runtime.from_dlpack(x) for x in (q, k, v, o)]
    fns = {"sdpa": lambda: sdpa_flash(q, k, v, causal=args.causal, scale=op.softmax_scale)}
    for _, cfg, _, _ in finalists:
        func = tvm.runtime.load_module(paths[cfg])["main"]
        fns[cfg] = lambda func=func: func(*tvm_args)
    times = bench_interleaved(fns, rounds=args.rounds)
    sdpa_t = times.pop("sdpa")
    best_cfg, best_t = min(times.items(), key=lambda x: x[1])
    best_err = err_of[best_cfg]
    best_tflops = op.num_flops / best_t / 1e12
    sdpa_tflops = op.num_flops / sdpa_t / 1e12
    ratio = best_tflops / sdpa_tflops
    rank_of_best = ranked.index(best_cfg) + 1

    msg = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{arch.name}]\n"
        f" DeCA FlashAttention: B = {B}, H = {H}, S_q = {S_q}, S_kv = {S_kv}, D = {D}, "
        f"causal = {args.causal}, "
        f"config: {best_cfg} (cost-model rank {rank_of_best}/{len(ranked)})\n"
        f" throughput: {best_tflops:.2f} TFLOPS, runtime: {best_t * 1e3:.4f} ms, max err: {best_err:.2e}\n"
        f" SDPA-flash: {sdpa_tflops:.2f} TFLOPS, runtime: {sdpa_t * 1e3:.4f} ms, ratio: {ratio:.3f}\n"
        f" compile time: {compile_time:.1f} s, total tuning time: {end - start:.1f} s\n"
    )
    print("✅✅✅ " + msg, flush=True)
    log.write(msg + "\n")
    log.flush()
    return best_cfg, best_tflops, sdpa_tflops


def main():
    import tvm
    from kernels.attn_bench import standard_shapes
    from src.utils.helper import GPUDetector

    args = parse_args()
    arch = GPUDetector.create_gpu_instance()
    major, minor = tvm.cuda(0).compute_version.split(".")
    arch_str = f"sm_{major}{minor}"

    shapes = [(B, H, S, S, D) for B, H, S, D in standard_shapes()] if args.sweep else [
        parse_shape(args.shape)
    ]
    log_path = os.path.join(ROOT, "deca_performance.log")
    summary = []
    with open(log_path, "a") as log:
        for shape in shapes:
            cfg, tflops, sdpa = tune(args, shape, arch, arch_str, log)
            summary.append((shape, cfg, tflops, sdpa))

    if len(summary) > 1:
        print(f"\nSummary on {arch.name}:")
        print(f"{'B':>3} {'H':>3} {'S':>6} {'D':>4} | {'DeCA':>8} {'SDPA':>8} {'ratio':>6} | config")
        for (B, H, S_q, _, D), cfg, tflops, sdpa in summary:
            print(f"{B:>3} {H:>3} {S_q:>6} {D:>4} | {tflops:>8.2f} {sdpa:>8.2f} {tflops / sdpa:>6.3f} | {cfg}")


if __name__ == "__main__":
    main()
