from operators.GEMM import GEMM
import tvm
import numpy as np
from RuntimeMod import RuntimeMod
from arch.RTX4090 import RTX4090
from Compute import Compute
import argparse
from CLayer import CLayer
parser = argparse.ArgumentParser()
# M, N, K for matmul_expr, K is reduce axis.
parser.add_argument("--shape", nargs="*", type=int, default=[4096, 4096, 4096])
args = parser.parse_args()
M, N, K = args.shape
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def build_wrapper(runtime_mod: RuntimeMod):
    runtime_mod.build()
    return runtime_mod.record


def test_gemm():
    arch = RTX4090()
    # M, N, K = 1024, 768, 768
    # gemm = GEMM(M, N, K, dtype="float32", name="GEMM")

    compute1 = Compute(0, arch, type_size=4)
    rtiles0_all = compute1.solve()
    compute = Compute(1, arch, type_size=4, rtiles0_all=rtiles0_all)
    compute_patterns = compute.solve()
    print(f"Find {len(compute_patterns)} compute patterns")
    print(compute_patterns)
    # exit()

    runtime_mods = []
    for compute_pattern in compute_patterns:
        gemm = GEMM(M, N, K, dtype="float32", name="GEMM")
        runtime_mods.append(RuntimeMod(gemm, arch, compute_pattern))

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     executor.map(lambda runtime_mod: runtime_mod.build(), runtime_mods)

    res = []
    for runtime_mod in runtime_mods:
        try:
            runtime_mod.build()
            runtime_mod.time_evaluator()
            print(runtime_mod.record)
            res.append((runtime_mod.throughput, runtime_mod.record))
        except Exception as _:
            print(f"{runtime_mod.compute_pattern} is not a valid compute pattern")
            continue

    res.sort(key=lambda x: x[0], reverse=True)
    print(f'✅✅✅The best throughput config: runtime mod {res[0][1]},\n throughput: {res[0][0]} TFLOPS')

def test_CLayer():
    arch = RTX4090()

    # Register level
    clayer_reg = CLayer(0, arch)
    rtiles0 = clayer_reg.solve()
    print(f"Find {len(rtiles0)} rtiles0")
    print(rtiles0)

    clayer_shared = CLayer(1, arch, rtiles0=rtiles0)
    tile_patterns = clayer_shared.solve()
    print(f"Find {len(tile_patterns)} tile patterns")
    print(tile_patterns)


    AI = lambda x: (x[-1][0] * x[-1][1] * x[-1][2] * (K // x[-1][2])) / ((x[-1][0] * x[-1][2] + x[-1][1] * x[-1][2] + x[-1][0] * x[-1][1]) * (K // x[-1][2]) * 4)
    sorted_tile_patterns = sorted(tile_patterns, key=AI, reverse=True)
    print(f"The best tile pattern: {sorted_tile_patterns[0]} \n -----------------------\n")
    print(sorted_tile_patterns)

    compute_patterns = sorted_tile_patterns
    runtime_mods = []
    for compute_pattern in compute_patterns:
        gemm = GEMM(M, N, K, dtype="float32", name="GEMM")
        runtime_mods.append(RuntimeMod(gemm, arch, compute_pattern))

    res = []
    for runtime_mod in runtime_mods:
        try:
            runtime_mod.build()
            runtime_mod.time_evaluator()
            print(runtime_mod.record)
            res.append((runtime_mod.throughput, runtime_mod.record))
        except Exception as _:
            print(f"{runtime_mod.compute_pattern} is not a valid compute pattern")
            continue

    res.sort(key=lambda x: x[0], reverse=True)
    print(f'✅✅✅The best throughput config: runtime mod {res[0][1]},\n throughput: {res[0][0]} TFLOPS')
    exit()


test_gemm()
# test_CLayer()