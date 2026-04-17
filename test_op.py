from operators.GEMM import GEMM
import tvm
import numpy as np
from RuntimeMod import RuntimeMod
from arch.RTX4090 import RTX4090
from Compute import Compute
from CLayer import CLayer
import os
import time
from datetime import datetime


def build_wrapper(runtime_mod: RuntimeMod):
    runtime_mod.build()
    return runtime_mod.record


def test_gemm():
    start_time = time.time()    
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

    compute_patterns = sorted(compute_patterns, key=lambda x: (x[-1][0] * x[-1][1]) / (x[-1][0] + x[-1][1]), reverse=False)
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
    
    end_time = time.time()

    res.sort(key=lambda x: x[0], reverse=True)
    print(f'✅✅✅The best throughput config: runtime mod {res[0][1]},\n throughput: {res[0][0]} TFLOPS')

    log_path = os.path.join(os.path.dirname(__file__), "deca_performance.log")
    runtime = 2 * M * K * N / res[0][0] / 1e9 # unit: ms
    with open(log_path, "a") as f:
        f.write(f"DeCA GEMM: M = {M}, N = {N}, K = {K}, throughput: {res[0][0]} TFLOPS\n, runtime: {runtime} ms, compile time: {end_time - start_time} seconds\n\n")

def test_CLayer(M, N, K):
    start_time = time.time()
    arch = RTX4090()

    # Register level
    clayer_reg = CLayer(0, arch)
    rtiles0 = clayer_reg.solve()
    # print(f"Find {len(rtiles0)} rtiles0")
    # print(rtiles0)

    clayer_shared = CLayer(1, arch, rtiles0=rtiles0)
    tile_patterns = clayer_shared.solve()
    # print(f"Find {len(tile_patterns)} tile patterns")
    # print(tile_patterns)

    AI = lambda x: (x[-1][0] * x[-1][1] * x[-1][2] * (K // x[-1][2])) / ((x[-1][0] * x[-1][2] + x[-1][1] * x[-1][2] + x[-1][0] * x[-1][1]) * (K // x[-1][2]) * 4)
    sorted_tile_patterns = sorted(tile_patterns, key=AI, reverse=True)
    # print(f"The best tile pattern: {sorted_tile_patterns[0]} \n -----------------------\n")
    # print(sorted_tile_patterns)

    compute_patterns = sorted_tile_patterns[0:49]
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

    end_time = time.time()
    runtime = 2 * M * K * N / res[0][0] / 1e9 # unit: ms
    res.sort(key=lambda x: x[0], reverse=True)
    log_path = os.path.join(os.path.dirname(__file__), "deca_performance.log")
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n DeCA: M = {M}, N = {N}, K = {K}, throughput: {res[0][0]} TFLOPS\n, runtime: {runtime} ms, compile time: {end_time - start_time} seconds\n\n")

    print(f"✅✅✅ DeCA: M = {M}, N = {N}, K = {K}, throughput: {res[0][0]} TFLOPS\n, runtime: {runtime} ms, compile time: {end_time - start_time} seconds\n\n")


# test_gemm()
# test_CLayer()

if __name__ == "__main__":
    shapes = [
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 4096, 1024),
        (1024, 768, 768),
        (1024, 768, 2304),
        (1024, 768, 3072),
        (1024, 2304, 2304),
        (1024, 3072, 768),
        (1024, 2304, 768),
        (512,256,1024),
        (1024, 1024, 1024),
        (512, 4608, 1024),
        (1024, 512, 1024),
        (2048, 512, 1024),
        (512, 1024, 1024),
        (2048, 1024, 1024),
        (512, 2048, 1024)
    ]
    for shape in shapes:
        M, N, K = shape
        test_CLayer(M, N, K)
