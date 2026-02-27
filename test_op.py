from operators.GEMM import GEMM
import tvm
import numpy as np
from RuntimeMod import RuntimeMod
from arch.RTX4090 import RTX4090
from Compute import Compute
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def build_wrapper(runtime_mod: RuntimeMod):
    runtime_mod.build()
    return runtime_mod.record


def test_gemm():
    arch = RTX4090()
    M, N, K = 512, 512, 512
    # gemm = GEMM(M, N, K, dtype="float32", name="GEMM")

    compute1 = Compute(0, arch, type_size=4)
    rtiles0_all = compute1.solve()
    compute = Compute(1, arch, type_size=4, rtiles0_all=rtiles0_all)
    compute_patterns = compute.solve()
    print(f"Find {len(compute_patterns)} compute patterns")
    print(compute_patterns)
    exit()

    runtime_mods = []
    for compute_pattern in compute_patterns:
        gemm = GEMM(M, N, K, dtype="float32", name="GEMM")
        runtime_mods.append(RuntimeMod(gemm, arch, compute_pattern))

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     executor.map(lambda runtime_mod: runtime_mod.build(), runtime_mods)

    for runtime_mod in runtime_mods:
        runtime_mod.build()
        runtime_mod.time_evaluator()
        print(runtime_mod.record)


test_gemm()
