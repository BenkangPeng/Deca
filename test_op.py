from operators.GEMM import GEMM
import tvm
import numpy as np


def test_gemm():
    dtype = "float32"
    M, N, K = 2048, 4096, 1024
    gemm = GEMM(M, N, K)
    sch = gemm.apply([128, 128, 8], [16, 8, 1], [1, 1, 1])

    ctx = tvm.cuda(0)
    cuda_mod = tvm.build(sch.mod, target="cuda")
    cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(dtype), ctx)
    cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(dtype), ctx)
    cuda_c = tvm.nd.array(np.zeros((M, N)).astype(dtype), ctx)

    num_flops = M * N * K * 2
    num_runs = 6
    timer_cuda_mod = cuda_mod.time_evaluator(cuda_mod.entry_name, ctx, number=num_runs)
    t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean
    print(f"CUDA GEMM time: {t} seconds")
    print(f"CUDA GEMM throughput: {num_flops / t / 1e12} TFLOPS")



test_gemm()
