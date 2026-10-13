"""
Shared shapes / references for the FP16 Tensor Core GEMM (C = A @ B).

A[M, K], B[K, N] and C[M, N] are row-major float16; accumulation is fp32 as in
cuBLAS (torch's default). Timing reuses the CUDA-event harness of the attention
benchmarks so that DeCA and torch are measured alike.
"""

import torch

from kernels.attn_bench import bench_cuda, bench_interleaved  # noqa: F401  (re-exported)

# Square GEMMs from 2K to 16K in steps of 2K.
SQUARE_SIZES = list(range(2048, 16384 + 1, 2048))


def standard_shapes():
    """Returns a list of (M, N, K)."""
    return [(n, n, n) for n in SQUARE_SIZES]


def gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def make_inputs(M: int, N: int, K: int, seed: int = 0):
    """Returns (a, b) with unit-variance fp16 entries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16, generator=g)
    return a, b


def torch_ref(inputs):
    """torch (cuBLAS) baseline, also the correctness reference."""
    a, b = inputs
    return a @ b


def bench_torch(M: int, N: int, K: int):
    """Returns (seconds, TFLOPS) of torch.matmul."""
    inputs = make_inputs(M, N, K)
    t = bench_cuda(lambda: torch_ref(inputs))
    return t, gemm_flops(M, N, K) / t / 1e12


class CublasHgemm:
    """`cublasHgemm` (fp16 inputs, fp16 accumulation) on torch tensors.

    torch only exposes fp32-accumulate fp16 GEMMs, so the fp16-accumulate
    baseline calls cuBLAS directly (ctypes) on torch's current stream.
    """

    _HALF_ONE, _HALF_ZERO = 0x3C00, 0x0000  # fp16 bit patterns of alpha / beta
    _OP_N = 0

    def __init__(self):
        import ctypes
        import os

        import nvidia.cublas

        lib_dir = os.path.join(os.path.dirname(nvidia.cublas.__file__), "lib")
        self._ct = ctypes
        self._lib = ctypes.CDLL(os.path.join(lib_dir, "libcublas.so.12"))
        self._handle = ctypes.c_void_p()
        assert self._lib.cublasCreate_v2(ctypes.byref(self._handle)) == 0
        self._alpha = ctypes.c_uint16(self._HALF_ONE)
        self._beta = ctypes.c_uint16(self._HALF_ZERO)

    def __call__(self, a, b, c):
        """c = a @ b for row-major a[M, K], b[K, N], c[M, N]."""
        ct = self._ct
        M, K = a.shape
        N = b.shape[1]
        stream = torch.cuda.current_stream().cuda_stream
        self._lib.cublasSetStream_v2(self._handle, ct.c_void_p(stream))
        # Row-major C = A B is column-major C^T = B^T A^T.
        status = self._lib.cublasHgemm(
            self._handle, self._OP_N, self._OP_N, N, M, K,
            ct.byref(self._alpha), ct.c_void_p(b.data_ptr()), N,
            ct.c_void_p(a.data_ptr()), K,
            ct.byref(self._beta), ct.c_void_p(c.data_ptr()), N,
        )
        assert status == 0, f"cublasHgemm failed with status {status}"
