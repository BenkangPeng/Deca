"""
FP16 Tensor Core GEMM operator (C = A @ B, fp32 or fp16 accumulation, fp16 output).

Like FlashAttention, a tile configuration (`GemmConfig`) fully determines a
low-level TIR kernel produced by `kernels.gemm_tir.make_gemm_tc`; the operator
also provides the inputs, the reference and the tolerance used by RuntimeMod.
The schedule-primitive `GEMM` operator keeps serving the CUDA-core and legacy
wmma paths.
"""

import math

import tvm

from operators.Op import Operator
from kernels.gemm_tir import GemmConfig, check_config, make_gemm_tc


class GEMMTC(Operator):
    # RuntimeMod reports compute-bound kernels in TFLOPS.
    metric = "TFLOPS"

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        dtype: str = "float16",
        acc: str = "fp32",
        acc_splits: int = 1,
        name: str = "GEMMTC",
    ):
        """
        Args:
            acc: accumulation precision, "fp32" (as torch) or "fp16" (as cublasHgemm)
            acc_splits: number of alternating fp16 accumulators (fp16 only)
        """
        assert dtype == "float16", "GEMMTC only supports float16 inputs"
        assert acc in ("fp32", "fp16")
        self.M, self.N, self.K = M, N, K
        self.dtype = dtype
        self.acc = acc
        self.acc_splits = acc_splits if acc == "fp16" else 1
        self.num_flops = 2 * M * N * K
        super().__init__(name=name)

    @property
    def shape(self):
        return (self.M, self.N, self.K)

    @property
    def tolerance(self):
        """(atol, rtol) vs the fp32 reference.

        fp32 accumulation: fp16 output rounding is ~5e-4 relative, the
        accumulation error grows like sqrt(K).
        fp16 accumulation: every k16 MMA rounds a partial sum of magnitude
        ~sqrt(k), so the error grows like K / sqrt(acc_splits); cublasHgemm and
        the kernel reach a max error of ~7e-4 * K at acc_splits = 1.
        """
        if self.acc == "fp16":
            return (2e-3 * self.K / math.sqrt(self.acc_splits), 1e-2)
        return (1e-2 * math.sqrt(self.K / 1024), 2e-3)

    def is_valid(self, cfg: GemmConfig) -> bool:
        if cfg.acc_f16 != (self.acc == "fp16"):
            return False
        try:
            check_config(cfg, self.M, self.N, self.K)
            return True
        except AssertionError:
            return False

    def build_mod(self, cfg: GemmConfig) -> tvm.IRModule:
        return make_gemm_tc(self.M, self.N, self.K, cfg)

    def compile(self, mod: tvm.IRModule):
        from kernels.gemm_tir import build_gemm_tc

        return build_gemm_tc(mod)

    def make_inputs(self):
        from kernels.gemm_bench import make_inputs

        return make_inputs(self.M, self.N, self.K)

    def make_output(self, inputs):
        import torch

        return torch.empty(self.M, self.N, device="cuda", dtype=torch.float16)

    def reference(self, inputs):
        """fp32 GEMM of the fp16 inputs (torch's own fp16 GEMM may use
        reduced-precision reductions, so it is not used as the reference)."""
        import torch

        a, b = inputs
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return a.float() @ b.float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def kernel_args(self, inputs, out):
        """Tensors in the kernel's argument order: a, b, c."""
        return (*inputs, out)

    def describe(self) -> str:
        acc = f", acc=fp16x{self.acc_splits}" if self.acc == "fp16" else ""
        return f"{self.name}(M={self.M}, N={self.N}, K={self.K}, {self.dtype}{acc})"
