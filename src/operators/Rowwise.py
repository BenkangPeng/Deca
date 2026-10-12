"""
Row-wise memory-bound operators over an [M, N] input (reduction over N).

Like FlashAttention, a config (`RowConfig`) fully determines a low-level TIR
kernel produced by `kernels.rowwise_tir.make_rowwise`; the operator also
provides the inputs, the torch reference and the tolerance used by RuntimeMod.
"""

import tvm

from operators.Op import Operator
from kernels.rowwise_tir import RowConfig, check_config, make_rowwise

# (atol, rtol) of the comparison with torch: |y - ref| <= atol + rtol * |ref|
TOLERANCE = {
    ("softmax", "float16"): (1e-5, 1e-2),
    ("softmax", "bfloat16"): (1e-5, 3e-2),
    ("rmsnorm", "float16"): (1e-2, 1e-2),
    ("rmsnorm", "bfloat16"): (3e-2, 3e-2),
}


class RowwiseOperator(Operator):
    op_kind: str = ""

    def __init__(self, M: int, N: int, dtype: str = "float16", eps: float = 1e-6, name: str = ""):
        from kernels.rowwise_bench import canonical_dtype, rowwise_bytes

        self.M, self.N = M, N
        self.dtype = canonical_dtype(dtype)
        self.eps = eps
        self.num_bytes = rowwise_bytes(self.op_kind, M, N)
        super().__init__(name=name or type(self).__name__)

    @property
    def shape(self):
        return (self.M, self.N)

    @property
    def tolerance(self):
        return TOLERANCE[(self.op_kind, self.dtype)]

    def is_valid(self, cfg: RowConfig) -> bool:
        try:
            check_config(cfg, self.op_kind, self.M, self.N, self.dtype)
            return True
        except AssertionError:
            return False

    def build_mod(self, cfg: RowConfig) -> tvm.IRModule:
        return make_rowwise(self.op_kind, self.M, self.N, self.dtype, cfg, self.eps)

    def compile(self, mod: tvm.IRModule):
        from kernels.rowwise_tir import build_rowwise

        return build_rowwise(mod)

    def make_inputs(self):
        from kernels.rowwise_bench import make_inputs

        return make_inputs(self.op_kind, self.M, self.N, self.dtype)

    def reference(self, inputs):
        from kernels.rowwise_bench import torch_ref

        return torch_ref(self.op_kind, inputs, self.eps)

    def kernel_args(self, inputs, out):
        """Tensors in the kernel's argument order: x[, w], y."""
        return (*inputs, out)

    def describe(self) -> str:
        return f"{self.name}(M={self.M}, N={self.N}, {self.dtype})"
