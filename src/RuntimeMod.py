import tvm
import numpy as np

from arch.Arch import Arch
from arch.RTX4090 import RTX4090
from operators.Op import Operator
from operators.GEMM import GEMM

from tvm.runtime.module import Module


class RuntimeMod:
    def __init__(
        self,
        operator: Operator,
        arch: Arch,
        compute_pattern: list[tuple[int, int, int]],
    ):
        """
        Args:
            operator: Operator, kernel to be tuned
            arch: Arch, the architecture of the GPU
            compute_pattern: list[tuple[int, int, int]], the compute pattern
        """
        self.operator = operator
        self.arch = arch
        self.compute_pattern = compute_pattern

    def build(self):
        """
        Build the runtime module
        """
        self.sch: tvm.tir.Schedule = self.operator.apply(
            self.compute_pattern[1], self.compute_pattern[0], [1, 1, 1]
        )
        self.runtime_mod: Module = tvm.build(self.sch.mod, target="cuda")
        self.timer_mod = self.runtime_mod.time_evaluator(
            self.runtime_mod.entry_name, tvm.cuda(0)
        )

    def show(self):
        """
        Show the IRModule
        """
        self.sch.mod.show(black_format=False)
        assert hasattr(self, "runtime_mod"), "Runtime module not built"
        cuda = self.runtime_mod.imported_modules[0].get_source()
        print(cuda)

    def get_mod(self):
        """
        Get the IRModule
        """
        assert hasattr(self, "sch"), "Schedule not built"
        return self.sch.mod

    def get_sch(self):
        """
        Get the Schedule
        """
        assert hasattr(self, "sch"), "Schedule not built"
        return self.sch

    def time_evaluator(self):
        """
        Time the runtime module
        """
        M, N, K = self.operator.M, self.operator.N, self.operator.K
        num_flops = self.operator.num_flops
        dtype = self.operator.dtype

        ctx = tvm.cuda(0)
        cuda_a = tvm.nd.array(np.arange(M * K).reshape((M, K)).astype(dtype), ctx)
        cuda_b = tvm.nd.array(np.arange(K * N).reshape((K, N)).astype(dtype), ctx)
        cuda_c = tvm.nd.array(np.zeros((M, N)).astype(dtype), ctx)

        t = self.timer_mod(cuda_a, cuda_b, cuda_c).mean
        self.execution_time = t
        self.throughput = num_flops / t / 1e12  # unit: TFLOPS
        self.record = f"{self.operator.name}({M}, {N}, {K}, {dtype}) \n\t{self.compute_pattern} \n\tcost time: {t * 1000} ms, throughput: {self.throughput} TFLOPS"
