import tvm
import numpy as np

from arch.Arch import Arch
from arch.RTX4090 import RTX4090
from operators.Op import Operator
from operators.GEMM import GEMM

from tvm.runtime.module import Module

from typing import Callable


class RuntimeMod:
    def __init__(
        self,
        operator: Operator,
        arch: Arch,
        use_tc: bool,
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
        self.use_tc = use_tc
        self.compute_pattern: list[tuple[int, int, int]] = compute_pattern

        # Other attributes initialized in member functions
        self.sch: tvm.tir.Schedule = None
        self.runtime_mod: Module = None
        self.timer_mod: Callable[..., tvm.runtime.module.BenchmarkResult] = None
        self.execution_time: float = 0.0
        self.throughput: float = 0.0
        self.record: str = ""

    def build(self):
        """
        Build the runtime module
        """
        if self.use_tc:
            self.build_with_tc()
        else:
            self.build_without_tc()

    def build_without_tc(self):
        """
        Build the runtime module with CUDA Core
        """
        self.sch: tvm.tir.Schedule = self.operator.apply_without_tc(
            self.compute_pattern[1], self.compute_pattern[0], [1, 1, 1]
        )
        self.runtime_mod: Module = tvm.build(self.sch.mod, target="cuda")
        self.timer_mod = self.runtime_mod.time_evaluator(
            self.runtime_mod.entry_name, tvm.cuda(0)
        )

    def build_with_tc(self):
        """
        Build the runtime module with Tensor Core
        """
        self.sch: tvm.tir.Schedule = self.operator.apply_with_tc(self.compute_pattern)
        self.runtime_mod: Module = tvm.build(self.sch.mod, target="cuda")
        self.timer_mod = self.runtime_mod.time_evaluator(
            self.runtime_mod.entry_name, tvm.cuda(0)
        )

    def show(self):
        """
        Show the IRModule
        """
        self.sch.mod.show(black_format=False)
        cuda = self.runtime_mod.imported_modules[0].get_source()
        print(cuda)

    def get_mod(self):
        """
        Get the IRModule
        """
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
