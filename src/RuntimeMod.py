import tvm
import numpy as np

from arch.Arch import Arch
from arch.RTX4090 import RTX4090
from operators.Op import Operator
from operators.FlashAttention import FlashAttention

from tvm.runtime.module import Module

from typing import Callable


class RuntimeMod:
    def __init__(
        self,
        operator: Operator,
        arch: Arch,
        use_tc: bool,
        compute_pattern,
    ):
        """
        Args:
            operator: Operator, kernel to be tuned
            arch: Arch, the architecture of the GPU
            compute_pattern: list[tuple[int, int, int]] for GEMM-like operators,
                or a `FAConfig` for FlashAttention
        """
        self.operator = operator
        self.arch = arch
        self.use_tc = use_tc
        self.compute_pattern = compute_pattern

        # Other attributes initialized in member functions
        self.sch: tvm.tir.Schedule = None
        self.ir_mod: tvm.IRModule = None
        self.runtime_mod: Module = None
        self.timer_mod: Callable[..., tvm.runtime.module.BenchmarkResult] = None
        self.execution_time: float = 0.0
        self.throughput: float = 0.0
        self.max_error: float = float("nan")
        self.record: str = ""

    @property
    def is_flash_attn(self) -> bool:
        return isinstance(self.operator, FlashAttention)

    @property
    def is_kernel_op(self) -> bool:
        """Operators that generate their kernel from a config (`build_mod`)."""
        return hasattr(self.operator, "build_mod")

    def build(self):
        """
        Build the runtime module
        """
        if self.is_flash_attn:
            self.build_flash_attn()
        elif self.is_kernel_op:
            self.ir_mod = self.operator.build_mod(self.compute_pattern)
            self.runtime_mod = self.operator.compile(self.ir_mod)
        elif self.use_tc:
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

    def build_flash_attn(self):
        """
        Build the FlashAttention kernel described by the `FAConfig`
        """
        from kernels.flash_attn_tir import build_flash_attn

        self.ir_mod = self.operator.build_mod(self.compute_pattern)
        self.runtime_mod = build_flash_attn(self.ir_mod)

    def load_flash_attn(self, path: str):
        """
        Load a FlashAttention kernel exported by `build_flash_attn_to_file`
        """
        self.load_kernel(path)

    def load_kernel(self, path: str):
        """
        Load a kernel of a `build_mod` operator exported to a shared library
        """
        self.ir_mod = self.operator.build_mod(self.compute_pattern)
        self.runtime_mod = tvm.runtime.load_module(path)

    def show(self):
        """
        Show the IRModule
        """
        self.get_mod().show(black_format=False)
        if not self.is_kernel_op:
            cuda = self.runtime_mod.imported_modules[0].get_source()
            print(cuda)

    def get_mod(self):
        """
        Get the IRModule
        """
        return self.ir_mod if self.is_kernel_op else self.sch.mod

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
        if self.is_flash_attn:
            return self.time_flash_attn()
        if self.is_kernel_op:
            return self.time_kernel_op()

        M, N, K = self.operator.M, self.operator.N, self.operator.K
        num_flops = self.operator.num_flops
        dtype = self.operator.dtype

        ctx = tvm.cuda(0)
        cuda_a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
        cuda_b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
        cuda_c = tvm.nd.array(np.zeros((M, N)).astype(dtype), ctx)

        t = self.timer_mod(cuda_a, cuda_b, cuda_c).mean
        self.execution_time = t
        self.throughput = num_flops / t / 1e12  # unit: TFLOPS
        self.record = f"{self.operator.name}({M}, {N}, {K}, {dtype}) \n\t{self.compute_pattern} \n\tcost time: {t * 1000} ms, throughput: {self.throughput} TFLOPS"

    def time_flash_attn(self, verify: bool = True, atol: float = 2e-2):
        """
        Time (and optionally verify against torch SDPA) the FlashAttention kernel.
        Inputs are torch tensors shared with TVM through DLPack, timed with the
        same CUDA-event harness as the SDPA baseline.
        """
        import torch
        from kernels.attn_bench import bench_cuda, make_qkv, sdpa_flash

        op = self.operator
        B, H, S_q, S_kv, D = op.shape
        q, k, v = make_qkv(B, H, S_q, S_kv, D)
        o = torch.empty_like(q)
        args = [tvm.nd.from_dlpack(x) for x in (q, k, v, o)]
        func = self.runtime_mod["main"]

        if verify:
            func(*args)
            torch.cuda.synchronize()
            ref = sdpa_flash(q, k, v, causal=op.causal, scale=op.softmax_scale)
            self.max_error = (o.float() - ref.float()).abs().max().item()
            if not self.max_error <= atol:
                raise RuntimeError(
                    f"{self.compute_pattern}: max error {self.max_error} exceeds {atol}"
                )

        t = bench_cuda(lambda: func(*args))
        self.execution_time = t
        self.throughput = op.num_flops / t / 1e12
        self.record = (
            f"{op.name}(B={B}, H={H}, S_q={S_q}, S_kv={S_kv}, D={D}, {op.dtype}"
            f"{', causal' if op.causal else ''}) "
            f"\n\t{self.compute_pattern} \n\tcost time: {t * 1000:.4f} ms, "
            f"throughput: {self.throughput:.2f} TFLOPS, max err: {self.max_error:.2e}"
        )

    def time_kernel_op(self, verify: bool = True):
        """
        Time (and optionally verify against the operator's torch reference) a
        `build_mod` operator. Throughput is reported in GB/s for memory-bound
        operators and in TFLOPS for operators with `metric == "TFLOPS"`.
        """
        import torch
        from kernels.attn_bench import bench_cuda

        op = self.operator
        inputs = op.make_inputs()
        out = op.make_output(inputs) if hasattr(op, "make_output") else torch.empty_like(inputs[0])
        args = [tvm.nd.from_dlpack(x) for x in op.kernel_args(inputs, out)]
        func = self.runtime_mod["main"]

        if verify:
            func(*args)
            torch.cuda.synchronize()
            ref = op.reference(inputs)
            atol, rtol = op.tolerance
            self.max_error, excess = 0.0, 0.0
            # Row chunks keep the fp32 temporaries small for vocab-sized inputs.
            step = max(1, (1 << 24) // ref.shape[-1])
            for r0 in range(0, ref.shape[0], step):
                r = ref[r0 : r0 + step].float()
                diff = (out[r0 : r0 + step].float() - r).abs()
                self.max_error = max(self.max_error, diff.max().item())
                excess = max(excess, (diff / (atol + rtol * r.abs())).max().item())
            del ref
            if not excess <= 1.0:
                raise RuntimeError(
                    f"{self.compute_pattern}: max error {self.max_error:.3e} exceeds "
                    f"atol={atol} + rtol={rtol} * |ref| (x{excess:.2f})"
                )

        t = bench_cuda(lambda: func(*args))
        self.execution_time = t
        if getattr(op, "metric", "GB/s") == "TFLOPS":
            self.throughput = op.num_flops / t / 1e12
            unit = "TFLOPS"
        else:
            self.throughput = op.num_bytes / t / 1e9
            unit = "GB/s"
        self.record = (
            f"{op.describe()} \n\t{self.compute_pattern} \n\tcost time: {t * 1000:.4f} ms, "
            f"throughput: {self.throughput:.1f} {unit}, max err: {self.max_error:.2e}"
        )
