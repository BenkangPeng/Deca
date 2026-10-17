"""TVM 0.27 migration checks. Run: python -m pytest tests -q (CUDA required)."""

import numpy as np
import pytest
import torch
import tvm

from arch import H20
from kernels.flash_attn_tir import FAConfig
from kernels.gemm_tir import GemmConfig
from kernels.rowwise_tir import RowConfig
from operators.FlashAttention import FlashAttention
from operators.GEMM import GEMM
from operators.GEMMTC import GEMMTC
from operators.RMSNorm import RMSNorm
from operators.Softmax import Softmax
from RuntimeMod import RuntimeMod
from utils.helper import GPUDetector

pytestmark = pytest.mark.skipif(not tvm.cuda().exist, reason="CUDA required")


def test_h20_detection():
    dev = tvm.cuda()
    if "H20" not in dev.device_name:
        pytest.skip("H20 hardware required")
    arch = GPUDetector.create_gpu_instance(verbose=False)
    assert isinstance(arch, H20)
    assert arch.sm_nums == dev.multi_processor_count
    assert arch.compute_sm_partition == [arch.sm_nums, 4]
    assert arch.compute_capability == (9, 0)
    assert arch.l2_bytes == dev.l2_cache_size_bytes
    assert arch.max_dyn_smem_per_block == 232448


@pytest.mark.parametrize("tc", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_scheduled_gemm(tc, wide):
    dtype = "float16" if tc else "float32"
    pattern = [(16, 16, 16), (32, 32, 16)] if tc else [(4, 4, 1), (32, 32, 16)]
    if wide:
        pattern = [(32, 32, 16), (64, 64, 32)] if tc else [(4, 8, 1), (32, 64, 32)]
    rt = RuntimeMod(GEMM(64, 64, 64, dtype), GPUDetector.create_gpu_instance(False), tc, pattern)
    rt.build()
    rng = np.random.default_rng(0)
    a, b = [rng.uniform(-1, 1, (64, 64)).astype(dtype) for _ in range(2)]
    args = [tvm.runtime.tensor(x, tvm.cuda()) for x in (a, b, np.zeros_like(a))]
    rt.runtime_mod["main"](*args)
    np.testing.assert_allclose(args[-1].numpy(), a.astype("float32") @ b.astype("float32"),
                               atol=0.05 if tc else 1e-4, rtol=0.02 if tc else 1e-4)
    rt.time_evaluator()
    assert rt.execution_time > 0


@pytest.mark.parametrize("acc,splits,epi", [("fp32", 1, True), ("fp32", 1, False),
                                              ("fp16", 1, True), ("fp16", 2, True)])
def test_gemm_kernel(acc, splits, epi, tmp_path):
    op = GEMMTC(128, 256, 128, acc=acc, acc_splits=splits)
    cfg = GemmConfig(raster=1, acc_f16=acc == "fp16", acc_splits=splits, epi_smem=epi)
    rt = RuntimeMod(op, GPUDetector.create_gpu_instance(False), True, cfg)
    rt.build()
    rt.time_kernel_op()
    assert rt.execution_time > 0
    # Exercise library persistence, not only in-process compilation.
    path = str(tmp_path / "gemm.so")
    rt.runtime_mod.export_library(path)
    rt.load_kernel(path)
    rt.time_kernel_op()


@pytest.mark.parametrize("op_cls", [Softmax, RMSNorm])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("mode,hint", [("reg", False), ("twopass", False), ("reg", True)])
def test_rowwise(op_cls, dtype, mode, hint):
    op = op_cls(8, 1032, dtype=dtype)
    cfg = RowConfig(threads_per_row=128, mode=mode, hint=hint)
    rt = RuntimeMod(op, GPUDetector.create_gpu_instance(False), False, cfg)
    rt.build()
    rt.time_kernel_op()
    assert rt.execution_time > 0


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("stages,q_shared,swizzle", [(1, False, True), (2, True, False)])
def test_flash_attention(causal, stages, q_shared, swizzle):
    op = FlashAttention(1, 1, 128, 128, 64, causal=causal)
    cfg = FAConfig(stages=stages, q_in_smem=q_shared, swizzle=swizzle)
    rt = RuntimeMod(op, GPUDetector.create_gpu_instance(False), True, cfg)
    rt.build()
    rt.time_flash_attn()
    assert rt.execution_time > 0


def test_hopper_large_shared_memory():
    arch = GPUDetector.create_gpu_instance(False)
    if arch.compute_capability < (9, 0):
        pytest.skip("Hopper shared memory required")
    cfg = GemmConfig(BK=64, raster=1)
    # 144 KiB exceeds the previous hard-coded Ada limit of 99 KiB.
    op = GEMMTC(128, 256, 256)
    assert op.is_valid(cfg)
    rt = RuntimeMod(op, arch, True, cfg)
    rt.build()
    rt.time_kernel_op()


def test_h20_solver_and_ranking():
    from Solver import solve_gemm_tc, solve_rowwise, solve_flash_attn
    from cost_model.CostModel import rank_gemm_configs, rank_rowwise_configs, rank_fa_configs
    from kernels.gemm_tir import smem_bytes

    arch = H20()
    gemm = solve_gemm_tc(arch, 1024, 1024, 256, count=10)
    rows = solve_rowwise(arch, "softmax", 128, 1032, count=10)
    attn = solve_flash_attn(arch, 1, 2, 4096, 4096, 64, count=10)
    assert gemm and rows and attn
    assert all(smem_bytes(c) <= arch.max_dyn_smem_per_block for c in gemm)
    assert rank_gemm_configs(gemm, arch, 1024, 1024, 256, top_n=1)
    assert rank_rowwise_configs(rows, arch, "softmax", 128, 1032, top_n=1)
    assert rank_fa_configs(attn, arch, (1, 2, 4096, 4096, 64), top_n=1)


def test_conv2d_script():
    from operators.Conv2D import Conv2D

    op = Conv2D(1, 2, 5, 5, 3, 3, 3, padding=1)
    mod = tvm.compile(op.ir_module, target="llvm", tir_pipeline="s_tir").mod
    rng = np.random.default_rng(1)
    a = rng.normal(size=(1, 2, 5, 5)).astype("float32")
    w = rng.normal(size=(3, 2, 3, 3)).astype("float32")
    aa, ww = [tvm.runtime.tensor(x) for x in (a, w)]
    out = tvm.runtime.empty((1, 3, 5, 5), "float32")
    mod["main"](aa, ww, out)
    ref = torch.nn.functional.conv2d(torch.from_numpy(a), torch.from_numpy(w), padding=1)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)


def test_cuda_source_dump(tmp_path):
    from kernels.rowwise_tir import make_rowwise, build_rowwise

    mod = make_rowwise("softmax", 8, 128, "float16", RowConfig())
    build_rowwise(mod, dump_dir=str(tmp_path))
    sources = list(tmp_path.glob("*.cu"))
    assert sources and "__global__" in sources[0].read_text()
