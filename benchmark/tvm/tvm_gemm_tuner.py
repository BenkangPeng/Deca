"""
python benchmark/tvm/tvm_gemm_tuner.py --shape 4096 4096 4096 --use_tc --data_type float16
"""

import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule.database.json_database import JSONDatabase
import time
import shutil
import os
import tvm.tir.tensor_intrin

os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shape", nargs="*", type=int, default=[4096, 4096, 4096])
parser.add_argument("--target", type=str, default="nvidia/geforce-rtx-4090")
parser.add_argument("--use_tc", dest="use_tc", action="store_true", default=False)
parser.add_argument("--data_type", type=str, default="float32")
# Use --skip_tuning to skip tuning and use the cached database
parser.add_argument(
    "--skip_tuning", dest="skip_tuning", action="store_true", default=False
)
args = parser.parse_args()

M, N, K = args.shape
dtype = args.data_type
target = args.target
use_tc = args.use_tc
skip_tuning = args.skip_tuning


def dump(src, path):
    with open(path, "w") as f:
        f.write(src)


def create_gemm_ir_module(M, N, K, dtype):
    """
    Fp16 GEMM with fp16 accumulation
    """

    @I.ir_module
    class GEMM_IRModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [K, N], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return GEMM_IRModule


def create_gemm_ir_module_fp32_acc(M, N, K, dtype):
    """
    Fp16 GEMM with fp32 accumulation
    """

    @I.ir_module
    class GEMM_IRModule_fp32_acc:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [K, N], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)
            # Use fp32 to accumulate the result
            accum_buffer = T.alloc_buffer([M, N], dtype="float32")
            for i, j, k in T.grid(M, N, K):
                with T.block("C_acc"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        accum_buffer[vi, vj] = T.float32(0)
                    accum_buffer[vi, vj] = accum_buffer[vi, vj] + T.cast(
                        A[vi, vk], "float32"
                    ) * T.cast(B[vk, vj], "float32")
            # Convert the accum_buffer to float16 and store it in C
            for i, j in T.grid(M, N):
                with T.block("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = T.cast(accum_buffer[vi, vj], dtype)

    return GEMM_IRModule_fp32_acc


if use_tc:
    work_dir = os.path.join(os.path.dirname(__file__), "tune_tmp_tc/")
else:
    work_dir = os.path.join(os.path.dirname(__file__), "tune_tmp/")

# target = "nvidia/nvidia-h100"
# target = "nvidia/geforce-rtx-4090"


def tune_gemm():
    # Choose using fp16 or fp32 accumulation
    # gemm_ir_module = create_gemm_ir_module(M, N, K, dtype)
    gemm_ir_module = create_gemm_ir_module_fp32_acc(M, N, K, dtype)
    start_time = time.time()
    if not skip_tuning:
        database = ms.tune_tir(
            mod=gemm_ir_module,
            target=target,
            work_dir=work_dir,
            # max times to run the candidate programs in device(e.g. GPU)
            max_trials_global=500,
        )
    else:
        path_workload = os.path.join(work_dir, "database_workload.json")
        path_tuning_record = os.path.join(work_dir, "database_tuning_record.json")
        database = JSONDatabase(path_workload, path_tuning_record, work_dir=work_dir)
    end_time = time.time()
    tuning_time = end_time - start_time

    sch = ms.tir_integration.compile_tir(database, gemm_ir_module, target)

    # shutil.rmtree(work_dir)
    rt_mod = tvm.build(sch.mod, "cuda")
    dump(
        rt_mod.imported_modules[0].get_source(),
        os.path.join(os.path.dirname(__file__), "gemm_ir_module.cu"),
    )
    dump(sch.mod.script(), os.path.join(os.path.dirname(__file__), "gemm_ir_module.py"))

    a_np = np.random.uniform(low=0.0, high=0.01, size=(M, K)).astype(dtype)
    b_np = np.random.uniform(low=0.0, high=0.01, size=(K, N)).astype(dtype)
    a_nd = tvm.nd.array(a_np, tvm.cuda(0))
    b_nd = tvm.nd.array(b_np, tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((M, N), dtype=dtype), tvm.cuda(0))

    rt_mod["main"](a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)

    num_flop = 2 * M * K * N
    evaluator = rt_mod.time_evaluator(rt_mod.entry_name, tvm.cuda(0), number=10)
    timer = evaluator(a_nd, b_nd, c_nd).mean
    print("time cost on average: %f ms" % (timer * 1e3))
    print("GEMM-Blocking: %f GFLOPS" % (num_flop / timer / 1e9))

    if use_tc:
        log_path = os.path.join(os.path.dirname(__file__), "tvm_tuning_tc.log")
    else:
        log_path = os.path.join(os.path.dirname(__file__), "tvm_tuning.log")
    with open(log_path, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}:\nTVM GEMM Tuner: {M}, {N}, {K}, dtype: {dtype}, target: {target}, use_tc: {use_tc}, tuning time: {tuning_time} seconds, runtime cost: {timer * 1e3} ms, throughput: {num_flop / timer / 1e12} TFLOPS\n\n\n"
        )


if __name__ == "__main__":
    tune_gemm()
