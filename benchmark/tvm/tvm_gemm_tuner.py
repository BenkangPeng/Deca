import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule.database.json_database import JSONDatabase
import time
import shutil
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shape", nargs="*", type=int, default=[4096, 4096, 4096])
parser.add_argument("--target", type=str, default="nvidia/geforce-rtx-4090")
parser.add_argument("--use_tc", dest="use_tc", action="store_true", default=False)
parser.add_argument("--data_type", type=str, default="float32")
args = parser.parse_args()

M, N, K = args.shape
dtype = args.data_type
target = args.target
use_tc = args.use_tc

def create_gemm_ir_module(M, N, K, dtype):
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


if use_tc:
    work_dir = os.path.join(os.path.dirname(__file__), "tune_tmp_tc/")
else:
    work_dir = os.path.join(os.path.dirname(__file__), "tune_tmp/")

# target = "nvidia/nvidia-h100"
# target = "nvidia/geforce-rtx-4090"

def tune_gemm():
    gemm_ir_module = create_gemm_ir_module(M, N, K, dtype)
    start_time = time.time()
    database = ms.tune_tir(
        mod=gemm_ir_module,
        target=target,
        work_dir=work_dir,
        # max times to run the candidate programs in device(e.g. GPU)
        max_trials_global=1000
    )
    end_time = time.time()
    tuning_time = end_time - start_time

    path_workload = os.path.join(work_dir, "database_workload.json")
    path_tuning_record = os.path.join(work_dir, "database_tuning_record.json")
    database = JSONDatabase(path_workload, path_tuning_record, work_dir=work_dir)
    sch = ms.tir_integration.compile_tir(database, gemm_ir_module, target)

    shutil.rmtree(work_dir)
    rt_mod = tvm.build(sch.mod, "cuda")

    a_np = np.random.uniform(size=(M, K)).astype(dtype)
    b_np = np.random.uniform(size=(K, N)).astype(dtype)
    a_nd = tvm.nd.array(a_np, tvm.cuda(0))
    b_nd = tvm.nd.array(b_np, tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((M, N), dtype=dtype), tvm.cuda(0))

    rt_mod["main"](a_nd, b_nd, c_nd)
    # np.testing.assert_allclose(c_nd.numpy(), a_np @ b_np, rtol=1e-5, atol=1e-5)

    num_flop = 2 * M * K * N
    evaluator = rt_mod.time_evaluator(rt_mod.entry_name, tvm.cuda(0), number=10)
    timer = evaluator(a_nd, b_nd, c_nd).mean
    print("time cost on average: %f ms" % (timer * 1e3))
    print("GEMM-Blocking: %f GFLOPS" %
        (num_flop / timer / 1e9))

    if use_tc:
        log_path = os.path.join(os.path.dirname(__file__), "tvm_tuning_tc.log")
    else:
        log_path = os.path.join(os.path.dirname(__file__), "tvm_tuning.log")
    with open(log_path, "a") as f:
        f.write(f"TVM GEMM Tuner: {M}, {N}, {K}, dtype: {dtype}, target: {target}, use_tc: {use_tc}, tuning time: {tuning_time} seconds, runtime cost: {timer * 1e3} ms, throughput: {num_flop / timer / 1e12} TFLOPS\n\n\n")



if __name__ == "__main__":
    tune_gemm()