"""
GEMM operator implementation
"""

import tvm
from tvm.script import tirx as T
from tvm.script import ir as I
from operators.Op import Operator
from utils.helper import dim3
from utils.tensor_core import schedule_tensorcore


def create_gemm_ir_module(M: int, N: int, K: int, dtype: str):
    @I.ir_module
    class GEMM_IRModule:
        @T.prim_func(s_tir=True)
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [K, N], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)
            for i, j, k in T.grid(M, N, K):
                with T.sblock("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.cast(0.0, dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return GEMM_IRModule


class GEMM(Operator):
    def __init__(
        self, M: int, N: int, K: int, dtype: str = "float32", name: str = "GEMM"
    ):
        """
        Args:
            M: int, the number of rows of the matrix A
            N: int, the number of columns of the matrix B
            K: int, the number of columns of the matrix A and the number of rows of the matrix B
            dtype: str, the data type of the matrix, "float32" or "float16"
            name: str, the name of the operator
        """

        # For unregular size, consider padding to the nearest power of 2
        # self.M = (1 << (M-1).bit_length())
        # self.N = (1 << (N-1).bit_length())
        # self.K = (1 << (K-1).bit_length())
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.num_flops = (
            M * N * K * 2
        )  # May need a pass to collect the number of flops.
        self.ir_module = create_gemm_ir_module(self.M, self.N, self.K, dtype)
        self.sch = tvm.s_tir.Schedule(self.ir_module)
        super().__init__(name="GEMM")

    def apply_without_tc(
        self, rtile0: list | tuple, rtile1: list | tuple, rtile2: list | tuple
    ):
        """
        Args:
            rtile0: list|tuple of int, the tile size of the shared memory
            rtile1: list|tuple of int, the tile size of the register level
            rtile2: list|tuple of int, the tile size of the third dimension
        """
        gridDim = dim3(x=self.N // rtile0[1], y=self.M // rtile0[0], z=1)
        blockDim = dim3(x=rtile0[1] // rtile1[1], y=rtile0[0] // rtile1[0], z=1)
        GRID_SIZE_X = gridDim.x * gridDim.y
        BLOCK_SIZE_X = blockDim.x * blockDim.y
        RM, RN, RK = rtile1
        BK = rtile0[2]

        sch = self.sch
        block_C = sch.get_sblock("C")
        block_shared_A = sch.cache_read(block_C, 0, "shared")
        block_shared_B = sch.cache_read(block_C, 1, "shared")
        block_local_C = sch.cache_write(block_C, 0, "local")

        i, j, k = sch.get_loops(block_C)
        by, vy, ty = sch.split(i, factors=[gridDim.y, RM, blockDim.y])
        bx, vx, tx = sch.split(j, factors=[gridDim.x, RN, blockDim.x])
        sch.reorder(by, bx, vy, vx, ty, tx, k)
        blockIdx_x = sch.fuse(by, bx)
        threadIdx_x = sch.fuse(ty, tx)

        sch.bind(blockIdx_x, "blockIdx.x")
        sch.bind(vy, "vthread.x")
        sch.bind(vx, "vthread.y")
        sch.bind(threadIdx_x, "threadIdx.x")
        k0, k1 = sch.split(k, factors=[None, BK])

        sch.compute_at(block_shared_A, k0)
        sch.compute_at(block_shared_B, k0)
        sch.reverse_compute_at(block_local_C, threadIdx_x)
        sch.decompose_reduction(block_C, threadIdx_x)

        # threads in the same threadblock collaboratively load A tile(BM x BK) to shared memory
        ax0, ax1 = sch.get_loops(block_shared_A)[-2:]
        ax_fused = sch.fuse(ax0, ax1)
        ax_unroll, ax_trd, ax1_vec = sch.split(
            ax_fused, factors=[None, BLOCK_SIZE_X, 4]
        )
        sch.unroll(ax_unroll)
        sch.bind(ax_trd, "threadIdx.x")
        sch.vectorize(ax1_vec)

        # threads in the same threadblock collaboratively load B tile(BK x BN) to shared memory
        ax0, ax1 = sch.get_loops(block_shared_B)[-2:]
        # sch.reorder(ax1, ax0)
        ax_fused = sch.fuse(ax0, ax1)
        ax_outer, ax_mid, ax_inner = sch.split(
            ax_fused, factors=[None, BLOCK_SIZE_X, 4]
        )
        sch.unroll(ax_outer)
        sch.bind(ax_mid, "threadIdx.x")
        sch.vectorize(ax_inner)

        return sch

    def apply_with_tc(self, compute_pattern: list[tuple[int, int, int]]):
        """
        Args:
            compute_pattern: list[tuple[int, int, int]], the compute pattern
        """
        return schedule_tensorcore(
            self.sch,
            compute_pattern,
            self.M,
            self.N,
            self.K,
            self.dtype,
            self.dtype,
        )
