"""
GEMM operator implementation
"""

import tvm
from tvm.script import tir as T
from tvm.script import ir as I
from operators.Op import Operator
from utils.helper import dim3
from utils.tensor_core import schedule_tensorcore


def create_gemm_ir_module(M: int, N: int, K: int, dtype: str):
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
        self.sch = tvm.tir.Schedule(self.ir_module)
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
        BLOCK_SIZE_X = blockDim.x * blockDim.y
        RM, RN, RK = rtile1
        BK = rtile0[2]

        sch = self.sch
        block_C = sch.get_block("C")
        block_shared_A = sch.cache_read(block_C, 0, "shared")
        block_shared_B = sch.cache_read(block_C, 1, "shared")
        # Register fragments of A and B, reloaded for every k of the tile.
        block_local_A = sch.cache_read(block_C, 0, "local")
        block_local_B = sch.cache_read(block_C, 1, "local")
        block_local_C = sch.cache_write(block_C, 0, "local")

        # Each thread owns a contiguous RM x RN tile of C, so that its A / B
        # fragments are contiguous in shared memory and read with 128-bit loads
        # (4 loads per 64 FMAs for an 8 x 8 tile instead of 16 scalar loads).
        i, j, k = sch.get_loops(block_C)
        by, ty, vy = sch.split(i, factors=[gridDim.y, blockDim.y, RM])
        bx, tx, vx = sch.split(j, factors=[gridDim.x, blockDim.x, RN])
        k0, k1 = sch.split(k, factors=[None, BK])
        sch.reorder(by, bx, ty, tx, k0, k1, vy, vx)
        blockIdx_x = sch.fuse(by, bx)
        threadIdx_x = sch.fuse(ty, tx)
        sch.bind(blockIdx_x, "blockIdx.x")
        sch.bind(threadIdx_x, "threadIdx.x")

        sch.compute_at(block_local_A, k1)
        sch.compute_at(block_local_B, k1)
        sch.compute_at(block_shared_A, k0)
        sch.compute_at(block_shared_B, k0)
        sch.reverse_compute_at(block_local_C, threadIdx_x)
        sch.decompose_reduction(block_C, k0)

        # A is stored k-major (BK x BM) in shared memory: a thread's RM rows of
        # one k are then contiguous.
        sch.transform_layout(block_shared_A, ("write", 0), lambda m, kk: (kk, m))

        # Threads of the block collaboratively load the A tile (BM x BK) and the
        # B tile (BK x BN) to shared memory with 16-byte global loads.
        for block_shared in (block_shared_A, block_shared_B):
            ax0, ax1 = sch.get_loops(block_shared)[-2:]
            ax_fused = sch.fuse(ax0, ax1)
            ax_unroll, ax_trd, ax_vec = sch.split(
                ax_fused, factors=[None, BLOCK_SIZE_X, 4]
            )
            sch.unroll(ax_unroll)
            sch.bind(ax_trd, "threadIdx.x")
            sch.vectorize(ax_vec)

        def vectorize_by_4(block):
            """Vectorizes the innermost loop in chunks of at most 4 lanes (float4)."""
            loop = sch.get_loops(block)[-1]
            extent = int(sch.get(loop).extent)
            if extent > 4 and extent % 4 == 0:
                outer, loop = sch.split(loop, factors=[None, 4])
                sch.unroll(outer)
            sch.vectorize(loop)

        vectorize_by_4(block_local_A)
        vectorize_by_4(block_local_B)
        sch.unroll(k1)
        sch.unroll(vy)
        sch.unroll(vx)
        vectorize_by_4(block_local_C)

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
