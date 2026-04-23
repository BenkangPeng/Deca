import tvm
from tvm import te
from .tc_intrin import (
    init_intrin_strides,
    register_wmma_load_intrin,
    register_wmma_fill_intrin,
    register_wmma_intrin,
    register_wmma_store_intrin,
)
import math
import numpy as np
from . import (
    getSpatialLoopRVs,
    getReduceLoopRVs,
    calculate_factors,
)
from tvm.tir import Schedule
from typing import Tuple, Optional
# from config import rProg, rTile


def schedule_tensorcore(
    tvm_schedule: Schedule,
    # rprog: rProg,
    compute_pattern: list[tuple[int, int, int]],
    M: int,
    N: int,
    K: int,
    input_dtype: str = "float16",
    output_dtype: str = "float16",
    verbose: bool = False,
) -> Schedule:
    """
    Schedule dense operator using Tensorcore
    """
    Mdim, Ndim = M, N
    s = tvm_schedule
    data_dtype = input_dtype
    out_dtype = output_dtype

    # Explicit memory access
    block_compute = s.get_block("C")
    AS: tvm.tir.BlockRV = s.cache_read(
        block_compute, read_buffer_index=0, storage_scope="shared"
    )
    BS: tvm.tir.BlockRV = s.cache_read(
        block_compute, read_buffer_index=1, storage_scope="shared"
    )
    AF: tvm.tir.BlockRV = s.cache_read(
        block_compute, read_buffer_index=0, storage_scope="wmma.matrix_a"
    )
    BF: tvm.tir.BlockRV = s.cache_read(
        block_compute, read_buffer_index=1, storage_scope="wmma.matrix_b"
    )
    CF: tvm.tir.BlockRV = s.cache_write(
        block_compute, write_buffer_index=0, storage_scope="wmma.accumulator"
    )

    if verbose:
        s.mod.show()

    # Extract Sokoban scheduling information
    #
    #  -------------------               -------             ---
    # |    |    |    |    |             |---|-`-|`--------> |   | wmma_m
    # |    |    |    |  `-|-----------> |---|---| warp_m     ---
    # |    |    |    |    |             |---|---|           wmma_n
    # |----|----|----|----| block_m      -------
    # |    |    |    |    |              warp_n
    # |    |    |    |    |
    # |    |    |    |    |
    #  -------------------
    #       block_n
    #

    # warp_size = 32
    # wmma_m, wmma_n = rprog.GetTile(2).SDimensions()
    # wmma_k = rprog.GetTile(2).RDimensions()[0]
    # warp_m, warp_n = rprog.GetTile(1).SDimensions()
    # block_m, block_n = rprog.GetTile(0).SDimensions()
    # rstep_size = rprog.GetTile(0).RDimensions()[0] // wmma_k

    warp_size = 32
    wmma_m, wmma_n, wmma_k = 16, 16, 16
    warp_m, warp_n = compute_pattern[0][0], compute_pattern[0][1]
    block_m, block_n = compute_pattern[1][0], compute_pattern[1][1]
    rstep_size = compute_pattern[1][2] // wmma_k

    block_row_warps = block_m // warp_m
    block_col_warps = block_n // warp_n
    warp_row_tiles = warp_m // wmma_m
    warp_col_tiles = warp_n // wmma_n
    offset = 8
    offsetCS = 0
    vec = 1

    # Define the stride of intrin functions
    # CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    # AS_stride = [AS_align, 1]
    # BS_stride = [BS_align, 1]
    # AF_stride = [wmma_k, 1]
    # BF_stride = [wmma_k, 1]
    AF_stride, AS_stride = init_intrin_strides(
        wmma_shape=[wmma_m, wmma_k],
        warp_s=warp_row_tiles,
        block_s=block_row_warps,
        rstep_size=rstep_size,
        offset=offset,
        layout="row_major",
    )
    BF_stride, BS_stride = init_intrin_strides(
        wmma_shape=[wmma_k, wmma_n],
        warp_s=warp_col_tiles,
        block_s=block_col_warps,
        rstep_size=rstep_size,
        offset=offset,
        layout="col_major",
    )

    AS_align = AS_stride[0]
    BS_align = BS_stride[0]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    C_stride = [Ndim, 1]

    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps

    loops_wmma_acc = s.get_loops(CF)

    m, n = loops_wmma_acc

    block_i, mc = s.split(m, factors=calculate_factors(s, m, block_factor_m))
    block_j, nc = s.split(n, factors=calculate_factors(s, n, block_factor_n))

    mm, mmi = s.split(mc, factors=calculate_factors(s, mc, wmma_m))
    nn, nni = s.split(nc, factors=calculate_factors(s, nc, wmma_n))
    mm, mmii = s.split(mm, factors=calculate_factors(s, mm, warp_row_tiles))
    nn, nnii = s.split(nn, factors=calculate_factors(s, nn, warp_col_tiles))

    s.reorder(block_i, block_j, mm, nn, mmii, nnii, mmi, nni)
    s.bind(block_i, "blockIdx.x")
    s.bind(block_j, "blockIdx.y")

    if verbose:
        s.mod.show()

    s.bind(mm, "threadIdx.y")
    s.bind(nn, "threadIdx.z")
    s.unroll(mmii)
    s.unroll(nnii)

    if verbose:
        s.mod.show()

    block_compute = s.get_block("C")
    s.compute_at(block_compute, nn)
    warp_i, warp_j = getSpatialLoopRVs(s, block_compute)
    warp_i, _ii = s.split(warp_i, factors=calculate_factors(s, warp_i, wmma_m))
    warp_j, _jj = s.split(warp_j, factors=calculate_factors(s, warp_j, wmma_n))
    (k,) = getReduceLoopRVs(s, block_compute)
    k, _k = s.split(k, factors=calculate_factors(s, k, wmma_k))
    ko, ki = s.split(k, factors=calculate_factors(s, k, rstep_size))
    s.reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
    s.unroll(ki)
    s.unroll(warp_i)
    s.unroll(warp_j)

    # Purpose: Optimize reduction operations by hoisting initialization
    #          out of hot loops
    # Ref: https://discuss.tvm.apache.org/t/optimizing-reduction-initialization-in-generated-cuda-code/18613
    # ------------------------------------------------------------
    # Original Problem:
    #   In reduction patterns (e.g., matrix multiplication), initialization
    #   (e.g., C[i,j] = 0) typically occurs INSIDE the reduction loop nest,
    #   guarded by a condition like:
    #     "if (k == 0): C[i,j] = 0"
    #   This causes two performance issues:
    #     1. Branch Divergence: Threads waste cycles checking the condition
    #        every iteration.
    #     2. Redundant Checks: Initialization only needs to run once, but
    #        the condition is evaluated N times.
    #
    # Solution: Decompose the reduction into two phases:
    #   1. Initialization Phase (runs once):
    #      - Sets output buffers to zero (e.g., C_init block)
    #      - Positioned OUTSIDE all computation loops
    #   2. Update Phase (runs in hot loops):
    #      - Pure computation without initialization checks (e.g., C_update
    #        block)
    #
    # Performance Impact:
    #   - Eliminates branch mispredictions (~20% speedup observed in generated
    #     CUDA kernels for a 5120 x 5120 x 5120 matrix multiplication operation).
    # ------------------------------------------------------------
    # How this code achieves it:
    #   - decompose_reduction(block_compute, loops[3]):
    #     * Identifies the reduction block (e.g., matrix multiply accumulation)
    #     * Splits it at the SECOND-TO-LAST loop level
    #     * Creates two blocks:
    #       - {block_name}_init: Initializes output
    #       - {block_name}_update: Pure computation
    initial_block = s.decompose_reduction(block_compute, nn)

    if verbose:
        s.mod.show()

    s.compute_at(AF, ki)
    m, i = getSpatialLoopRVs(s, AF)
    m, m_ii = s.split(m, factors=calculate_factors(s, m, wmma_m))
    i, i_jj = s.split(i, factors=calculate_factors(s, i, wmma_k))
    s.reorder(m, i, m_ii, i_jj)
    s.unroll(m)
    s.unroll(i)

    if verbose:
        s.mod.show()

    s.compute_at(BF, ki)
    i, n = getSpatialLoopRVs(s, BF)
    i, i_ii = s.split(i, factors=calculate_factors(s, i, wmma_k))
    n, n_ii = s.split(n, factors=calculate_factors(s, n, wmma_n))
    s.reorder(i, n, i_ii, n_ii)
    s.unroll(i)
    s.unroll(n)

    if verbose:
        s.mod.show()

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s.compute_at(stage, ko)
        xo, yo = getSpatialLoopRVs(s, stage)
        # NOTE: `index` of xo in getSpatialLoopRVs(s, stage) equals to 0.
        s.storage_align(
            stage, buffer_index=0, axis=0, factor=strides - 1, offset=strides
        )
        t = s.fuse(xo, yo)
        t, vi = s.split(t, factors=calculate_factors(s, t, vec))
        t, tx = s.split(t, factors=calculate_factors(s, t, warp_size))
        t, ty = s.split(t, factors=calculate_factors(s, t, block_row_warps))
        t, tz = s.split(t, factors=calculate_factors(s, t, block_col_warps))
        s.bind(ty, "threadIdx.y")
        s.bind(tz, "threadIdx.z")
        s.bind(tx, "threadIdx.x")
        s.unroll(t)
        s.vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    if verbose:
        s.mod.show()

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype)
            * BL_gemm[k_gemm, jj].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    intrin_name = register_wmma_load_intrin(
        (wmma_m, wmma_n, wmma_k),
        (warp_row_tiles, warp_col_tiles),
        (block_row_warps, block_col_warps),
        rstep_size,
        AF_stride,
        AS_stride,
        "wmma.matrix_a",
        "row_major",
        data_dtype,
    )
    s.tensorize(m_ii, intrin_name)

    if verbose:
        s.mod.show()

    intrin_name = register_wmma_load_intrin(
        (wmma_m, wmma_n, wmma_k),
        (warp_row_tiles, warp_col_tiles),
        (block_row_warps, block_col_warps),
        rstep_size,
        BF_stride,
        BS_stride,
        "wmma.matrix_b",
        "col_major",
        data_dtype,
    )
    s.tensorize(i_ii, intrin_name)

    if verbose:
        s.mod.show()

    intrin_name = register_wmma_intrin(
        AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape
    )
    s.tensorize(_ii, intrin_name)

    intrin_name = register_wmma_fill_intrin(
        (wmma_m, wmma_n, wmma_k), CF_stride, out_dtype
    )
    s.tensorize(s.get_loops(initial_block)[-2], intrin_name)

    if verbose:
        s.mod.show()

    intrin_name = register_wmma_store_intrin(C_stride, CF_stride, shape, out_dtype)
    s.tensorize(mmi, intrin_name)

    if verbose:
        s.mod.show()

    return s
