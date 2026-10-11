from typing import Any
from z3.z3 import BitVecRef
from z3 import *
from arch.RTX4090 import RTX4090
from arch.H100 import H100
from arch.V100 import V100

from src.utils.helper import GPUDetector, byte_size

arch = GPUDetector.create_gpu_instance(verbose=False)


def solve_reg(
    num_regs,
    type_size=4,
    use_tc=False,
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    count: int = 20,
):
    """
    Solve the register tiling problem for the given number of registers.
    Args:
        num_regs: int, the number of registers that can be used by a single thread
        type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        use_tc: bool, whether to use tensor core
        M, N, K: int, the shape of the matrix
        count: int, the number of solutions to find
    Returns:
        list of tuples, each tuple contains the tiling sizes for the register level
    """

    results = []

    if use_tc:
        s = Solver()
        rm, rn, rk = BitVecs("rm rn rk", 64)

        # Ensure that the tile size for each warp computation is a power of two and a multiple of 16.
        for var in [rm, rn]:
            s.add(var % 16 == 0)
            s.add(UGT(var, 0))
            s.add(var & (var - 1) == 0)

        # Set warp_k to 16
        s.add(rk == 16)

        # Limit the tile size for each warp
        s.add(ULE(rm, 256))
        s.add(ULE(rn, 256))

        # print("=== All constraints ===")
        # for a in s.assertions():
        #     print(a)

        for i in range(count):
            if s.check() == sat:
                m = s.model()
                res = (m[rm].as_long(), m[rn].as_long(), m[rk].as_long())
                results.append(res)

                # BLOCKING CLAUSE:
                # "Find a new solution where (rm,rn,rk) is NOT the current (rm,rn,rk)"
                s.add(Or(rm != m[rm], rn != m[rn], rk != m[rk]))
            else:
                break
    else:
        s = Solver()
        # Each thread computes a tile of size rm x rn x rk
        rm, rn, rk = BitVecs("rm rn rk", 64)

        s.add(ULE(rm, 256))
        s.add(ULE(rn, 256))

        s.add(UGE(rm, 2))
        s.add(UGE(rn, 2))

        # m_bv = BitVecVal(num_regs, 64)
        m_bv = num_regs
        # reduce dimension to 1
        s.add(rk == 1)
        for var in [rm, rn]:
            s.add(UGT(var, 0))
            s.add(var & (var - 1) == 0)

        # used_reg = (
        #     (rm * rk + rn * rk + rm * rn) * type_size / 4
        # )  # Each register can store 4 bytes of data
        used_reg = (rm * rn) * type_size / 4
        # used_reg > 0 && used_reg <= M
        s.add(ULE(used_reg, m_bv))
        s.add(UGT(used_reg, 0))
        # s.add(Or(rm == rn, rm == rn >> 1, rn == rm >> 1))

        # print("=== All constraints ===")
        # for a in s.assertions():
        #     print(a)

        for i in range(count):
            if s.check() == sat:
                m = s.model()
                res = (m[rm].as_long(), m[rn].as_long(), m[rk].as_long())
                results.append(res)

                # BLOCKING CLAUSE:
                # "Find a new solution where (rm,rn,rk) is NOT the current (rm,rn,rk)"
                s.add(Or(rm != m[rm], rn != m[rn], rk != m[rk]))
            else:
                break
    return results


def IsComputeIntensive(rtile1, K, dtype: str, arch=RTX4090()):
    """
    return True if the schedule(shared memory level) is compute intensive
    K: int, the number of columns of the matrix A
    """
    compute_workload = rtile1[0] * rtile1[1] * rtile1[2] * (UDiv(K, rtile1[2]))
    compute_throughput = arch.peak_flops  # / self.arch.compute_max_core[0]
    if False:  # TODO: add tensor core support
        compute_throughput = arch.peak_tc_flops
    compute_latency = compute_workload / compute_throughput

    memory_workload = (
        (rtile1[0] * rtile1[2] + rtile1[1] * rtile1[2] + rtile1[0] * rtile1[1])
        * (UDiv(K, rtile1[2]))
        * byte_size(dtype)
    )  # 4 bytes for fp32, 2 bytes for fp16

    memory_throughput = arch.memory_bw(0)
    memory_latency = memory_workload / memory_throughput
    return compute_latency > memory_latency


def solve_shared(
    smem_size,
    num_regs,
    rtile0,
    type_size=4,
    use_tc=False,
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    count: int = 80,
):
    """
    Solve the shared memory tiling problem for the given shared memory size, number of registers(each threadblock can use), and tile size of level 0(register level).
    Args:
        smem_size: int, the size of the shared memory in bytes
        num_regs: int, the number of registers that can be used by a single threadblock
        rtile0: list of int, the tile size of the register level
        type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        use_tc: bool, whether to use tensor core
        M, N, K: int, the shape of the matrix
        count: int, the number of solutions to find
    Returns:
        list of tuples, each tuple contains the tiling sizes for the shared memory level
    """
    results = []
    
    if use_tc:
        opt = Solver()
        bm, bn, bk = BitVecs("bm bn bk", 64)

        # Limit the tile size for each block
        opt.add(ULE(bm, 512))
        opt.add(ULE(bn, 512))
        opt.add(ULE(bk, 512))

        # Limit the tile size for each block to be a power of two and greater than 0
        for var in [bm, bn, bk]:
            opt.add(UGT(var, 0))
            opt.add(var & (var - 1) == 0)

        # Limit the tile size for each block to be an integer multiple of the tile size at the warp level
        rm, rn, rk = rtile0
        opt.add(bm % rm == 0)
        opt.add(bn % rn == 0)
        opt.add(bk % rk == 0)

        # Limit some combinations of bm, bn, and bk
        opt.add(Or(bk == 8, bk == 16, bk == 32, bk == 64))
        opt.add(
            Or(bm == bn, bm == bn >> 1, bn == bm >> 1)
        )
        opt.add(And(bm >= bk, bn >= bk))

        # Limit the size of shared memory used by each block
        smem: int = smem_size
        offset = 8
        total = (
            (bm * (bk + offset)) + (bn * (bk + offset))
        ) * type_size  # Not sure that whether we should add (bm * bn) to the total
        opt.add(UGT(total, 0))
        opt.add(ULE(total, smem))

        # Limit the size of the grid
        grid_size = UDiv(M, bm) * UDiv(N, bn)
        sm_nums = arch.sm_nums
        opt.add(UGE(grid_size, sm_nums))

        # Limit the number of threads
        num_threads = UDiv(bm, rm) * UDiv(bn, rn) * 32
        opt.add(UGE(num_threads, 32))
        opt.add(ULE(num_threads, 1024))

        for i in range(count):
            if opt.check() == sat:
                m = opt.model()
                res = (m[bm].as_long(), m[bn].as_long(), m[bk].as_long())
                results.append(res)

                # BLOCKING CLAUSE:
                # "Find a new solution where (a,b,c) is NOT the current (a,b,c)"
                opt.add(Or(bm != m[bm], bn != m[bn], bk != m[bk]))
            else:
                break
    else:
        opt = Solver()
        bm, bn, bk = BitVecs("bm bn bk", 64)

        # Limit the tile size for each block
        opt.add(ULE(bm, 512))
        opt.add(ULE(bn, 512))
        opt.add(ULE(bk, 512))

        ########################
        rm, rn, rk = rtile0
        opt.add(Or(bk == 8, bk == 16, bk == 32))
        opt.add(And((UDiv(bm, rm) * UDiv(bn, rn)) % 32 == 0))
        opt.add(Or(bm == bn, bm == bn >> 1, bn == bm >> 1))
        opt.add(And(bm > bk, bn > bk))
        ########################

        opt.add(UGE(bn, bm))
        # opt.add(ULT(bk, 16))

        for var in [bm, bn, bk]:
            opt.add(UGT(var, 0))
            opt.add(var & (var - 1) == 0)
            # opt.add(ULT(var, smem))

        # opt.add(Or(bk == 8, bk == 16, bk == 32))
        rm, rn, rk = rtile0

        opt.add(bm % rm == 0)
        opt.add(bn % rn == 0)
        opt.add(bk % rk == 0)

        # smem = BitVecVal(smem_size, 64)
        smem: Any = smem_size

        # Constraits 1: each threadblock can use at most `num_regs` registers
        num_threads = UDiv(bm, rm) * UDiv(bn, rn)  # number of threads in a threadblock
        opt.add(num_threads % 32 == 0)
        opt.add(ULE(num_threads, 1024))

        # reg_usage_each_thread = (
        #     (rm * rk + rn * rk + rm * rn) * type_size / 4
        # )  # each register can store 4 bytes of data
        reg_usage_each_thread = rm * rn * type_size / 4

        reg_usage_total = num_threads * reg_usage_each_thread
        opt.add(ULE(reg_usage_total, num_regs))
        # opt.add(
        #     And(num_threads % 128 == 0, ULE(num_threads, 1024))
        # )  # Align with hardware: each threadblock has integer multiple of 128(warp_size * compute_sm_partition) threads

        # Constraits 2: the total size of the shared memory used by the threadblock is not too small or too large
        total = (
            (bm * bk) + (bn * bk)
        ) * type_size  # Not sure that whether we should add (bm * bn) to the total
        opt.add(UGT(total, 0))
        opt.add(ULE(total, smem))

        # opt.add(Or(bm == bn, bm == (bn >> 1), bn == (bm >> 1)))
        # opt.add(And(bm > bk, bn > bk))

        #######################

        grid_size = UDiv(M, bm) * UDiv(N, bn)
        sm_nums = arch.sm_nums
        opt.add(UGE(grid_size, sm_nums))

        # print("=== All constraints ===", flush=True)
        # for a in opt.assertions():
        #     print(a, flush=True)

        for i in range(count):
            if opt.check() == sat:
                m = opt.model()
                res = (m[bm].as_long(), m[bn].as_long(), m[bk].as_long())
                results.append(res)

                # BLOCKING CLAUSE:
                # "Find a new solution where (a,b,c) is NOT the current (a,b,c)"
                opt.add(Or(bm != m[bm], bn != m[bn], bk != m[bk]))
            else:
                break
    return results


def solve_flash_attn(
    arch,
    B: int,
    H: int,
    S_q: int,
    S_kv: int,
    D: int,
    type_size: int = 2,
    max_est_regs: int = 320,
    count: int = 256,
):
    """
    Enumerate FlashAttention forward tile configurations.
    Args:
        arch: Arch, the target GPU (uses `max_dyn_smem_per_block`, `sm_nums`)
        B, H, S_q, S_kv, D: the attention shape
        type_size: bytes per input element (2 for fp16)
        max_est_regs: upper bound of the estimated live 32-bit registers per
            thread; a light spill above 255 is tolerated, heavy spills are not
        count: the maximum number of solutions to find
    Returns:
        list of FAConfig
    """
    from kernels.flash_attn_tir import FAConfig

    s = Solver()
    br, bc, nw, st, qs, swz = BitVecs("br bc nw st qs swz", 32)

    s.add(Or(br == 64, br == 128))
    s.add(Or(bc == 32, bc == 64, bc == 128))
    s.add(Or(nw == 4, nw == 8))
    s.add(And(UGE(st, 1), ULE(st, 3)))
    s.add(ULE(qs, 1))
    s.add(ULE(swz, 1))

    # Each warp owns a multiple of 16 Q rows; the tiles must divide the sequence.
    s.add(URem(br, nw * 16) == 0)
    s.add(URem(BitVecVal(S_q, 32), br) == 0)
    s.add(URem(BitVecVal(S_kv, 32), bc) == 0)
    # Do not pipeline deeper than the number of KV tiles.
    s.add(ULE(st, UDiv(BitVecVal(S_kv, 32), bc)))

    # Cooperative 16-byte cp.async loads must evenly cover the tiles.
    threads = nw * 32
    cpr = D // 8
    s.add(URem(br * cpr, threads) == 0)
    s.add(URem(bc * cpr, threads) == 0)

    # Shared memory: Q tile + `stages` K and V tiles (+8 halves of padding per row without swizzle).
    ld = If(swz == 1, BitVecVal(D, 32), BitVecVal(D + 8, 32))
    smem = (br * ld + 2 * st * bc * ld) * type_size
    s.add(ULE(smem, arch.max_dyn_smem_per_block))

    # Registers: O accumulator + S accumulator + P + Q fragments + K/V fragments.
    mt = UDiv(br, nw * 16)
    o_acc = mt * (D // 2)
    s_acc = mt * UDiv(bc, 2)
    p_h = mt * UDiv(bc, 4)
    q_frag = If(qs == 1, mt * 8, mt * (D // 4))
    kv_frag = 8
    regs = o_acc + s_acc + p_h + q_frag + kv_frag + mt * 4 + 32
    s.add(ULE(regs, max_est_regs))

    s.add(ULE(threads, 1024))

    # Enough CTAs to fill every SM at least once.
    grid = UDiv(BitVecVal(S_q, 32), br) * (B * H)
    s.add(UGE(grid, arch.sm_nums))

    results = []
    for _ in range(count):
        if s.check() != sat:
            break
        m = s.model()
        vals = [m[x].as_long() for x in (br, bc, nw, st, qs, swz)]
        results.append(
            FAConfig(
                Br=vals[0], Bc=vals[1], num_warps=vals[2], stages=vals[3],
                q_in_smem=bool(vals[4]), swizzle=bool(vals[5]),
            )
        )
        s.add(Or(br != vals[0], bc != vals[1], nw != vals[2], st != vals[3], qs != vals[4], swz != vals[5]))
    return results


def solve_rowwise(
    arch,
    op: str,
    M: int,
    N: int,
    type_size: int = 2,
    twopass_min_n: int = 16384,
    count: int = 1024,
):
    """
    Enumerate row-wise (Softmax / RMSNorm) kernel configurations.
    Args:
        arch: Arch, the target GPU (uses `sm_nums`, `max_registers_per_sm`)
        op: "softmax" or "rmsnorm"
        M, N: rows and reduced dimension
        type_size: bytes per element (2 for fp16 / bf16)
        twopass_min_n: rows at least this long may also be streamed twice
            (always allowed when no register-cached config fits)
        count: the maximum number of solutions to find
    Returns:
        list of RowConfig
    """
    from kernels.rowwise_tir import RowConfig

    s = Solver()
    tpr, rpb, vec, mode, unroll, hint, chunks = BitVecs("tpr rpb vec mode unroll hint chunks", 32)
    n = BitVecVal(N, 32)

    s.add(Or([tpr == v for v in (32, 64, 128, 256, 512, 1024)]))
    s.add(Or([rpb == v for v in (1, 2, 4, 8, 16)]))
    s.add(Or(vec == 4, vec == 8))
    s.add(ULE(mode, 1))
    s.add(Or(unroll == 1, unroll == 2, unroll == 4))
    # Cache-hinted accesses (8 or 16 bytes).
    s.add(ULE(hint, 1), Implies(hint == 1, UGE(vec * type_size, 8)))

    threads = tpr * rpb
    s.add(UGE(threads, 64), ULE(threads, 1024))
    s.add(URem(n, vec) == 0)
    s.add(URem(BitVecVal(M, 32), rpb) == 0)
    # A row must keep every vector lane of its threads busy at least once.
    s.add(ULE(tpr * vec, n))
    # Enough CTAs to fill every SM at least once.
    s.add(UGE(UDiv(BitVecVal(M, 32), rpb), arch.sm_nums))

    span = tpr * vec
    s.add(chunks == UDiv(n + span - 1, span))
    # Registers: cached 16-bit values (2 per register) + bookkeeping (+ weight chunk).
    cached = If(mode == 0, chunks, unroll)
    regs = cached * UDiv(vec * type_size, 4) + 24 + (UDiv(vec, 2) if op == "rmsnorm" else 0)
    s.add(ULE(regs, 255), ULE(regs * threads, arch.max_registers_per_sm))
    # reg mode is fully unrolled; twopass needs at least one full loop iteration.
    s.add(Implies(mode == 0, And(unroll == 1, ULE(chunks, 32))))
    s.add(Implies(mode == 1, ULE(unroll, UDiv(n, span))))

    reg_fits = Solver()
    reg_fits.add(s.assertions())
    reg_fits.add(mode == 0)
    if N < twopass_min_n and reg_fits.check() == sat:
        s.add(mode == 0)

    results = []
    for _ in range(count):
        if s.check() != sat:
            break
        m = s.model()
        vals = [m[x].as_long() for x in (tpr, rpb, vec, mode, unroll, hint)]
        results.append(
            RowConfig(
                threads_per_row=vals[0], rows_per_block=vals[1], vec=vals[2],
                mode="twopass" if vals[3] else "reg", unroll=vals[4], hint=bool(vals[5]),
            )
        )
        s.add(Or(tpr != vals[0], rpb != vals[1], vec != vals[2], mode != vals[3],
                 unroll != vals[4], hint != vals[5]))
    return results


if __name__ == "__main__":
    print(solve_reg(255, type_size=4, count=20))
    print(solve_shared(49152, 65536, [4, 2, 1], type_size=4, count=40))
