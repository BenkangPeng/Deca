from z3 import *
from arch.RTX4090 import RTX4090

# from utils.helper import IsComputeIntensive


def solve_reg(num_regs, type_size=4, count=20):
    """
    Solve the register tiling problem for the given number of registers.
    Args:
        num_regs: int, the number of registers that can be used by a single thread
        type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        count: int, the number of solutions to find
    Returns:
        list of tuples, each tuple contains the tiling sizes for the register level
    """
    s = Solver()
    # Each thread computes a tile of size rm x rn x rk
    rm, rn, rk = BitVecs("rm rn rk", 64)

    m_bv = BitVecVal(num_regs, 64)
    # reduce dimension to 1
    s.add(rk == 1)
    for var in [rm, rn, rk]:
        s.add(UGT(var, 0))
        s.add(var & (var - 1) == 0)

    used_reg = (
        (rm * rk + rn * rk + rm * rn) * type_size / 4
    )  # Each register can store 4 bytes of data
    # used_reg > 0 && used_reg <= M
    s.add(ULE(used_reg, m_bv))
    s.add(UGT(used_reg, 0))
    s.add(Or(rm == rn, rm == rn >> 1, rn == rm >> 1))

    results = []
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


def IsComputeIntensive(rtile1, K, arch=RTX4090()):
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
        * 4
    )  # 4 bytes for fp32

    memory_throughput = arch.memory_bw(0)
    memory_latency = memory_workload / memory_throughput
    return compute_latency > memory_latency


def solve_shared(smem_size, num_regs, rtile0, type_size=4, count=5):
    """
    Solve the shared memory tiling problem for the given shared memory size, number of registers(each threadblock can use), and tile size of level 0(register level).
    Args:
        smem_size: int, the size of the shared memory in bytes
        num_regs: int, the number of registers that can be used by a single threadblock
        rtile0: list of int, the tile size of the register level
        type_size: int, the size of the data type in bytes, 4 bytes for fp32, 2 bytes for fp16
        count: int, the number of solutions to find
    Returns:
        list of tuples, each tuple contains the tiling sizes for the shared memory level
    """
    s = Solver()

    bm, bn, bk = BitVecs("bm bn bk", 64)
    s.add(Or(bk == 8, bk == 16, bk == 32))
    rm, rn, rk = rtile0
    smem = BitVecVal(smem_size, 64)

    # Constraits 1: each threadblock can use at most `num_regs` registers
    num_threads = UDiv(bm, rm) * UDiv(bn, rn)  # number of threads in a threadblock
    reg_usage_each_thread = (
        (rm * rk + rn * rk + rm * rn) * type_size / 4
    )  # each register can store 4 bytes of data
    reg_usage_total = num_threads * reg_usage_each_thread
    s.add(ULE(reg_usage_total, num_regs))
    s.add(
        And(num_threads % 128 == 0, ULE(num_threads, 1024))
    )  # Align with hardware: each threadblock has integer multiple of 128(warp_size * compute_sm_partition) threads

    for var in [bm, bn, bk]:
        s.add(UGT(var, 0))
        s.add(var & (var - 1) == 0)
        s.add(ULT(var, smem))

    # Constraits 2: the total size of the shared memory used by the threadblock is not too small or too large
    total = (
        (bm * bk) + (bn * bk)
    ) * type_size  # Not sure that whether we should add (bm * bn) to the total
    s.add(UGT(total, 0))
    s.add(ULE(total, smem))

    s.add(Or(bm == bn, bm == (bn >> 1), bn == (bm >> 1)))
    s.add(And(bm > bk, bn > bk))

    results = []
    for i in range(count):
        if s.check() == sat:
            m = s.model()
            res = (m[bm].as_long(), m[bn].as_long(), m[bk].as_long())
            results.append(res)

            # BLOCKING CLAUSE:
            # "Find a new solution where (a,b,c) is NOT the current (a,b,c)"
            s.add(Or(bm != m[bm], bn != m[bn], bk != m[bk]))
        else:
            break
    return results


if __name__ == "__main__":
    print(solve_reg(255, type_size=4, count=20))
    print(solve_shared(49152, 65536, [4, 2, 1], type_size=4, count=40))
