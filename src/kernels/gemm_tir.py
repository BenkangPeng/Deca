"""
FP16 Tensor Core GEMM (C = A @ B, fp32 or fp16 accumulation) generated as low-level TIR.

Like the FlashAttention kernel, the kernel is produced with the TVMScript
IRBuilder and every fragment loop is unrolled at the Python level.  It uses a
`cp.async` multi-stage shared-memory ring buffer, `ldmatrix` (`.trans` for the
row-major B operand), `mma.sync.m16n8k16` (f16 x f16 -> f32) with
double-buffered register fragments, an XOR-swizzled shared-memory layout,
grouped CTA rasterization for L2 reuse and an optional shared-memory staged
epilogue with 16-byte coalesced stores.

Layout: A[M, K], B[K, N], C[M, N], all row-major float16.

Shared-memory swizzle: a tile is viewed as 128-byte lines of eight 16-byte
chunks and chunk' = chunk ^ (line % 8).  A rows are BK * 2 = 64 or 128 bytes
(one line holds 2 or 1 rows), B rows are BN * 2 >= 128 bytes and are swizzled
per row (key = row % 8).  Both the cp.async writes and the 8 rows of every
ldmatrix 8x8 matrix then touch 8 distinct bank groups.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, asdict

import tvm
from tvm import tirx as tir
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import builder as T
from tvm.script import tirx as Tx

from kernels.build import FAST_MATH_OPTIONS, build_kernel

MAX_REGS = 255
# Opt-in dynamic shared memory per block on sm_80+ GeForce / sm_89 (99 KB).
MAX_DYN_SMEM = 101376
# Padding (halves) of the epilogue staging rows: keeps the fragment writes of
# the 8 rows of an m16n8 tile in distinct banks.
EPI_PAD = 8


@dataclass(frozen=True)
class GemmConfig:
    BM: int = 128  # CTA tile rows
    BN: int = 256  # CTA tile columns
    BK: int = 32  # K depth of one pipeline stage
    warp_m: int = 64  # warp tile rows
    warp_n: int = 64  # warp tile columns
    stages: int = 3  # shared-memory ring-buffer depth (>= 2)
    raster: int = 8  # CTA rows per rasterization group (1: plain row-major order)
    epi_smem: bool = True  # stage C through shared memory for 16-byte stores
    # fp16 accumulation (mma f16.f16.f16.f16, as cublasHgemm): twice the MMA rate
    # on GeForce GPUs and half the accumulator registers, at a lower accuracy.
    acc_f16: bool = False
    # fp16 accumulators alternated over the k16 steps and summed in fp32 by the
    # epilogue; 2 halves both the partial-sum magnitude and the roundings per
    # accumulator (about cuBLAS accuracy at K = 16K) for twice the registers.
    acc_splits: int = 1

    @property
    def num_warps(self) -> int:
        return (self.BM // self.warp_m) * (self.BN // self.warp_n)

    @property
    def threads(self) -> int:
        return self.num_warps * 32

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return (
            f"{self.BM}x{self.BN}x{self.BK},warp={self.warp_m}x{self.warp_n},"
            f"stages={self.stages},raster={self.raster},epi_smem={int(self.epi_smem)}"
            + (f",acc=f16x{self.acc_splits}" if self.acc_f16 else "")
        )


def pipeline_smem_bytes(cfg: GemmConfig) -> int:
    return cfg.stages * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 2


def epilogue_smem_bytes(cfg: GemmConfig) -> int:
    return cfg.BM * (cfg.BN + EPI_PAD) * 2 if cfg.epi_smem else 0


def smem_bytes(cfg: GemmConfig) -> int:
    """The epilogue reuses the pipeline buffers once the main loop is done."""
    return max(pipeline_smem_bytes(cfg), epilogue_smem_bytes(cfg))


def est_regs_per_thread(cfg: GemmConfig) -> int:
    """Rough 32-bit register estimate of the live fragments."""
    MT, NT8 = cfg.warp_m // 16, cfg.warp_n // 8
    # 4 accumulators per n8 tile (two half2 registers in fp16)
    acc = MT * NT8 * (2 * cfg.acc_splits if cfg.acc_f16 else 4)
    frags = 2 * (MT * 4 + NT8 // 2 * 4)  # double-buffered A / B fragments
    return acc + frags + 24  # addressing, loop and pipeline bookkeeping


def check_config(
    cfg: GemmConfig, M: int, N: int, K: int, max_smem: int = MAX_DYN_SMEM
) -> None:
    assert cfg.stages >= 2
    assert cfg.acc_splits in (1, 2) and (cfg.acc_f16 or cfg.acc_splits == 1)
    assert smem_bytes(cfg) <= max_smem
    assert cfg.BK in (32, 64)
    assert cfg.warp_m % 16 == 0 and cfg.warp_n % 16 == 0
    assert cfg.BM % cfg.warp_m == 0 and cfg.BN % cfg.warp_n == 0
    assert cfg.BN >= 64  # B rows span whole 128-byte lines
    assert M % cfg.BM == 0 and N % cfg.BN == 0 and K % cfg.BK == 0
    assert 64 <= cfg.threads <= 1024
    # Every thread issues the same number of 16-byte cp.async for each operand.
    assert (cfg.BM * cfg.BK // 8) % cfg.threads == 0
    assert (cfg.BK * cfg.BN // 8) % cfg.threads == 0
    assert (M // cfg.BM) % cfg.raster == 0
    assert K // cfg.BK >= cfg.stages
    if cfg.epi_smem:
        assert (cfg.BM * cfg.BN // 8) % cfg.threads == 0
    assert est_regs_per_thread(cfg) <= MAX_REGS


def make_gemm_tc(M: int, N: int, K: int, cfg: GemmConfig) -> tvm.IRModule:
    check_config(cfg, M, N, K)
    BM, BN, BK, ST = cfg.BM, cfg.BN, cfg.BK, cfg.stages
    WM, WN = cfg.warp_m, cfg.warp_n
    NWN = BN // WN
    NT = cfg.threads
    MT, NT8 = WM // 16, WN // 8  # m16 / n8 tiles per warp
    NP = NT8 // 2  # ldmatrix.x4.trans loads per k-step (two n8 tiles each)
    KS = BK // 16  # k-steps per stage (even, so a stage starts on fragment buffer 0)
    NK = K // BK
    GM, GN = M // BM, N // BN
    CPR_A, CPR_B = BK // 8, BN // 8  # 16-byte chunks per smem row
    RPL_A = 8 // CPR_A  # A rows per 128-byte line
    ACC, ACC_PTX = ("float16", "fp16") if cfg.acc_f16 else ("float32", "fp32")
    SPL = cfg.acc_splits
    A_STAGE, B_STAGE = BM * BK, BK * BN
    B_OFF = ST * A_STAGE
    SMEM = smem_bytes(cfg) // 2
    WAIT = ST - 2

    with IRBuilder() as ib:
        with T.prim_func(s_tir=True):
            T.func_name("main")
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            a = T.arg("a", T.handle())
            b = T.arg("b", T.handle())
            c = T.arg("c", T.handle())
            A = IRBuilder.name("A", T.match_buffer(a, (M, K), "float16"))
            B = IRBuilder.name("B", T.match_buffer(b, (K, N), "float16"))
            C = IRBuilder.name("C", T.match_buffer(c, (M, N), "float16"))

            bx = T.env_thread("blockIdx.x")
            tx = T.env_thread("threadIdx.x")
            with T.launch_thread(bx, GM * GN), T.launch_thread(tx, NT):
                warp = tx // 32
                lane = tx % 32
                # Grouped rasterization: `raster` CTA rows sweep the CTA columns
                # together so that their A rows and B columns stay in L2.
                group = cfg.raster * GN
                bm = (bx // group) * cfg.raster + (bx % group) % cfg.raster
                bn = (bx % group) // cfg.raster
                wm0 = (warp // NWN) * WM
                wn0 = (warp % NWN) * WN

                with ExitStack() as stack:

                    def local(name, n, dtype):
                        buf = T.decl_buffer((n,), dtype, scope="local")
                        return IRBuilder.name(name, buf)

                    smem = IRBuilder.name(
                        "smem",
                        T.decl_buffer((SMEM,), "float16", scope="shared.dyn"),
                    )
                    a_frag = local("a_frag", 2 * MT * 8, "float16")
                    b_frag = local("b_frag", 2 * NP * 8, "float16")
                    # m16n8 accumulator tile: 4 values per thread in the same order
                    # for fp32 and fp16 (the latter packed as two half2 registers).
                    acc = local("acc", SPL * MT * NT8 * 4, ACC)
                    C_flat = IRBuilder.name(
                        "C_flat",
                        T.decl_buffer((M * N,), "float16", data=C.data),
                    )

                    def sync():
                        T.evaluate(T.tvm_storage_sync("shared"))

                    def commit():
                        T.evaluate(Tx.ptx.cp.async_.commit_group())

                    def wait(n):
                        T.evaluate(Tx.ptx.cp.async_.wait_group(n))

                    def cp_async(s_off, G, g_off):
                        T.evaluate(
                            Tx.s_tir.cp_async_raw.legacy(smem.data, s_off, G.data, g_off, 16, dtype="float16")
                        )

                    # ---------------- global -> shared (cp.async) ----------------
                    # A: chunk `tx` of every NT-chunk slab. NT / 8 is a multiple of
                    # 8 lines, so the line key only depends on tx.
                    a_line = tx // 8
                    a_st_off = a_line * 64 + tir.bitwise_xor(tx % 8, a_line % 8) * 8
                    a_g_row = bm * BM + tx // CPR_A
                    a_g_col = (tx % CPR_A) * 8

                    # B: NT is a multiple of CPR_B (both powers of two, CPR_B <= 32).
                    b_rpi = NT // CPR_B
                    b_ch = tx % CPR_B
                    b_g_col = bn * BN + b_ch * 8

                    def load_stage(kt, slot):
                        for it in range(BM * CPR_A // NT):
                            cp_async(
                                slot * A_STAGE + it * NT * 8 + a_st_off,
                                A,
                                (a_g_row + it * (NT // CPR_A)) * K + kt * BK + a_g_col,
                            )
                        for it in range(BK * CPR_B // NT):
                            row = it * b_rpi + tx // CPR_B
                            off = (
                                row * BN
                                + (b_ch // 8) * 64
                                + tir.bitwise_xor(b_ch % 8, row % 8) * 8
                            )
                            cp_async(B_OFF + slot * B_STAGE + off, B, (kt * BK + row) * N + b_g_col)

                    # ---------------- shared -> registers (ldmatrix) ----------------
                    # Per-thread swizzled bases; the per-access parts (k-step,
                    # m / n tile) are immediates XOR-ed / added to them, so nvcc
                    # keeps only a handful of address registers.
                    #
                    # A (x4): row = wm0 + mt*16 + lane%16, chunk = 2*ks + lane//16.
                    # 16 rows are whole lines (key unchanged) and 2*ks only sets
                    # bits that are zero in the per-thread sub-chunk.
                    a_row = wm0 + lane % 16
                    a_ln = a_row // RPL_A
                    a_sub = (a_row % RPL_A) * CPR_A + lane // 16
                    a_ld_base = a_ln * 64 + tir.bitwise_xor(a_sub, a_ln % 8) * 8

                    def a_frag_off(mt, ks):
                        return tir.bitwise_xor(a_ld_base, ks * 16) + mt * 16 * BK

                    # B (x4.trans): row = ks*16 + (mi%2)*8 + lane%8,
                    # chunk = wn0/8 + 2p + mi//2 with mi = lane // 8; wn0/8 is a
                    # multiple of NT8 > 2p, so 2p only sets free chunk bits.
                    b_mi = lane // 8
                    b_row = (b_mi % 2) * 8 + lane % 8
                    b_c0 = wn0 // 8 + b_mi // 2
                    b_ld_base = (
                        b_row * BN + (b_c0 // 8) * 64 + tir.bitwise_xor(b_c0 % 8, lane % 8) * 8
                    )

                    def b_frag_off(ks, p):
                        line_add, sub_add = divmod(2 * p, 8)
                        return (
                            tir.bitwise_xor(b_ld_base, sub_add * 8) + line_add * 64 + ks * 16 * BN
                        )

                    def load_frags(slot, ks, buf):
                        for mt in range(MT):
                            T.evaluate(
                                Tx.ptx_legacy.ldmatrix(
                                    False, 4, ".b16", a_frag.data, (buf * MT + mt) * 8,
                                    smem.data, slot * A_STAGE + a_frag_off(mt, ks),
                                    dtype="float16",
                                )
                            )
                        for p in range(NP):
                            T.evaluate(
                                Tx.ptx_legacy.ldmatrix(
                                    True, 4, ".b16", b_frag.data, (buf * NP + p) * 8,
                                    smem.data, B_OFF + slot * B_STAGE + b_frag_off(ks, p),
                                    dtype="float16",
                                )
                            )

                    def mma(buf, mt, j):
                        T.evaluate(
                            Tx.ptx_legacy.mma(
                                "m16n8k16", "row", "col", "fp16", "fp16", ACC_PTX,
                                a_frag.data, (buf * MT + mt) * 8,
                                b_frag.data, (buf * NP + j // 2) * 8 + (j % 2) * 4,
                                # k-steps alternate between the SPL accumulator sets
                                acc.data, (((buf % SPL) * MT + mt) * NT8 + j) * 4,
                                False, dtype=ACC,
                            )
                        )

                    def compute(buf):
                        # Serpentine order over n: consecutive MMAs share an operand.
                        for mt in range(MT):
                            js = range(NT8) if mt % 2 == 0 else range(NT8 - 1, -1, -1)
                            for j in js:
                                mma(buf, mt, j)

                    def stage_body(kt, slot):
                        """One BK step on stage `slot` (a Python int) at k-tile `kt`."""
                        for ks in range(KS):
                            buf = ks % 2
                            if ks == 0:
                                # Refill the slot consumed by the previous stage.
                                with T.If(kt + ST - 1 < NK):
                                    with T.Then():
                                        load_stage(kt + ST - 1, (slot + ST - 1) % ST)
                                commit()
                            if ks + 1 < KS:
                                load_frags(slot, ks + 1, 1 - buf)
                            else:
                                # The next stage must have landed, and every warp
                                # must be done with this one before it is refilled.
                                wait(WAIT)
                                sync()
                                load_frags((slot + 1) % ST, 0, 1 - buf)
                            compute(buf)

                    # ---------------- prologue ----------------
                    for s in range(ST - 1):
                        load_stage(s, s)
                        commit()
                    for i in range(SPL * MT * NT8 * 4):
                        T.buffer_store(acc, tir.const(0, ACC), [i])
                    wait(WAIT)
                    sync()
                    load_frags(0, 0, 0)

                    # ------- main loop, unrolled by ST so that the slots are static -------
                    with T.serial(0, NK // ST) as it:
                        for s in range(ST):
                            stage_body(it * ST + s, s)
                    for s in range(NK % ST):
                        stage_body((NK // ST) * ST + s, s)

                    # ---------------- epilogue ----------------
                    def acc_val(mt, nt, h, cc):
                        idx = (mt * NT8 + nt) * 4 + 2 * h + cc
                        if SPL == 2:
                            # Sum the two fp16 partial accumulators in fp32.
                            other = acc[MT * NT8 * 4 + idx]
                            return T.Cast(
                                "float16",
                                T.Cast("float32", acc[idx]) + T.Cast("float32", other),
                            )
                        return acc[idx] if cfg.acc_f16 else T.Cast("float16", acc[idx])

                    if cfg.epi_smem:
                        wait(0)
                        sync()
                        LD = BN + EPI_PAD
                        s_off0 = (wm0 + lane // 4) * LD + wn0 + (lane % 4) * 2
                        for mt in range(MT):
                            for nt in range(NT8):
                                for h in range(2):
                                    for cc in range(2):
                                        T.buffer_store(
                                            smem,
                                            acc_val(mt, nt, h, cc),
                                            [s_off0 + (mt * 16 + h * 8) * LD + nt * 8 + cc],
                                        )
                        sync()
                        CPR_C = BN // 8
                        c_row = tx // CPR_C
                        c_ch = tx % CPR_C
                        for it in range(BM * CPR_C // NT):
                            row = it * (NT // CPR_C) + c_row
                            T.buffer_store(
                                C_flat,
                                smem[tir.Ramp(row * LD + c_ch * 8, 1, 8)],
                                [tir.Ramp((bm * BM + row) * N + bn * BN + c_ch * 8, 1, 8)],
                            )
                    else:
                        g_off0 = (bm * BM + wm0 + lane // 4) * N + bn * BN + wn0 + (lane % 4) * 2
                        for mt in range(MT):
                            for nt in range(NT8):
                                for h in range(2):
                                    for cc in range(2):
                                        T.buffer_store(
                                            C_flat,
                                            acc_val(mt, nt, h, cc),
                                            [g_off0 + (mt * 16 + h * 8) * N + nt * 8 + cc],
                                        )

    func = ib.get()
    return tvm.IRModule({"main": func})


GEMM_NVCC_OPTIONS = FAST_MATH_OPTIONS


def build_gemm_tc(
    mod: tvm.IRModule,
    target: tvm.target.Target | None = None,
    dump_dir: str | None = None,
):
    """Compiles the kernel; `dump_dir` keeps the generated .cu for inspection."""
    return build_kernel(mod, target, dump_dir, GEMM_NVCC_OPTIONS)


def build_gemm_tc_to_file(shape, cfg: GemmConfig, arch: str, path: str) -> str:
    """Builds in a worker process (no CUDA context needed) and exports a shared library."""
    M, N, K = shape
    mod = make_gemm_tc(M, N, K, cfg)
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    build_gemm_tc(mod, target).export_library(path)
    return path
