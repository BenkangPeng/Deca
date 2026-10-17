"""
FlashAttention-2 style forward kernel generated as low-level TIR.

The kernel is produced with the TVMScript IRBuilder so that every fragment
loop is unrolled at the Python level (all register indices are compile-time
constants).  It uses `cp.async` multi-stage K/V pipelining, `ldmatrix`,
`mma.sync.m16n8k16` (f16 x f16 -> f32), a register-resident online softmax,
and keeps P in registers (the m16n8 accumulator layout equals the m16n8k16
A-operand layout).

Layout: Q, K, V, O are contiguous [B, H, S, D] float16.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, asdict

import tvm
from tvm import tirx as tir
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import builder as T
from tvm.script import tirx as Tx

from kernels.build import FAST_MATH_OPTIONS, build_kernel, cuda_target  # noqa: F401

LOG2E = 1.4426950408889634
NEG_INF = -1e30


@dataclass(frozen=True)
class FAConfig:
    Br: int = 128  # Q rows per CTA
    Bc: int = 64  # KV rows per pipeline step
    num_warps: int = 4
    stages: int = 1  # K/V shared-memory ring-buffer depth
    q_in_smem: bool = False  # re-read Q fragments from smem every KV step
    swizzle: bool = True  # XOR swizzle (else +8 half padding)

    @property
    def threads(self) -> int:
        return self.num_warps * 32

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        return (
            f"Br={self.Br},Bc={self.Bc},warps={self.num_warps},stages={self.stages},"
            f"q_smem={int(self.q_in_smem)},swz={int(self.swizzle)}"
        )


def smem_bytes(cfg: FAConfig, D: int) -> int:
    ld = D if cfg.swizzle else D + 8
    return (cfg.Br * ld + 2 * cfg.stages * cfg.Bc * ld) * 2


def est_regs_per_thread(cfg: FAConfig, D: int) -> int:
    """Rough 32-bit register estimate of the live fragments."""
    MT = cfg.Br // cfg.num_warps // 16
    o_acc = MT * D // 8 * 4
    s_acc = MT * cfg.Bc // 8 * 4
    p_h = MT * cfg.Bc // 8 * 2
    q_frag = 2 * MT * 4 if cfg.q_in_smem else MT * D // 16 * 4
    kv_frag = 2 * 4  # double-buffered x4 ldmatrix fragments
    return o_acc + s_acc + p_h + q_frag + kv_frag + MT * 4 + 32


def check_config(cfg: FAConfig, S_q: int, S_kv: int, D: int) -> None:
    assert D % 64 == 0, "head dim must be a multiple of 64"
    assert cfg.Br % (16 * cfg.num_warps) == 0
    assert cfg.Bc % 16 == 0
    assert S_q % cfg.Br == 0 and S_kv % cfg.Bc == 0
    cpr = D // 8
    assert cfg.threads % cpr == 0
    for rows in (cfg.Br, cfg.Bc):
        assert (rows * cpr) % cfg.threads == 0
    assert (cfg.Br // cfg.num_warps * cpr) % 32 == 0
    assert cfg.stages >= 1


def make_flash_attn_fwd(
    B: int,
    H: int,
    S_q: int,
    S_kv: int,
    D: int,
    cfg: FAConfig,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tvm.IRModule:
    """`causal` applies the top-left aligned mask (key j visible to query i iff j <= i)."""
    check_config(cfg, S_q, S_kv, D)
    assert not causal or S_q == S_kv, "causal attention requires S_q == S_kv"
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    sl2 = scale * LOG2E

    Br, Bc, NW, ST = cfg.Br, cfg.Bc, cfg.num_warps, cfg.stages
    NT = cfg.threads
    WM = Br // NW
    MT = WM // 16
    NKV = S_kv // Bc
    DK = D // 16  # k-steps of S = Q K^T
    NS = Bc // 8  # n-tiles of S
    DN = D // 8  # n-tiles of O
    CPR = D // 8  # 16-byte chunks per row
    LD = D if cfg.swizzle else D + 8
    Q_OFF = 0
    K_OFF = Br * LD
    V_OFF = K_OFF + ST * Bc * LD
    SMEM = V_OFF + ST * Bc * LD
    WAIT = 2 * ST - 2
    full_mask = tir.const(0xFFFFFFFF, "uint32")

    def soff(base, row, col, row_mod8=None, xor_form=True):
        """Element offset in shared memory; `col` must be a multiple of 8.

        Swizzle: chunk' = chunk ^ (row % 8). Every row starts at a multiple of
        D elements and col < D, so `row_start + chunk' * 8` equals
        `(row_start + (row % 8) * 8) ^ col`: one per-thread base XOR-ed with an
        immediate, which keeps nvcc from hoisting one register per address.
        """
        if cfg.swizzle:
            r8 = row % 8 if row_mod8 is None else row_mod8
            if xor_form:
                return tir.bitwise_xor(base + row * D + r8 * 8, col)
            return base + row * D + tir.bitwise_xor(col // 8, r8) * 8
        return base + row * LD + col

    with IRBuilder() as ib:
        with T.prim_func(s_tir=True):
            T.func_name("main")
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            q = T.arg("q", T.handle())
            k = T.arg("k", T.handle())
            v = T.arg("v", T.handle())
            o = T.arg("o", T.handle())
            Q = IRBuilder.name("Q", T.match_buffer(q, (B, H, S_q, D), "float16"))
            K = IRBuilder.name("K", T.match_buffer(k, (B, H, S_kv, D), "float16"))
            V = IRBuilder.name("V", T.match_buffer(v, (B, H, S_kv, D), "float16"))
            O = IRBuilder.name("O", T.match_buffer(o, (B, H, S_q, D), "float16"))

            bx = T.env_thread("blockIdx.x")
            by = T.env_thread("blockIdx.y")
            tx = T.env_thread("threadIdx.x")
            with T.launch_thread(bx, S_q // Br), T.launch_thread(by, B * H), T.launch_thread(
                tx, NT
            ):
                warp = tx // 32
                lane = tx % 32
                if causal:
                    # Q tiles near the end see the most keys; schedule them first.
                    qt = (S_q // Br - 1) - bx
                    nkv = ((qt + 1) * Br + (Bc - 1)) // Bc
                    min_nkv = (Br + Bc - 1) // Bc
                else:
                    qt, nkv, min_nkv = bx, NKV, NKV
                q_base = by * (S_q * D)
                kv_base = by * (S_kv * D)

                with ExitStack() as stack:

                    def local(name, n, dtype):
                        buf = T.decl_buffer((n,), dtype, scope="local")
                        return IRBuilder.name(name, buf)

                    smem = IRBuilder.name(
                        "smem",
                        T.decl_buffer((SMEM,), "float16", scope="shared.dyn"),
                    )
                    # K/V (and Q when re-read from smem) fragments are double-buffered so
                    # that the ldmatrix of step i+1 is issued before the MMAs of step i.
                    q_frag = local("q_frag", 2 * MT * 8 if cfg.q_in_smem else MT * DK * 8, "float16")
                    k_frag = local("k_frag", 2 * 8, "float16")
                    v_frag = local("v_frag", 2 * 8, "float16")
                    s_acc = local("s_acc", MT * NS * 4, "float32")
                    p_h = local("p_h", MT * NS * 4, "float16")
                    o_acc = local("o_acc", MT * DN * 4, "float32")
                    m_i = local("m_i", MT * 2, "float32")
                    l_i = local("l_i", MT * 2, "float32")
                    tmp = local("tmp", 4, "float32")
                    O_flat = IRBuilder.name(
                        "O_flat",
                        T.decl_buffer((B * H * S_q * D,), "float16", data=O.data),
                    )

                    def store(buf, idx, val):
                        T.buffer_store(buf, val, [idx])

                    def sync():
                        T.evaluate(T.tvm_storage_sync("shared"))

                    def commit():
                        T.evaluate(Tx.ptx.cp.async_.commit_group())

                    def wait(n):
                        T.evaluate(Tx.ptx.cp.async_.wait_group(n))

                    def shfl_xor(x, m):
                        return T.call_extern("float32", "__shfl_xor_sync", full_mask, x, m, 32)

                    def load_tile(G, g_base, row0, rows, s_base):
                        rows_per_it = NT // CPR
                        r = tx // CPR
                        cc = tx % CPR
                        for it in range(rows * CPR // NT):
                            row = it * rows_per_it + r
                            T.evaluate(
                                Tx.s_tir.cp_async_raw.legacy(
                                    smem.data,
                                    soff(s_base, row, cc * 8, row_mod8=(it * rows_per_it + r) % 8),
                                    G.data,
                                    g_base + (row0 + row) * D + cc * 8,
                                    16,
                                    dtype="float16",
                                )
                            )

                    def ldm_q(dst_off, mt, kt):
                        row = warp * WM + mt * 16 + lane % 16
                        col = kt * 16 + (lane // 16) * 8
                        T.evaluate(
                            Tx.ptx_legacy.ldmatrix(
                                False, 4, ".b16", q_frag.data, dst_off, smem.data,
                                soff(Q_OFF, row, col, row_mod8=lane % 8), dtype="float16",
                            )
                        )

                    def mma(a, aoff, b, boff, c, coff):
                        T.evaluate(
                            Tx.ptx_legacy.mma(
                                "m16n8k16", "row", "col", "fp16", "fp16", "fp32",
                                a.data, aoff, b.data, boff, c.data, coff, False,
                                dtype="float32",
                            )
                        )

                    def slot(j):
                        return 0 if ST == 1 else (j % ST)

                    def run_pipelined(steps, preloaded=False):
                        """steps: [(load(buf), compute(buf))]; buffers alternate 0/1.

                        TVM emits ldmatrix/mma as volatile asm that nvcc keeps in
                        program order, so the software pipeline is spelled out here.
                        """
                        if not preloaded:
                            steps[0][0](0)
                        for i, (_, compute) in enumerate(steps):
                            if i + 1 < len(steps):
                                steps[i + 1][0]((i + 1) % 2)
                            compute(i % 2)

                    def qk_steps(k_base):
                        steps = []
                        for kt in range(DK):
                            for p in range(NS // 2):

                                def load(buf, kt=kt, p=p):
                                    if cfg.q_in_smem and p == 0:
                                        for mt in range(MT):
                                            ldm_q(((kt % 2) * MT + mt) * 8, mt, kt)
                                    row = p * 16 + (lane // 16) * 8 + lane % 8
                                    col = kt * 16 + ((lane // 8) % 2) * 8
                                    T.evaluate(
                                        Tx.ptx_legacy.ldmatrix(
                                            False, 4, ".b16", k_frag.data, buf * 8, smem.data,
                                            soff(k_base, row, col, row_mod8=lane % 8),
                                            dtype="float16",
                                        )
                                    )

                                def compute(buf, kt=kt, p=p):
                                    for mt in range(MT):
                                        if cfg.q_in_smem:
                                            qoff = ((kt % 2) * MT + mt) * 8
                                        else:
                                            qoff = (mt * DK + kt) * 8
                                        for n in range(2):
                                            mma(
                                                q_frag, qoff, k_frag, buf * 8 + n * 4,
                                                s_acc, (mt * NS + 2 * p + n) * 4,
                                            )

                                steps.append((load, compute))
                        return steps

                    def pv_steps(v_base):
                        steps = []
                        for kk in range(Bc // 16):
                            for p in range(DN // 2):

                                def load(buf, kk=kk, p=p):
                                    mi = lane // 8
                                    row = kk * 16 + (mi % 2) * 8 + lane % 8
                                    col = p * 16 + (mi // 2) * 8
                                    T.evaluate(
                                        Tx.ptx_legacy.ldmatrix(
                                            True, 4, ".b16", v_frag.data, buf * 8, smem.data,
                                            soff(v_base, row, col, row_mod8=lane % 8),
                                            dtype="float16",
                                        )
                                    )

                                def compute(buf, kk=kk, p=p):
                                    for mt in range(MT):
                                        for n in range(2):
                                            mma(
                                                p_h, (mt * NS + 2 * kk) * 4, v_frag, buf * 8 + n * 4,
                                                o_acc, (mt * DN + 2 * p + n) * 4,
                                            )

                                steps.append((load, compute))
                        return steps

                    # ---------------- prologue ----------------
                    load_tile(Q, q_base, qt * Br, Br, Q_OFF)
                    commit()

                    def guarded(s, fn):
                        if s < min_nkv:
                            fn()
                        elif s < NKV:
                            with T.If(s < nkv):
                                with T.Then():
                                    fn()

                    for s in range(ST):
                        guarded(s, lambda s=s: load_tile(K, kv_base, s * Bc, Bc, K_OFF + s * Bc * LD))
                        commit()
                        guarded(s, lambda s=s: load_tile(V, kv_base, s * Bc, Bc, V_OFF + s * Bc * LD))
                        commit()

                    for i in range(MT * DN * 4):
                        store(o_acc, i, T.float32(0))
                    for i in range(MT * 2):
                        store(m_i, i, T.float32(NEG_INF))
                        store(l_i, i, T.float32(0))

                    wait(2 * ST)
                    sync()
                    if not cfg.q_in_smem:
                        for mt in range(MT):
                            for kt in range(DK):
                                ldm_q((mt * DK + kt) * 8, mt, kt)

                    # ---------------- main loop ----------------
                    with T.serial(0, nkv) as j:
                        wait(WAIT)
                        sync()
                        with T.If(j >= 1):
                            with T.Then():
                                with T.If(j + ST - 1 < nkv):
                                    with T.Then():
                                        load_tile(
                                            V, kv_base, (j + ST - 1) * Bc, Bc,
                                            V_OFF + slot(j + ST - 1) * (Bc * LD),
                                        )
                                commit()

                        k_base = K_OFF + slot(j) * (Bc * LD)
                        for i in range(MT * NS * 4):
                            store(s_acc, i, T.float32(0))
                        run_pipelined(qk_steps(k_base))

                        wait(WAIT)
                        sync()
                        with T.If(j + ST < nkv):
                            with T.Then():
                                load_tile(K, kv_base, (j + ST) * Bc, Bc, k_base)
                        commit()
                        v_base = V_OFF + slot(j) * (Bc * LD)
                        pv = pv_steps(v_base)
                        pv[0][0](0)  # first V fragment load overlaps the softmax

                        if causal:
                            # Only tiles crossing the diagonal need masking. Each row
                            # sees key 0 in tile 0, so its running max is finite
                            # before any fully masked row-tile shows up.
                            with T.If((j + 1) * Bc - 1 > qt * Br):
                                with T.Then():
                                    # col - row of element (mt, h, nt, c) is
                                    # diff + nt * 8 + c - mt * 16 - h * 8
                                    diff = j * Bc + (lane % 4) * 2 - (qt * Br + warp * WM + lane // 4)
                                    for mt in range(MT):
                                        for h in range(2):
                                            for nt in range(NS):
                                                for c in range(2):
                                                    idx = (mt * NS + nt) * 4 + 2 * h + c
                                                    store(
                                                        s_acc, idx,
                                                        T.Select(
                                                            diff > mt * 16 + h * 8 - nt * 8 - c,
                                                            T.float32(NEG_INF),
                                                            s_acc[idx],
                                                        ),
                                                    )

                        # ---------------- online softmax ----------------
                        for mt in range(MT):
                            for h in range(2):
                                r = mt * 2 + h
                                vals = [
                                    s_acc[(mt * NS + nt) * 4 + 2 * h + c]
                                    for nt in range(NS)
                                    for c in range(2)
                                ]
                                while len(vals) > 1:
                                    vals = [
                                        T.max(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)
                                    ]
                                store(tmp, 0, vals[0])
                                store(tmp, 0, T.max(tmp[0], shfl_xor(tmp[0], 1)))
                                store(tmp, 0, T.max(tmp[0], shfl_xor(tmp[0], 2)))
                                store(tmp, 0, T.max(tmp[0], m_i[r]))  # m_new
                                store(tmp, 1, T.exp2((m_i[r] - tmp[0]) * T.float32(sl2)))  # alpha
                                store(m_i, r, tmp[0])
                                store(tmp, 2, tmp[0] * T.float32(sl2))
                                sums = []
                                for nt in range(NS):
                                    for c in range(2):
                                        idx = (mt * NS + nt) * 4 + 2 * h + c
                                        store(
                                            s_acc, idx,
                                            T.exp2(s_acc[idx] * T.float32(sl2) - tmp[2]),
                                        )
                                        store(p_h, idx, T.Cast("float16", s_acc[idx]))
                                        sums.append(s_acc[idx])
                                while len(sums) > 1:
                                    sums = [sums[i] + sums[i + 1] for i in range(0, len(sums), 2)]
                                store(l_i, r, l_i[r] * tmp[1] + sums[0])
                                for dn in range(DN):
                                    for c in range(2):
                                        idx = (mt * DN + dn) * 4 + 2 * h + c
                                        store(o_acc, idx, o_acc[idx] * tmp[1])

                        # ---------------- O += P V ----------------
                        run_pipelined(pv, preloaded=True)

                    # ---------------- epilogue ----------------
                    for mt in range(MT):
                        for h in range(2):
                            r = mt * 2 + h
                            store(tmp, 0, l_i[r] + shfl_xor(l_i[r], 1))
                            store(tmp, 0, tmp[0] + shfl_xor(tmp[0], 2))
                            store(tmp, 0, T.float32(1.0) / tmp[0])
                            row = warp * WM + mt * 16 + h * 8 + lane // 4
                            for dn in range(DN):
                                for c in range(2):
                                    idx = (mt * DN + dn) * 4 + 2 * h + c
                                    store(
                                        smem,
                                        soff(Q_OFF, row, dn * 8, row_mod8=(lane // 4) % 8)
                                        + (lane % 4) * 2 + c,
                                        T.Cast("float16", o_acc[idx] * tmp[0]),
                                    )
                    T.evaluate(T.tvm_storage_sync("warp"))
                    for it in range(WM * CPR // 32):
                        ch = it * 32 + lane
                        row_l = ch // CPR
                        cc = ch % CPR
                        row = warp * WM + row_l
                        T.buffer_store(
                            O_flat,
                            smem[
                                tir.Ramp(
                                    soff(Q_OFF, row, cc * 8, row_mod8=row_l % 8, xor_form=False),
                                    1, 8,
                                )
                            ],
                            [tir.Ramp(q_base + (qt * Br + row) * D + cc * 8, 1, 8)],
                        )

    func = ib.get()
    return tvm.IRModule({"main": func})


FA_NVCC_OPTIONS = FAST_MATH_OPTIONS


def build_flash_attn_to_file(
    shape, cfg: FAConfig, softmax_scale, arch: str, path: str, causal: bool = False
) -> str:
    """Builds in a worker process (no CUDA context needed) and exports a shared library."""
    B, H, S_q, S_kv, D = shape
    mod = make_flash_attn_fwd(B, H, S_q, S_kv, D, cfg, softmax_scale, causal)
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    build_flash_attn(mod, target).export_library(path)
    return path


def build_flash_attn(
    mod: tvm.IRModule,
    target: tvm.target.Target | None = None,
    dump_dir: str | None = None,
):
    """Compiles the kernel; `dump_dir` keeps the generated .cu for inspection."""
    return build_kernel(mod, target, dump_dir, FA_NVCC_OPTIONS)
