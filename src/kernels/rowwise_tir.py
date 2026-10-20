"""
Row-wise reduction kernels (Softmax over the last dim, RMSNorm) generated as
low-level TIR with the TVMScript IRBuilder.

Every row of the [M, N] input is handled by `threads_per_row` threads (a
multiple of a warp); `rows_per_block` rows share a CTA.  Each thread owns the
16-byte (or 8-byte) chunks `(c * threads_per_row + t) * vec` of its row, so a
warp always touches contiguous 512 B (256 B) segments.

Modes:
- reg:     the whole row is cached in registers (in the input dtype), so the
           row is read from DRAM exactly once and written once.
- twopass: rows too long for registers are streamed twice: pass 1 computes an
           online (max, sum) for softmax / a sum of squares for RMSNorm,
           pass 2 re-reads the row (mostly from L2) and writes the output.

Reductions: warp `__shfl_xor_sync`, then a shared-memory exchange between the
warps of one row.  All math is in fp32.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import tvm
from tvm import tir
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T

from kernels.build import FAST_MATH_OPTIONS, build_kernel

LOG2E = 1.4426950408889634
NEG_INF = -1e30
OPS = ("softmax", "rmsnorm")
DTYPES = ("float16", "bfloat16")
MODES = ("reg", "twopass")
MAX_REGS_PER_SM = 65536


@dataclass(frozen=True)
class RowConfig:
    threads_per_row: int = 128
    rows_per_block: int = 1
    vec: int = 8  # elements per vector access (8 -> 16 B for 16-bit dtypes)
    mode: str = "reg"
    unroll: int = 1  # chunks loaded back to back per twopass iteration
    hint: bool = False  # L2 eviction-priority loads and streaming stores

    @property
    def threads(self) -> int:
        return self.threads_per_row * self.rows_per_block

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        s = f"tpr={self.threads_per_row},rpb={self.rows_per_block},vec={self.vec},mode={self.mode}"
        s += f",unroll={self.unroll}" if self.mode == "twopass" else ""
        return s + (",hint=1" if self.hint else "")


def num_chunks(cfg: RowConfig, N: int) -> int:
    return -(-N // (cfg.threads_per_row * cfg.vec))


def est_regs_per_thread(cfg: RowConfig, N: int, op: str) -> int:
    """32-bit register estimate of the `E` cached elements per thread + bookkeeping.

    Fitted to ptxas (sm_90, 16-bit dtypes): nvcc keeps part of the row as fp32
    across the reduction phases, about 0.75 registers per element. A row owned
    by a single warp has no block barrier between the phases, so the fp32
    copies of the whole row stay live and the usage grows to E + E^2 / 256
    (e.g. 80 registers at E = 64, 192 at E = 128 instead of 64 and 88).
    """
    cached = num_chunks(cfg, N) if cfg.mode == "reg" else cfg.unroll
    elems = cached * cfg.vec
    if cfg.threads_per_row == 32:
        regs = elems + elems * elems // 256 + 16
    else:
        regs = elems * 3 // 4 + 16
    if op == "rmsnorm":
        regs += cfg.vec // 2
    return regs


def check_config(cfg: RowConfig, op: str, M: int, N: int, dtype: str) -> None:
    assert op in OPS and dtype in DTYPES and cfg.mode in MODES
    assert cfg.vec in (2, 4, 8) and N % cfg.vec == 0
    assert not cfg.hint or cfg.vec in (4, 8), "cache hints need 8- or 16-byte accesses"
    assert cfg.threads_per_row % 32 == 0 and cfg.threads <= 1024
    assert M % cfg.rows_per_block == 0
    assert cfg.unroll >= 1
    regs = est_regs_per_thread(cfg, N, op)
    assert regs <= 255 and regs * cfg.threads <= MAX_REGS_PER_SM


def make_rowwise(
    op: str, M: int, N: int, dtype: str, cfg: RowConfig, eps: float = 1e-6
) -> tvm.IRModule:
    """Softmax: y = softmax(x, -1).  RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w."""
    check_config(cfg, op, M, N, dtype)
    TPR, RPB, VEC = cfg.threads_per_row, cfg.rows_per_block, cfg.vec
    NT = cfg.threads
    NW = TPR // 32  # warps per row
    SPAN = TPR * VEC
    C = num_chunks(cfg, N)
    C_FULL = N // SPAN  # chunks that are in bounds for every thread
    U = cfg.unroll
    l2e = T.float32(LOG2E)
    full_mask = tir.const(0xFFFFFFFF, "uint32")
    f32 = lambda v: T.Cast("float32", v)  # noqa: E731

    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("main")
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            x = T.arg("x", T.handle())
            w = T.arg("w", T.handle()) if op == "rmsnorm" else None
            y = T.arg("y", T.handle())
            X2 = IRBuilder.name("X", T.match_buffer(x, (M, N), dtype))
            Wt = IRBuilder.name("W", T.match_buffer(w, (N,), dtype)) if w is not None else None
            Y2 = IRBuilder.name("Y", T.match_buffer(y, (M, N), dtype))

            bx = T.env_thread("blockIdx.x")
            tx = T.env_thread("threadIdx.x")
            with T.launch_thread(bx, M // RPB), T.launch_thread(tx, NT):
                t = tx % TPR
                rb = tx // TPR
                lane = tx % 32
                wr = t // 32
                row_base = (bx * RPB + rb) * N

                with ExitStack() as stack:

                    def local(name, n, dt):
                        buf = stack.enter_context(T.decl_buffer((n,), dt, scope="local"))
                        return IRBuilder.name(name, buf)

                    def flat(name, buf):
                        return IRBuilder.name(
                            name,
                            stack.enter_context(T.decl_buffer((M * N,), dtype, data=buf.data)),
                        )

                    X = flat("X_flat", X2)
                    Y = flat("Y_flat", Y2)

                    red = None
                    if NW > 1:
                        red = IRBuilder.name(
                            "red",
                            stack.enter_context(
                                T.decl_buffer((2 * RPB * NW,), "float32", scope="shared")
                            ),
                        )
                    cache = local("xh", (C if cfg.mode == "reg" else U) * VEC, dtype)
                    oh = local("oh", VEC, dtype)
                    wh = local("wh", VEC, dtype) if op == "rmsnorm" else None
                    tmp = local("tmp", 8, "float32")

                    def store(buf, idx, val):
                        T.buffer_store(buf, val, [idx])

                    def col(c):
                        return (c * TPR + t) * VEC

                    def is_tail(c):
                        return isinstance(c, int) and c >= C_FULL

                    # 32-bit words per access of the hinted helpers in kernels.build
                    words = VEC * 2 // 4

                    def load(c, slot, keep=False):
                        """`keep`: the row is read again soon (twopass pass 1)."""

                        def body():
                            if cfg.hint:
                                T.evaluate(T.call_extern(
                                    "void", f"ld_hint_{'last' if keep else 'first'}_v{words}",
                                    T.address_of(cache[slot * VEC]),
                                    T.address_of(X[row_base + col(c)]),
                                ))
                            else:
                                T.buffer_store(
                                    cache,
                                    X[tir.Ramp(row_base + col(c), 1, VEC)],
                                    [tir.Ramp(slot * VEC, 1, VEC)],
                                )

                        if is_tail(c):
                            with T.If(col(c) < N):
                                with T.Then():
                                    body()
                        else:
                            body()

                    def values(c, slot, neutral):
                        """fp32 values of a cached chunk; out-of-row lanes read as `neutral`."""
                        vals = [f32(cache[slot * VEC + i]) for i in range(VEC)]
                        if is_tail(c):
                            valid = col(c) < N
                            vals = [T.Select(valid, v, T.float32(neutral)) for v in vals]
                        return vals

                    def tree(vals, fn):
                        while len(vals) > 1:
                            nxt = [fn(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                            if len(vals) % 2:
                                nxt.append(vals[-1])
                            vals = nxt
                        return vals[0]

                    def emit_out(c, slot, fn):
                        """Writes chunk `c` with y_i = fn(x_i (fp32), i)."""

                        def body():
                            if wh is not None:
                                T.buffer_store(
                                    wh, Wt[tir.Ramp(col(c), 1, VEC)], [tir.Ramp(0, 1, VEC)]
                                )
                            for i in range(VEC):
                                store(oh, i, fn(f32(cache[slot * VEC + i]), i))
                            if cfg.hint:
                                T.evaluate(T.call_extern(
                                    "void", f"st_stream_v{words}",
                                    T.address_of(Y[row_base + col(c)]), T.address_of(oh[0]),
                                ))
                            else:
                                T.buffer_store(
                                    Y, oh[tir.Ramp(0, 1, VEC)],
                                    [tir.Ramp(row_base + col(c), 1, VEC)],
                                )

                        if is_tail(c):
                            with T.If(col(c) < N):
                                with T.Then():
                                    body()
                        else:
                            body()

                    def row_reduce(val, fn, region, dst):
                        """tmp[dst] = fn-reduction of `val` over all threads of the row."""
                        store(tmp, dst, val)
                        for m in (16, 8, 4, 2, 1):
                            store(
                                tmp, dst,
                                fn(tmp[dst], T.call_extern(
                                    "float32", "__shfl_xor_sync", full_mask, tmp[dst], m, 32
                                )),
                            )
                        if NW > 1:
                            base = region * RPB * NW + rb * NW
                            with T.If(lane == 0):
                                with T.Then():
                                    store(red, base + wr, tmp[dst])
                            T.evaluate(T.tvm_storage_sync("shared"))
                            store(tmp, dst, tree([red[base + i] for i in range(NW)], fn))

                    add = lambda a, b: a + b  # noqa: E731

                    def chunk_groups():
                        """(python chunk list | loop) layout of the streaming passes."""
                        n_loop = C_FULL // U
                        rest = list(range(n_loop * U, C))
                        return n_loop, rest

                    def stream(per_group):
                        """Runs `per_group(chunks)` over all chunks, U at a time."""
                        n_loop, rest = chunk_groups()
                        if n_loop > 0:
                            with T.serial(0, n_loop) as g:
                                per_group([g * U + u for u in range(U)])
                        for c in rest:
                            per_group([c])

                    if op == "softmax":
                        if cfg.mode == "reg":
                            for c in range(C):
                                load(c, c)
                            row_reduce(
                                tree(sum((values(c, c, NEG_INF) for c in range(C)), []), T.max),
                                T.max, 0, 1,
                            )
                            store(tmp, 1, tmp[1] * l2e)
                            row_reduce(
                                tree(
                                    [T.exp2(v * l2e - tmp[1])
                                     for c in range(C) for v in values(c, c, NEG_INF)],
                                    add,
                                ),
                                add, 1, 2,
                            )
                            store(tmp, 2, T.float32(1.0) / tmp[2])
                            for c in range(C):
                                emit_out(
                                    c, c,
                                    lambda v, i: T.Cast(dtype, T.exp2(v * l2e - tmp[1]) * tmp[2]),
                                )
                        else:
                            # tmp[0]: running max (scaled by log2e), tmp[3]: running sum
                            store(tmp, 0, T.float32(NEG_INF))
                            store(tmp, 3, T.float32(0))

                            def online(chunks):
                                for s, c in enumerate(chunks):
                                    load(c, s, keep=True)
                                vals = [
                                    v * l2e
                                    for s, c in enumerate(chunks)
                                    for v in values(c, s, NEG_INF)
                                ]
                                store(tmp, 4, T.max(tmp[0], tree(list(vals), T.max)))
                                store(
                                    tmp, 3,
                                    tmp[3] * T.exp2(tmp[0] - tmp[4])
                                    + tree([T.exp2(v - tmp[4]) for v in vals], add),
                                )
                                store(tmp, 0, tmp[4])

                            stream(online)
                            row_reduce(tmp[0], T.max, 0, 1)
                            row_reduce(tmp[3] * T.exp2(tmp[0] - tmp[1]), add, 1, 2)
                            store(tmp, 2, T.float32(1.0) / tmp[2])

                            def write(chunks):
                                for s, c in enumerate(chunks):
                                    load(c, s)
                                for s, c in enumerate(chunks):
                                    emit_out(
                                        c, s,
                                        lambda v, i: T.Cast(
                                            dtype, T.exp2(v * l2e - tmp[1]) * tmp[2]
                                        ),
                                    )

                            stream(write)
                    else:
                        # Same rounding as torch: (x * r) is rounded to the input
                        # dtype before the multiplication by the weight.
                        def rms_out(v, i):
                            normed = f32(T.Cast(dtype, v * tmp[1]))
                            return T.Cast(dtype, normed * f32(wh[i]))

                        def finish(sumsq_idx):
                            store(
                                tmp, 1,
                                T.rsqrt(tmp[sumsq_idx] * T.float32(1.0 / N) + T.float32(eps)),
                            )

                        if cfg.mode == "reg":
                            for c in range(C):
                                load(c, c)
                            row_reduce(
                                tree(
                                    [v * v for c in range(C) for v in values(c, c, 0.0)], add
                                ),
                                add, 0, 2,
                            )
                            finish(2)
                            for c in range(C):
                                emit_out(c, c, rms_out)
                        else:
                            store(tmp, 3, T.float32(0))

                            def sumsq(chunks):
                                for s, c in enumerate(chunks):
                                    load(c, s, keep=True)
                                store(
                                    tmp, 3,
                                    tmp[3] + tree(
                                        [v * v for s, c in enumerate(chunks)
                                         for v in values(c, s, 0.0)],
                                        add,
                                    ),
                                )

                            stream(sumsq)
                            row_reduce(tmp[3], add, 0, 2)
                            finish(2)

                            def write(chunks):
                                for s, c in enumerate(chunks):
                                    load(c, s)
                                for s, c in enumerate(chunks):
                                    emit_out(c, s, rms_out)

                            stream(write)

    return tvm.IRModule({"main": ib.get()})


def build_rowwise(
    mod: tvm.IRModule,
    target: tvm.target.Target | None = None,
    dump_dir: str | None = None,
):
    return build_kernel(mod, target, dump_dir, FAST_MATH_OPTIONS)


def build_rowwise_to_file(
    op: str, shape, dtype: str, cfg: RowConfig, eps: float, arch: str, path: str
) -> str:
    """Builds in a worker process (no CUDA context needed) and exports a shared library."""
    M, N = shape
    mod = make_rowwise(op, M, N, dtype, cfg, eps)
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    build_rowwise(mod, target).export_library(path)
    return path
