"""
Shared compilation helpers for the hand-written TIR kernels.
"""

from __future__ import annotations

from pathlib import Path

import tvm

FAST_MATH_OPTIONS = ["--use_fast_math"]
# Extra nvcc options, only set while one of these kernels is being built so
# that other operators keep TVM's default compilation flags.
_extra_nvcc_options: list[str] = []


def _vec_ld(name, keep, n):
    regs = ",".join(f"%{i}" for i in range(n))
    outs = ", ".join(f'"=r"(r.{c})' for c in "xyzw"[:n])
    if keep:
        # L2 eviction priorities need an explicit cache policy operand.
        body = (
            f'  asm("{{\\n .reg .b64 pol;\\n createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\\n'
            f' ld.global.nc.L2::cache_hint.v{n}.u32 {{{regs}}}, [%{n}], pol;\\n}}"\n'
        )
    else:
        body = f'  asm("ld.global.nc.L1::no_allocate.v{n}.u32 {{{regs}}}, [%{n}];"\n'
    return (
        f"__device__ __forceinline__ void {name}(void* dst, const void* src) {{\n"
        f"  uint{n} r;\n"
        + body
        + f'      : {outs} : "l"(src));\n'
        f"  *(uint{n}*)dst = r;\n}}\n"
    )


def _vec_st(name, n):
    regs = ",".join(f"%{i + 1}" for i in range(n))
    ins = ", ".join(f'"r"(v.{c})' for c in "xyzw"[:n])
    return (
        f"__device__ __forceinline__ void {name}(void* dst, const void* src) {{\n"
        f"  uint{n} v = *(const uint{n}*)src;\n"
        f'  asm volatile("st.global.cs.v{n}.u32 [%0], {{{regs}}};" :: "l"(dst), {ins} : "memory");\n'
        f"}}\n"
    )


# Cache-hinted 8-/16-byte global accesses between a register array and global
# memory, callable from TIR via `call_extern(dst, src)`
# (`ld_hint_{first,last}_v{2,4}` and `st_stream_v{2,4}`).
CACHE_HINT_HELPERS = "".join(
    # Read-once data bypasses L1; data read again soon is kept in L2.
    [_vec_ld(f"ld_hint_{p}_v{n}", p == "last", n) for p in ("first", "last") for n in (2, 4)]
    + [_vec_st(f"st_stream_v{n}", n) for n in (2, 4)]
)


@tvm.register_global_func("tvm_callback_cuda_compile", override=True)
def _tvm_callback_cuda_compile(code):  # pylint: disable=unused-argument
    from tvm.support import nvcc

    if "ld_hint_" in code or "st_stream_" in code:
        # After TVM's own preamble, which defines the vector types.
        pos = code.find("extern \"C\" __global__")
        code = code[:pos] + CACHE_HINT_HELPERS + code[pos:]
    return nvcc.compile_cuda(
        code, target_format="fatbin", options=list(_extra_nvcc_options), compiler="nvcc"
    )


def cuda_target(device_id: int = 0) -> tvm.target.Target:
    dev = tvm.cuda(device_id)
    major, minor = dev.compute_version.split(".")
    return tvm.target.Target({"kind": "cuda", "arch": f"sm_{major}{minor}"})


def build_kernel(
    mod: tvm.IRModule,
    target: tvm.target.Target | None = None,
    dump_dir: str | None = None,
    nvcc_options: list[str] | None = None,
):
    """Compiles a kernel module; `dump_dir` keeps the generated .cu for inspection."""
    target = target or cuda_target()
    _extra_nvcc_options[:] = FAST_MATH_OPTIONS if nvcc_options is None else nvcc_options
    try:
        with target, tvm.transform.PassContext(opt_level=3):
            runtime_mod = tvm.compile(mod, target=target, tir_pipeline="s_tir").mod
        if dump_dir:
            output = Path(dump_dir)
            output.mkdir(parents=True, exist_ok=True)
            for i, device_mod in enumerate(runtime_mod.imports):
                (output / f"kernel_{i}.cu").write_text(device_mod.inspect_source())
        return runtime_mod
    finally:
        _extra_nvcc_options.clear()
