"""
Shared shapes / references for the row-wise operators (Softmax, RMSNorm).

Both operators are memory bound, so they are reported in GB/s of effective
DRAM traffic (read x [+ weight] + write y). Timing reuses the CUDA-event
harness of the attention benchmarks so that DeCA and torch are measured alike.
"""

import torch
import torch.nn.functional as F

from kernels.attn_bench import bench_cuda, bench_interleaved  # noqa: F401  (re-exported)

ROWWISE_OPS = ("softmax", "rmsnorm")
HIDDEN_SIZES = [1024, 2048, 4096, 8192]
HIDDEN_ROWS = [4096, 16384, 65536]
VOCAB_SIZES = [32000, 128256]
VOCAB_ROWS = [1024, 4096, 8192]
RMSNORM_EPS = 1e-6

TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE_ALIASES = {"fp16": "float16", "bf16": "bfloat16", "float16": "float16", "bfloat16": "bfloat16"}


def canonical_dtype(dtype: str) -> str:
    return DTYPE_ALIASES[dtype]


def standard_shapes(op: str):
    """Returns a list of (M, N)."""
    shapes = [(M, N) for N in HIDDEN_SIZES for M in HIDDEN_ROWS]
    if op == "softmax":
        shapes += [(M, N) for N in VOCAB_SIZES for M in VOCAB_ROWS]
    return shapes


def rowwise_bytes(op: str, M: int, N: int, type_size: int = 2) -> int:
    nbytes = 2 * M * N * type_size
    return nbytes + N * type_size if op == "rmsnorm" else nbytes


def make_inputs(op: str, M: int, N: int, dtype: str, seed: int = 0):
    """Returns the kernel inputs: (x,) for softmax, (x, w) for RMSNorm."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tdt = TORCH_DTYPES[canonical_dtype(dtype)]
    x = torch.randn(M, N, device="cuda", dtype=tdt, generator=g)
    if op == "softmax":
        return (x,)
    w = (1.0 + 0.1 * torch.randn(N, device="cuda", generator=g)).to(tdt)
    return (x, w)


def torch_ref(op: str, inputs, eps: float = RMSNORM_EPS):
    """Eager torch baseline (also the correctness reference)."""
    if op == "softmax":
        (x,) = inputs
        return torch.softmax(x, dim=-1)
    x, w = inputs
    return F.rms_norm(x, (x.shape[-1],), w, eps=eps)


def bench_torch(op: str, M: int, N: int, dtype: str):
    """Returns (seconds, GB/s) of the eager torch operator."""
    inputs = make_inputs(op, M, N, dtype)
    t = bench_cuda(lambda: torch_ref(op, inputs))
    return t, rowwise_bytes(op, M, N) / t / 1e9
