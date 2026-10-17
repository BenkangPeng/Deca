Deca generates and tunes CUDA kernels for GEMM, FlashAttention, Softmax, and RMSNorm.
This checkout targets **TVM 0.27** with CUDA and supports NVIDIA H20/H20-3e,
H100, RTX 4090, and the existing V100 architecture model.

Activate the TVM environment and preserve its TVM source path when adding Deca:

```bash
conda activate tvm
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
# If CUDA-enabled PyTorch is not already installed (CUDA 12.9 on this machine):
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-dev.txt
```

TVM must already be built with CUDA enabled; a CUDA toolkit with `nvcc` on PATH
and CUDA-enabled PyTorch compatible with your driver are required. If installing
PyTorch separately, select its CUDA wheel for your driver/toolkit rather than a
CPU-only wheel. See the [official CUDA wheel commands](https://pytorch.org/get-started/previous-versions/). `z3-solver` is required for tile enumeration; pytest is for the
regression suite.

Run the GPU correctness suite:

```bash
python -m pytest -q
```

Run a small tuning job (GPU 0 means the first device visible to the process):

```bash
python test_op.py --shape 1024 1024 256
python test_op.py --shape 1024 1024 256 --use_tc --dtype float16
python test_gemm_tc.py --shape 1024 1024 256 --top_n 2 --jobs 1 --rounds 1
python test_rowwise.py --op softmax --shape 128 1032 --top_n 2 --jobs 1 --rounds 1
python test_flash_attn.py --shape 1 2 4096 64 --top_n 2 --jobs 1 --rounds 1
```

Set `CUDA_VISIBLE_DEVICES` before running to choose a GPU. H20 and H20-3e are
automatically detected and compile for `sm_90`. The H20 model uses 78 SMs and
Hopper's 227 KiB opt-in shared-memory limit. Detection updates the model's SM
count, memory capacity, shared-memory bandwidth and L2 capacity using CUDA.
DRAM bandwidth is modeled as 4000 GB/s for H20 and 4916.736 GB/s for H20-3e
(the installed card reports a 3201 MHz DDR memory clock and 6144-bit bus).
Compute and L2 bandwidth ceilings are analytical estimates for ranking; they
are not measured performance guarantees. V100 retains its existing legacy
WMMA path; the hand-written `mma.sync.m16n8k16`/`cp.async` kernels require SM80+.

The migration uses `tvm.tirx` for IR, `tvm.s_tir` for schedules and MetaSchedule,
`T.sblock` and explicit `s_tir=True` for scheduled functions, and the `s_tir`
lowering pipeline. Runtime arrays use `tvm.runtime.Tensor`/`from_dlpack`;
compilation uses `tvm.compile(...).mod`. CUDA callbacks follow the one-argument
TVM 0.27 interface and use `tvm.support.nvcc`. Existing library export/reload,
CUDA source dumps, and DLPack sharing are covered by the regression tests.

The optional MetaSchedule baseline in `benchmark/tvm/tvm_gemm_tuner.py` also
needs `python -m pip install -r requirements-benchmark.txt`. Its default target
is inferred from the visible GPU; `--target` can override it. Run this optional
baseline with local scratch storage (`TMPDIR=/tmp`) to avoid artifact cleanup
races observed on the machine's shared temporary filesystem.


Validated on the installed TVM 0.27.0 and NVIDIA H20-3e: 29 regression tests
pass, including CUDA-core/WMMA GEMM, FP16/FP32 tensor-core accumulation,
FP16/BF16 Softmax and RMSNorm, causal/non-causal attention, Conv2D IR, CUDA
source dumps, DLPack sharing, and library export/reload. The original GEMM,
GEMMTC, RMSNorm, and FlashAttention tuning CLIs were also exercised, including
spawned compile workers. Tests use short CUDA-event timing; use the CLI scripts
for performance measurements.

If attention reports no valid configurations, the script now includes build and
verification counts and chains the original failure. It also initializes
PyTorch CUDA before compiling. On this machine, PyTorch `2.13.0+cu130` failed
with `cudaErrorInsufficientDriver` although TVM could detect the H20-3e; the
CUDA 12.9 PyTorch build worked. Reinstall from the `cu129` index above, then
check `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`.
Use `--jobs 1` to compile directly in the main process when debugging.
