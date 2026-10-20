This project is built on [TVM commit d4e7bd3e](https://github.com/apache/tvm/commit/d4e7bd3e).

## Supported GPUs

| GPU | arch class | notes |
| --- | --- | --- |
| NVIDIA GeForce RTX 4090 | `arch.RTX4090` | |
| NVIDIA H100 | `arch.H100` | |
| NVIDIA H20 / H20-3e (sm_90) | `arch.H20` | SM count, clock, memory size and L2 size are read from the device at start-up; H20-3e uses the HBM3e bandwidth (4.9 TB/s) |

The GPU is detected from its device name (`src/utils/helper.py`). On a shared
multi-GPU machine pick an idle GPU (check `nvidia-smi`) with:

```bash
conda activate tvm
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx>
```

## Operators

```bash
python test_op.py --shape 4096 4096 4096                  # fp32 CUDA-core GEMM
python test_gemm_tc.py --shape 4096 4096 4096             # fp16 tensor-core GEMM (mma.sync)
python test_gemm_tc.py --sweep                            # square 2K-16K grid
python test_rowwise.py --op softmax --shape 16384 4096    # softmax / rmsnorm, --dtype fp16|bf16
python test_rowwise.py --op softmax --sweep
python test_flash_attn.py --shape 4 16 4096 128           # B H S D, add --causal
python test_flash_attn.py --sweep
```

Each script tunes the operator (Z3 solver + cost model ranking + measurement)
and reports the result against PyTorch; results are appended to
`deca_performance.log`.

## H20-3e results (vs PyTorch 2.13 / CUDA 12.9)

- fp32 GEMM (TF32 off), 2048-8192: 22.9-26.2 TFLOPS, 0.83-0.94x of cuBLAS.
- fp16 tensor-core GEMM, 2048-16384: 75-94 TFLOPS, 0.64-0.77x of cuBLAS.
  The kernel uses `mma.sync`, whose sustained peak on H20 is ~84 TFLOPS;
  cuBLAS uses Hopper `wgmma`.
- Softmax fp16/bf16: 2.0-5.8x of `torch.softmax`; large rows reach
  ~3.8 TB/s (the device-copy bandwidth).
- FlashAttention fp16: D=64 0.99-1.04x, D=128 0.93-0.94x of SDPA-flash;
  causal 0.96-1.10x.
