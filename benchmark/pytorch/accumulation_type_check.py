
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

M = N = K = 8192
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)

def bench():
    for _ in range(20):
        torch.matmul(A, B, out=C)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    iters = 100
    start.record()
    for _ in range(iters):
        torch.matmul(A, B, out=C)
    end.record()
    end.synchronize()

    ms = start.elapsed_time(end) / iters
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops

if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation

    try:
        for use_fp16_acc in (False, True):
            torch.backends.cuda.matmul.allow_fp16_accumulation = use_fp16_acc
            ms, tflops = bench()
            print(f"FP16 accumulation={use_fp16_acc}: "
                  f"{ms:.3f} ms, {tflops:.1f} TFLOPS")
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old
else:
    print("Current PyTorch version does not have allow_fp16_accumulation switch")
