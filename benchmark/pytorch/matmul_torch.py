import torch
import time
import os

def test_matmul_perf(M, N, K, dtype=torch.float32, iterations=10):
    device = torch.device("cuda")
    
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    # Warm up
    for _ in range(4):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(iterations):
        torch.matmul(a, b)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    flops_per_matmul = 2.0 * M * N * K
    tflops = (flops_per_matmul / avg_time) / 1e12

    log_path = os.path.join(os.path.dirname(__file__), "matmul_torch_performance.log")
    with open(log_path, "a") as f:
        f.write(f"M={M}, N={N}, K={K}\n")
        f.write(f"dtype: {dtype}\n")
        f.write(f"average time: {avg_time*1000:.3f} ms\n")
        f.write(f"performance: {tflops:.2f} TFLOPS\n")
        f.write("-" * 30 + "\n")

if __name__ == "__main__":
    # M N K
    shapes = [
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 4096, 1024),
        (1024, 768, 768),
        (1024, 768, 2304),
        (1024, 768, 3072),
        (1024, 2304, 2304),
        (1024, 3072, 768),
        (1024, 2304, 768),
        # Conv2D
        (512, 1024, 256),
        (1024, 1024, 1024),
        (512, 1024, 4608),
        (1024, 1024, 512),
        (2048, 1024, 512),
        (512, 1024, 1024),
        (2048, 1024, 1024),
        (512, 1024, 2048)
    ]
    for shape in shapes:
        M, N, K = shape
        test_matmul_perf(M, N, K, dtype=torch.float32)

    # FP16 Tensor Core square GEMMs (baseline of test_gemm_tc.py)
    for n in range(2048, 16384 + 1, 2048):
        test_matmul_perf(n, n, n, dtype=torch.float16)

