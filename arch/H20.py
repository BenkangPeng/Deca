from .H100 import H100


class H20(H100):
    """NVIDIA H20 / H20-3e (Hopper, sm_90).

    The H20 is a cut-down H100: the SM itself is identical (227 KB opt-in shared
    memory per block, 64 resident warps, 65536 registers), but only 78 SMs are
    enabled and the tensor-core throughput is capped. `configure_device` then
    specializes the model with the attributes reported by the visible device,
    which also distinguishes the HBM3 (96 GB) and HBM3e (141 GB, "-3e") cards.
    """

    def __init__(self, para_opt=True):
        super().__init__(para_opt=para_opt)
        # Only for tagging; replaced by the reported name in `configure_device`.
        self.name = "NVIDIA H20"
        self.global_mem_capacity = "96GB"

        self._set_sm_count(78)
        # SM boost clock (Hz), used for per-cycle estimates in the cost model.
        self.sm_clock_hz = 1.98e9
        # Shared memory delivers 128 B/clk/SM.
        self.smem_bw_per_sm = 128 * self.sm_clock_hz

        # Peak DRAM bandwidth (GB/s): 4.0 TB/s HBM3 on the 96 GB card.
        self.dram_bw = 4000
        self.bandwidth = [self.dram_bw, self.smem_bw_per_sm * self.sm_nums / 1e9]

        # Compute throughput (GFLOPS). FP32 CUDA cores: 78 SMs x 128 FMA x 2 x 1.98 GHz.
        self.peak_flops = 39500
        # Dense FP16 tensor-core peak of the datasheet (wgmma, FP32 accumulation).
        self.peak_tc_flops = 148000
        self.peak_tc_flops_fp32acc = 148000
        # `mma.sync` (no wgmma) ceiling measured on an idle H20-3e with a
        # register-only m16n8k16 loop: 84 TFLOPS sustained (short bursts reach
        # 90-95), about 0.6x of the wgmma peak. cuBLAS (wgmma) reaches
        # 115-138 TFLOPS on 2K-16K square GEMMs.
        self.peak_mma_sync_flops = 84000
        # The FP16-accumulate MMA rate is not faster than FP32 accumulation on Hopper.
        self.peak_mma_sync_f16acc_flops = self.peak_mma_sync_flops

        # DRAM latency under load used for Little's law in the row-wise cost
        # model: 78 SMs share ~4.9 TB/s, so each SM needs more bytes in flight
        # than on H100; 1.2 us fits the measured Softmax rankings best.
        self.dram_latency_s = 1.2e-6

        self.l2_bytes = 60 * 1024 * 1024
        # Achievable L2 read bandwidth (GB/s), measured on H20-3e with all SMs
        # streaming an L2-resident buffer (ld.global.cg).
        self.l2_bw = 7000
        # Relative slowdown per pipeline stage beyond the latency-hiding need of
        # the FP16 Tensor Core GEMM: the tensor cores are slow relative to the
        # memory system on H20, so two stages suffice and deeper rings only add
        # shared memory and slot arithmetic (measured 3-8% for 3-4 stages).
        self.gemm_excess_stage_penalty = 0.04
        # Cost of the per-KV-step block barriers of FlashAttention relative to
        # the resident CTAs: at D = 128, two 4-warp CTAs per SM beat one 8-warp
        # CTA by 5-8% on H20 at the same number of resident warps.
        self.fa_cta_sync_penalty = 0.05

    def _set_sm_count(self, sm_nums: int):
        self.sm_nums = sm_nums
        self.compute_max_core = [sm_nums]
        self.mem_max_core = [sm_nums]
        self.compute_sm_partition = [sm_nums, 4]
        self.smem_sm_partition = [sm_nums, 4]

    def configure_device(self, device):
        """Specializes the model with the attributes of `device` (a `tvm.cuda` device)."""
        self.name = device.device_name
        self._set_sm_count(device.multi_processor_count)
        self.global_mem_capacity = f"{device.total_global_memory / 2**30:.0f}GiB"
        # `max_clock_rate` is reported in kHz.
        self.sm_clock_hz = device.max_clock_rate * 1e3
        self.smem_bw_per_sm = 128 * self.sm_clock_hz
        self.peak_flops = self.sm_nums * 128 * 2 * self.sm_clock_hz / 1e9
        # H20-3e: HBM3e at 3201 MHz (DDR) over a 6144-bit bus = 4916.7 GB/s.
        self.dram_bw = 4916.736 if "3e" in self.name.lower() else 4000
        self.bandwidth = [self.dram_bw, self.smem_bw_per_sm * self.sm_nums / 1e9]
        self.l2_bytes = device.l2_cache_size_bytes
        return self
