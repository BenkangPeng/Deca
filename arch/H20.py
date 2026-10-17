"""Hopper H20 and H20-3e resource model.

Limits are shared by both variants; GPUDetector fills device-specific capacity,
SM count and clock from CUDA; DRAM bandwidth is selected by the H20 variant.
Compute ceilings are estimates for ranking, not guaranteed measured throughput.
"""

from .H100 import H100


class H20(H100):
    def __init__(self, para_opt=True):
        super().__init__(para_opt=para_opt)
        self.name = "NVIDIA H20"
        self.global_mem_capacity = "96GB"
        self.sm_nums = 78
        self.compute_max_core = [78]
        self.mem_max_core = [78]
        self.compute_sm_partition = [78, 4]
        self.smem_sm_partition = [78, 4]
        self.dram_bw = 4000
        self.smem_bw_per_sm = 128 * 1.98e9
        self.bandwidth = [self.dram_bw, self.smem_bw_per_sm * self.sm_nums / 1e9]
        self.peak_flops = 44000
        self.peak_tc_flops = 148000
        self.peak_tc_flops_fp32acc = 148000
        self.peak_mma_sync_flops = 148000
        self.peak_mma_sync_f16acc_flops = 148000
        self.l2_bytes = 60 * 1024 * 1024
        # An estimate used only by the analytical cost model.
        self.l2_bw = 10000 * 78 / 132

    def configure_device(self, device):
        """Specialize the model using the visible CUDA device (including H20-3e)."""
        self.name = device.device_name
        self.sm_nums = device.multi_processor_count
        self.compute_max_core = [self.sm_nums]
        self.mem_max_core = [self.sm_nums]
        self.compute_sm_partition = [self.sm_nums, 4]
        self.smem_sm_partition = [self.sm_nums, 4]
        self.global_mem_capacity = f"{device.total_global_memory / 2**30:.1f}GiB"
        self.smem_cap = [device.max_shared_memory_per_block]
        self.smem_bw_per_sm = 128 * device.max_clock_rate * 1000
        # H20-3e: CUDA reports 3201 MHz DDR over a 6144-bit bus.
        self.dram_bw = 4916.736 if "3e" in self.name.lower() else 4000
        self.bandwidth = [self.dram_bw, self.smem_bw_per_sm * self.sm_nums / 1e9]
        self.l2_bytes = device.l2_cache_size_bytes
        return self
