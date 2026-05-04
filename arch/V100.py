from .Arch import *


class V100(Arch):
    # compute 7.0
    def __init__(self, para_opt=True):
        super().__init__()
        # Only for tagging.
        self.name = "Quadro GV100"
        self.global_mem_capacity = "32GB"

        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        self.num_level = 2

        # NOTE: The memory interface of V100 (HBM2) is 4096-bit wide.
        #       Volta's highly tuned 16 GB HBM2 memory subsystem delivers
        #       900 GB/sec peak memory bandwidth. Although the white paper
        #       does not give the memory bandwidth of the 32GB version of
        #       the V100 GPU, the V100 product specification indicates that
        #       the memory bandwidth of both versions (16GB and 32GB) of
        #       the V100 is 900GB/s.
        #
        #       The GPU boost clock of V100 is 1530 MHz, and the bank size
        #       of the shared memory is 32 with 32-bit (4 bytes) data width
        #       per bank. In fact, the number of banks per shared memory is
        #       usually designed to 32 for the reason that it could provide
        #       4-byte data for each of the the 32 threads in a warp. So
        #       ecifically, each shared memory can provide 4 bytes * 32 banks
        #       = 128 bytes per cycle. So the memory bandwidth of the shared
        #       memory is 128 bytes * 1530 MHz = 191.25 GB/s, and the 80 SMs
        #       in V100 can provide 191.25 GB/s * 80 = 15300 GB/s.
        # REF: https://www.nvidia.cn/content/dam/en-zz/zh_cn/Solutions/Data-Center/tesla-v100/volta-v100s-datasheet-a4-nvidia-1209899-r1-web-zhcn.pdf
        #      https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
        #
        # bandwidth in GB/second
        self.bandwidth = [900, 15300]
        # compute throughput in GFLOPS
        self.peak_flops = 16076
        self.peak_tc_flops = 128000
        self.limit = []
        # NOTE: The max register number that can be used by a single thread
        #       is 255, but the max number of registers that can be used by a
        #       single block can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
        #       ),
        #       which is 65536 for the V100 GPU.
        self.reg_cap = [65536, 255]
        # NOTE: The max shared memory size that can be used by a single block
        #       can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        #       ),
        #       which is 49152 for the V100 GPU.
        self.smem_cap = [49152]
        # NOTE: There are 80 SMs on the V100 GPU, and each SM has 4 sub-cores.
        self.compute_max_core = [80]
        self.mem_max_core = [80]
        self.para_opt = para_opt

        self.warp_size = 32
        # NOTE: There are 80 SMs on the V100 GPU, and each SM has 4 sub-cores.
        self.compute_sm_partition = [80, 4]
        self.smem_sm_partition = [80, 4]
        self.compute_block_schedule_way = ["warp", "active block"]
        self.smem_block_schedule_way = ["warp", "active block"]
        # transaction size in bytes.
        self.transaction_size = [32]

        # The number of banks per shared memory is usually designed to 32 for
        # the reason that it could provide 4-byte data for each of the the 32
        # threads in a warp.
        self.smem_bank_size = 4
        self.bank_number = 32
        self.compute_capability = "compute_70"

        # for active block estimation
        #
        # not used in latest Roller.
        self.max_active_blocks = 32
        # Each shared memory/L1 cache has 96 KB capacity on V100.
        #
        # not used in latest Roller.
        self.max_smem_usage = 96 * 1024 - 1
        # The max number of threads per SM can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR
        #       ),
        # which is 1024 for the V100 GPU.
        #
        # not used in latest Roller.
        self.max_threads_per_sm = 2048

        self.sm_nums = self.compute_sm_partition[0]
        self.compute_capability = (7, 0)
        self.shared_mem_config_options = [0, 8, 16, 32, 64, 96]
        self.max_resident_blocks_per_sm = self.max_active_blocks
        self.max_resident_warps_per_sm = 64
        self.max_registers_per_sm = self.reg_cap[0]
