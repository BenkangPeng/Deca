from collections import namedtuple
from arch.RTX4090 import RTX4090
dim3 = namedtuple("dim3", ["x", "y", "z"])


def IsComputeIntensive(rtile1, K, arch = RTX4090()):
    """
    return True if the schedule(shared memory level) is compute intensive
    K: int, the number of columns of the matrix A
    """
    compute_workload = rtile1[0] * rtile1[1] * rtile1[2] * (K // rtile1[2])
    compute_throughput = arch.peak_flops  # / self.arch.compute_max_core[0]
    if False: # TODO: add tensor core support
        compute_throughput = arch.peak_tc_flops
    compute_latency = compute_workload / compute_throughput

    memory_workload = (rtile1[0] * rtile1[2] + rtile1[1] * rtile1[2] + rtile1[0] * rtile1[1]) * (K // rtile1[2]) * 4 # 4 bytes for fp32

    memory_throughput = arch.memory_bw(0)
    memory_latency = memory_workload / memory_throughput
    return compute_latency > memory_latency
