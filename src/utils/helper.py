from collections import namedtuple
from arch.RTX4090 import RTX4090
from arch.H100 import H100
from arch.V100 import V100
import pycuda.driver as cuda
import pycuda.autoinit

dim3 = namedtuple("dim3", ["x", "y", "z"])


def IsComputeIntensive(rtile1, K, arch=RTX4090()):
    """
    return True if the schedule(shared memory level) is compute intensive
    K: int, the number of columns of the matrix A
    """
    compute_workload = rtile1[0] * rtile1[1] * rtile1[2] * (K // rtile1[2])
    compute_throughput = arch.peak_flops  # / self.arch.compute_max_core[0]
    if False:  # TODO: add tensor core support
        compute_throughput = arch.peak_tc_flops
    compute_latency = compute_workload / compute_throughput

    memory_workload = (
        (rtile1[0] * rtile1[2] + rtile1[1] * rtile1[2] + rtile1[0] * rtile1[1])
        * (K // rtile1[2])
        * 4
    )  # 4 bytes for fp32

    memory_throughput = arch.memory_bw(0)
    memory_latency = memory_workload / memory_throughput
    return compute_latency > memory_latency


class GPUDetector:
    """GPU Model Detector"""

    # GPU model mapping table
    GPU_MODEL_MAP = {
        "NVIDIA H100 80GB HBM3": H100,
        "NVIDIA GeForce RTX 4090": RTX4090,
        "Quadro GV100": V100,
    }

    @classmethod
    def detect_gpu(cls, verbose: bool):
        """Detect current GPU model"""
        try:
            device = cuda.Context.get_device()
            device_name = device.name()
            if verbose:
                print(f"Detected GPU: {device_name}")

            # Search for matching GPU in the mapping table
            if device_name in cls.GPU_MODEL_MAP.keys():
                return cls.GPU_MODEL_MAP[device_name]
            else:
                raise NotImplementedError(
                    f"Unsupported GPU model detection: {device_name}, please report to us."
                    f"Or you can insert your own GPU model in the mapping table. But we can "
                    f"only ensure the correct configuration of {cls.GPU_MODEL_MAP.keys()}."
                )

        except Exception as e:
            print(f"GPU detection failed: {e}")
            exit(-1)

    @classmethod
    def create_gpu_instance(cls, verbose=True):
        """Create corresponding GPU class instance"""
        gpu_class = cls.detect_gpu(verbose=verbose)
        return gpu_class()
