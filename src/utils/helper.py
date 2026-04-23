import tvm
import math

from collections import namedtuple
from typing import Tuple

from arch.RTX4090 import RTX4090
from arch.H100 import H100
from arch.V100 import V100

import pycuda.driver as cuda
import pycuda.autoinit

from tvm.tir import Schedule
from tvm.tir.schedule import LoopRV, BlockRV


dim3 = namedtuple("dim3", ["x", "y", "z"])


def byte_size(dtype: str) -> int:
    if dtype == "float32":
        return 4
    elif dtype == "float16":
        return 2
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def IsComputeIntensive(rtile1, K, dtype: str, arch=RTX4090()):
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
        * byte_size(dtype)
    )  # 4 bytes for fp32, 2 bytes for fp16

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


def getSpecifiedLoopRVs(
    sch: Schedule, block: BlockRV, iter_type
) -> Tuple[tvm.tir.schedule.LoopRV]:
    """Helper function to get specified loop RVs"""
    alllooprvs = sch.get_loops(block)
    targetlooprvs = []
    itervars = sch.get(block).iter_vars
    for i in range(len(itervars)):
        itervar = itervars[i]
        if itervar.iter_type == iter_type:
            targetlooprvs.append(alllooprvs[len(alllooprvs) - len(itervars) + i])
    return targetlooprvs


def getSpatialLoopRVs(sch: Schedule, block: BlockRV) -> Tuple[tvm.tir.schedule.LoopRV]:
    """Helper function to get spatial LoopRVs"""
    return getSpecifiedLoopRVs(sch, block, iter_type=tvm.tir.IterVar.DataPar)


def getReduceLoopRVs(sch: Schedule, block: BlockRV) -> Tuple[tvm.tir.schedule.LoopRV]:
    """Helper function to get reduce LoopRVs"""
    return getSpecifiedLoopRVs(sch, block, iter_type=tvm.tir.IterVar.CommReduce)


def calculate_factors(sch: Schedule, looprv: LoopRV, factor: int) -> Tuple[int, int]:
    """Helper function to transform old `factor` argument of `sch.split`
    in TVM 0.8 to new `factors` argument of `sch.split` in LatestTVM"""
    return [math.ceil(int(sch.get(looprv).extent) / factor), factor]
